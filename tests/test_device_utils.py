import sys
import types

import pytest  # type: ignore[import-not-found]

import device_utils


class FakeCudaProps:
	def __init__(self, name: str, total_memory: int, major: int, minor: int) -> None:
		self.name = name
		self.total_memory = total_memory
		self.major = major
		self.minor = minor


class FakeCudaModule:
	def __init__(self, available: bool, props_list: list[FakeCudaProps]) -> None:
		self._available = available
		self._props = props_list

	def is_available(self) -> bool:
		return self._available

	def device_count(self) -> int:
		return len(self._props) if self._available else 0

	def get_device_name(self, idx: int) -> str:
		return self._props[idx].name

	def get_device_properties(self, idx: int) -> FakeCudaProps:
		if not self._available:
			raise RuntimeError("CUDA unavailable")
		return self._props[idx]


def install_fake_torch(
	monkeypatch,
	*,
	available: bool,
	total_memory_gib: float,
	capability: tuple[int, int] = (8, 0),
) -> None:
	props = FakeCudaProps(
		name="Test GPU",
		total_memory=int(total_memory_gib * 1024**3),
		major=capability[0],
		minor=capability[1],
	)
	fake_torch = types.ModuleType("torch")
	fake_torch.cuda = FakeCudaModule(available=available, props_list=[props])
	fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
	fake_torch.version = types.SimpleNamespace(cuda="12.1")

	monkeypatch.setitem(sys.modules, "torch", fake_torch)


@pytest.fixture(autouse=True)
def clear_torch_module():
	original = sys.modules.pop("torch", None)
	try:
		yield
	finally:
		if original is not None:
			sys.modules["torch"] = original


def test_auto_selects_gpu_with_enough_vram(monkeypatch) -> None:
	install_fake_torch(monkeypatch, available=True, total_memory_gib=12)
	monkeypatch.setattr(device_utils, "_probe_nvidia_smi", lambda: None)

	info = device_utils.select_torch_device(preferred="auto", model_name="large-v3", compute_type="auto")

	assert info.backend == "cuda"
	assert info.resolved_compute_type == "float16"
	assert info.messages and info.messages[0] == "Detected Test GPU GPU"
	assert not info.issues


def test_auto_falls_back_when_vram_low(monkeypatch) -> None:
	install_fake_torch(monkeypatch, available=True, total_memory_gib=2)
	monkeypatch.setattr(device_utils, "_probe_nvidia_smi", lambda: None)

	info = device_utils.select_torch_device(preferred="auto", model_name="large-v3", compute_type="auto")

	assert info.backend == "cpu"
	assert any("VRAM too low" in issue for issue in info.issues)
	assert any("lowering the compute type" in issue for issue in info.issues)
	assert info.messages and info.messages[0] == "Detected Test GPU GPU"


def test_fp16_falls_back_when_gpu_does_not_support(monkeypatch) -> None:
	install_fake_torch(monkeypatch, available=True, total_memory_gib=12, capability=(5, 0))
	monkeypatch.setattr(device_utils, "_probe_nvidia_smi", lambda: None)

	info = device_utils.select_torch_device(preferred="auto", model_name="small", compute_type="auto")

	assert info.backend == "cuda"
	assert info.resolved_compute_type == "float16"
	assert any("FP16 may be unsupported" in issue for issue in info.issues)


def test_lower_compute_type_reduces_vram_requirement(monkeypatch) -> None:
	install_fake_torch(monkeypatch, available=True, total_memory_gib=5)
	monkeypatch.setattr(device_utils, "_probe_nvidia_smi", lambda: None)

	info = device_utils.select_torch_device(preferred="auto", model_name="large-v3", compute_type="int8_float16")

	assert info.backend == "cuda"
	assert info.resolved_compute_type == "int8_float16"
	assert not any("VRAM too low" in issue for issue in info.issues)
