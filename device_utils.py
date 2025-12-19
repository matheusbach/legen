"""Device detection utilities.

Provides a robust way to detect available accelerators (CUDA, MPS) and returns
useful metadata that can be used to pick the best compute backend.

When possible we rely on PyTorch for accurate information, falling back to
``nvidia-smi`` for a lightweight probe so that we can still inform the user
about available GPUs even when PyTorch is not ready to use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import importlib
import math
import shutil
import subprocess
import warnings
import sys
from typing import List, Optional, Tuple


_GIB = 1024 ** 3

_MODEL_VRAM_REQUIREMENTS_GB = {
	"tiny": {"int8": 0.4, "int8_float16": 0.6, "float16": 1.0, "float32": 1.8},
	"tiny.en": {"int8": 0.4, "int8_float16": 0.6, "float16": 1.0, "float32": 1.8},
	"base": {"int8": 0.5, "int8_float16": 0.8, "float16": 1.1, "float32": 2.0},
	"base.en": {"int8": 0.5, "int8_float16": 0.8, "float16": 1.1, "float32": 2.0},
	"small": {"int8": 0.9, "int8_float16": 1.3, "float16": 2.0, "float32": 3.5},
	"small.en": {"int8": 0.9, "int8_float16": 1.3, "float16": 2.0, "float32": 3.5},
	"medium": {"int8": 2.2, "int8_float16": 3.0, "float16": 5.0, "float32": 9.0},
	"medium.en": {"int8": 2.2, "int8_float16": 3.0, "float16": 5.0, "float32": 9.0},
	"large": {"int8": 3.5, "int8_float16": 4.5, "float16": 10.0, "float32": 18.0},
	"large-v1": {"int8": 3.5, "int8_float16": 4.5, "float16": 10.0, "float32": 18.0},
	"large-v2": {"int8": 3.5, "int8_float16": 4.5, "float16": 10.0, "float32": 18.0},
	"large-v3": {"int8": 3.5, "int8_float16": 4.5, "float16": 10.0, "float32": 18.0},
	"large-v3-turbo": {"int8": 2.0, "int8_float16": 2.8, "float16": 6.0, "float32": 10.0},
	"turbo": {"int8": 2.0, "int8_float16": 2.8, "float16": 6.0, "float32": 10.0},
	"distil-large-v2": {"int8": 2.0, "int8_float16": 2.8, "float16": 6.0, "float32": 10.0},
	"distil-medium.en": {"int8": 1.1, "int8_float16": 1.6, "float16": 3.0, "float32": 5.5},
	"distil-small.en": {"int8": 0.5, "int8_float16": 0.8, "float16": 1.5, "float32": 2.5},
}
_DEFAULT_MODEL_VRAM_GB = {
	"int8": 1.0,
	"int8_float16": 1.5,
	"float16": 6.0,
	"float32": 10.0,
}
_FALLBACK_MODEL_VRAM_GB = 6.0

_GPU_ONLY_COMPUTE_TYPES = {"float16", "fp16", "bfloat16", "int8_float16", "int8_bfloat16"}
_FP16_COMPUTE_TYPES = {"float16", "fp16", "bfloat16", "int8_float16", "int8_bfloat16"}

_COMPUTE_CANONICAL = {
	"int8": "int8",
	"int8_float16": "int8_float16",
	"int8_bfloat16": "int8_float16",
	"int8_float32": "float32",
	"float16": "float16",
	"fp16": "float16",
	"bfloat16": "float16",
	"float32": "float32",
	"int16": "float32",
	"default": "float16",
	"auto": "float16",
}

_TORCH_WARNINGS_CONFIGURED = False


@dataclass
class DeviceInfo:
	"""Structured information about the selected compute backend."""

	backend: str
	n_gpus: int = 0
	gpu_names: List[str] = field(default_factory=list)
	gpu_vram_bytes: List[int] = field(default_factory=list)
	gpu_capabilities: List[Tuple[int, int]] = field(default_factory=list)
	cuda_version: Optional[str] = None
	driver_version: Optional[str] = None
	messages: List[str] = field(default_factory=list)
	issues: List[str] = field(default_factory=list)
	notes: List[str] = field(default_factory=list)
	resolved_compute_type: Optional[str] = None
	selected_gpu_index: Optional[int] = None

	def primary_gpu_name(self) -> Optional[str]:
		if self.selected_gpu_index is not None and 0 <= self.selected_gpu_index < len(self.gpu_names):
			return self.gpu_names[self.selected_gpu_index]
		return self.gpu_names[0] if self.gpu_names else None


def _format_gib(byte_count: int | float | None) -> str:
	if byte_count is None:
		return "unknown"
	return f"{byte_count / _GIB:.1f} GiB"


def _normalize_model_name(model_name: Optional[str]) -> str:
	if not model_name:
		return ""
	return str(model_name).strip().lower()



def _canonical_compute(compute_type: Optional[str]) -> str:
	key = (compute_type or "float16").lower()
	return _COMPUTE_CANONICAL.get(key, "float16")


def _estimate_required_vram_bytes(model_name: Optional[str], compute_type: Optional[str]) -> int:
	normalized = _normalize_model_name(model_name)
	canonical = _canonical_compute(compute_type)
	model_table = _MODEL_VRAM_REQUIREMENTS_GB.get(normalized)
	if model_table is not None:
		requirement_gb = model_table.get(canonical)
		if requirement_gb is None:
			requirement_gb = _DEFAULT_MODEL_VRAM_GB.get(canonical, _FALLBACK_MODEL_VRAM_GB)
	else:
		requirement_gb = _DEFAULT_MODEL_VRAM_GB.get(canonical, _FALLBACK_MODEL_VRAM_GB)
	return int(math.ceil(requirement_gb * _GIB))


def _gpu_supports_fp16(capability: Optional[Tuple[int, int]]) -> bool:
	if capability is None:
		return False
	major, minor = capability
	if major is None or minor is None:
		return False
	return (major > 5) or (major == 5 and minor >= 3)


def _probe_nvidia_smi() -> Optional[DeviceInfo]:
	"""Try to query ``nvidia-smi`` for GPU information."""

	if shutil.which("nvidia-smi") is None:
		return None

	try:
		out = subprocess.check_output(
			[
				"nvidia-smi",
				"--query-gpu=name,memory.total,driver_version",
				"--format=csv,noheader,nounits",
			],
			text=True,
			stderr=subprocess.DEVNULL,
		)
	except Exception:
		return None

	lines = [line.strip() for line in out.splitlines() if line.strip()]
	if not lines:
		return None

	gpu_names: List[str] = []
	gpu_vram: List[int] = []
	driver_version: Optional[str] = None

	for line in lines:
		parts = [part.strip() for part in line.split(",")]
		if not parts:
			continue
		gpu_names.append(parts[0])
		if len(parts) > 1:
			try:
				gpu_vram.append(int(float(parts[1])) * 1024 * 1024)
			except (TypeError, ValueError):
				gpu_vram.append(0)
		if len(parts) > 2 and driver_version is None:
			driver_version = parts[2]

	return DeviceInfo(
		backend="cuda",
		n_gpus=len(gpu_names),
		gpu_names=gpu_names,
		gpu_vram_bytes=gpu_vram,
		driver_version=driver_version,
	)


def _suppress_known_torch_warnings() -> None:
	"""Silence noisy torch.cuda capability warnings on older GPUs."""

	global _TORCH_WARNINGS_CONFIGURED
	if _TORCH_WARNINGS_CONFIGURED:
		return

	patterns = [
		r"torch\.cuda",
		r"torch\._C",
	]
	for module_pattern in patterns:
		warnings.filterwarnings(
			"ignore",
			category=UserWarning,
			module=module_pattern,
		)

	message_patterns = [
		r"Found GPU\d+ .*cuda capability",
		r"Please install PyTorch with a following CUDA",
		r"not compatible with the current PyTorch installation",
	]
	for message_pattern in message_patterns:
		warnings.filterwarnings(
			"ignore",
			category=UserWarning,
			message=message_pattern,
		)

	_TORCH_WARNINGS_CONFIGURED = True


@contextmanager
def _suppress_torch_cuda_calls():
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			category=UserWarning,
			module=r"torch\.cuda",
		)
		warnings.filterwarnings(
			"ignore",
			category=UserWarning,
			module=r"torch\._C",
		)
		yield


def _load_torch_module():
	existing = sys.modules.get("torch")
	if existing is not None:
		return existing

	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		module = importlib.import_module("torch")

	for warning_msg in caught:
		filename = getattr(warning_msg, "filename", "") or ""
		normalized = filename.replace("\\", "/")
		if isinstance(warning_msg.message, UserWarning) and "torch/cuda" in normalized:
			continue
		warnings.showwarning(
			warning_msg.message,
			warning_msg.category,
			warning_msg.filename,
			warning_msg.lineno,
		)

	return module


def _resolve_compute_type(
	backend: str,
	requested: Optional[str],
	supports_fp16: bool,
	auto_mode: bool,
) -> Tuple[str, List[str]]:
	req = (requested or "auto").lower()
	issues: List[str] = []

	if req in {"auto", "default"}:
		if backend == "cuda":
			if not supports_fp16:
				issues.append("FP16 may be unsupported on this GPU; it could run slower or fail. Consider int8_float16 if issues occur.")
			return "float16", issues
		if backend == "mps":
			return "float16", issues
		return "float32", issues

	if backend == "cpu" and req in _GPU_ONLY_COMPUTE_TYPES:
		replacement = "float32" if auto_mode else req
		issues.append(f"{req} requires a GPU; using {replacement}.")
		return replacement, issues

	if backend in {"cuda", "mps"} and req in _FP16_COMPUTE_TYPES and not supports_fp16:
		issues.append(f"{req} may be unsupported on detected GPU; it could run slower or fail. Consider int8_float16 or float32 if problems occur.")
		return req, issues

	return req, issues


def select_torch_device(
	preferred: str = "auto",
	*,
	model_name: Optional[str] = None,
	compute_type: Optional[str] = None,
) -> DeviceInfo:
	"""Select and validate the best compute device."""

	pref = (preferred or "auto").lower()
	auto_mode = pref == "auto"
	requested_compute = (compute_type or "auto").lower()

	info = DeviceInfo(backend="cpu")

	torch_module = None
	torch_import_error = None
	try:
		_suppress_known_torch_warnings()
		torch_module = _load_torch_module()
	except Exception as exc:  # pragma: no cover - depends on environment
		torch_import_error = exc

	cuda_available = False
	cuda_device_count = 0
	cuda_names: List[str] = []
	cuda_vram: List[int] = []
	cuda_capabilities: List[Tuple[int, int]] = []

	if torch_module is not None and getattr(torch_module, "cuda", None) is not None:
		try:
			with _suppress_torch_cuda_calls():
				cuda_available = bool(torch_module.cuda.is_available())
		except Exception:
			cuda_available = False

		try:
			with _suppress_torch_cuda_calls():
				cuda_device_count = int(torch_module.cuda.device_count())
		except Exception:
			cuda_device_count = 0

		if cuda_available and cuda_device_count:
			for idx in range(cuda_device_count):
				name = None
				total_mem = None
				capability = None
				try:
					with _suppress_torch_cuda_calls():
						props = torch_module.cuda.get_device_properties(idx)
				except Exception:
					props = None

				if props is not None:
					name = getattr(props, "name", None)
					total_mem = getattr(props, "total_memory", None)
					capability = (
						getattr(props, "major", None),
						getattr(props, "minor", None),
					)

				if name is None:
					try:
						name = torch_module.cuda.get_device_name(idx)
					except Exception:
						name = f"CUDA GPU {idx}"

				cuda_names.append(str(name))
				cuda_vram.append(int(total_mem) if total_mem is not None else 0)
				if capability is not None and capability[0] is not None and capability[1] is not None:
					cuda_capabilities.append((int(capability[0]), int(capability[1])))
				else:
					cuda_capabilities.append((0, 0))

	mps_available = False
	if torch_module is not None:
		mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
		if mps_backend is not None and hasattr(mps_backend, "is_available"):
			try:
				mps_available = bool(mps_backend.is_available())
			except Exception:
				mps_available = False

	smi_info = _probe_nvidia_smi()
	if not cuda_names and smi_info is not None:
		cuda_names = smi_info.gpu_names
		cuda_vram = smi_info.gpu_vram_bytes
		info.driver_version = smi_info.driver_version
		info.n_gpus = smi_info.n_gpus

	if torch_module is not None:
		info.cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)

	if cuda_available and cuda_device_count:
		info.backend = "cuda"
		info.n_gpus = cuda_device_count
		info.gpu_names = cuda_names
		info.gpu_vram_bytes = cuda_vram
		info.gpu_capabilities = cuda_capabilities
		info.selected_gpu_index = 0 if cuda_device_count else None
	elif pref == "cuda":
		info.backend = "cpu"
		info.issues.append("CUDA backend unavailable in PyTorch; using CPU.")
		if torch_import_error is not None:
			info.notes.append(f"PyTorch import failed: {torch_import_error}")
	elif pref == "mps":
		if mps_available:
			info.backend = "mps"
		else:
			info.backend = "cpu"
			info.issues.append("MPS backend unavailable; using CPU.")
	elif pref == "rocm":
		info.backend = "cpu"
		info.issues.append("ROCm backend not implemented; using CPU.")
	elif pref == "cpu":
		info.backend = "cpu"
	else:  # auto mode
		if cuda_available and cuda_device_count:
			info.backend = "cuda"
			info.n_gpus = cuda_device_count
			info.gpu_names = cuda_names
			info.gpu_vram_bytes = cuda_vram
			info.gpu_capabilities = cuda_capabilities
			info.selected_gpu_index = 0 if cuda_device_count else None
		elif mps_available:
			info.backend = "mps"
		else:
			info.backend = "cpu"

	primary_gpu = info.primary_gpu_name()
	if primary_gpu is not None:
		info.messages.append(f"Detected {primary_gpu} GPU")

	if info.backend == "cuda" and primary_gpu is None and cuda_names:
		info.messages.append(f"Detected {cuda_names[0]} GPU")

	initial_backend_for_compute = info.backend

	available_vram = None
	if initial_backend_for_compute == "cuda" and info.selected_gpu_index is not None:
		if 0 <= info.selected_gpu_index < len(info.gpu_vram_bytes):
			available_vram = info.gpu_vram_bytes[info.selected_gpu_index]
	if available_vram is None and initial_backend_for_compute == "cuda" and info.gpu_vram_bytes:
		available_vram = info.gpu_vram_bytes[0]

	supports_fp16 = False
	if initial_backend_for_compute == "cuda" and info.selected_gpu_index is not None:
		idx = info.selected_gpu_index
		capability = None
		if 0 <= idx < len(info.gpu_capabilities):
			capability = info.gpu_capabilities[idx]
		supports_fp16 = _gpu_supports_fp16(capability)
	elif initial_backend_for_compute == "mps":
		supports_fp16 = True

	resolved_compute_candidate, compute_issues = _resolve_compute_type(
		initial_backend_for_compute,
		requested_compute,
		supports_fp16,
		auto_mode,
	)

	requirement_bytes = None
	compute_label = _canonical_compute(resolved_compute_candidate)
	if initial_backend_for_compute == "cuda":
		requirement_bytes = _estimate_required_vram_bytes(model_name, resolved_compute_candidate)

	if (
		initial_backend_for_compute == "cuda"
		and available_vram is not None
		and requirement_bytes is not None
		and available_vram < requirement_bytes
	):
		message = (
			f"VRAM too low for model {model_name or 'selected model'} using compute type {compute_label} (~{_format_gib(requirement_bytes)} required, found {_format_gib(available_vram)})."
		)
		if auto_mode:
			message += " Falling back to CPU."
		else:
			message += " GPU execution may fail."
		message += " Consider lowering the compute type (e.g. --transcription_compute_type=int8_float16) or selecting a smaller model via --transcription_model."
		if auto_mode:
			message += " To force GPU usage, rerun with --transcription_device=cuda."
		info.issues.append(message)
		if auto_mode:
			info.backend = "cpu"
			info.selected_gpu_index = None

	if info.backend != "cuda" and auto_mode and not primary_gpu and smi_info is not None and smi_info.gpu_names:
		info.messages.append("No compatible GPU ready; using CPU.")

	if info.backend in {"cuda", "mps"}:
		info.resolved_compute_type = resolved_compute_candidate
		info.issues.extend(compute_issues)
	else:
		info.resolved_compute_type = None

	if info.backend == "cpu" and primary_gpu is None and not info.messages:
		info.messages.append("Using CPU for transcription.")

	if (
		info.backend == "cuda"
		and not supports_fp16
		and info.resolved_compute_type == "float32"
	):
		info.notes.append("Consider reinstalling PyTorch with newer CUDA support for FP16 acceleration.")

	return info


def select_torch_device_str(
	preferred: str = "auto",
	*,
	model_name: Optional[str] = None,
	compute_type: Optional[str] = None,
) -> str:
	"""Compatibility helper returning just the backend string."""

	info = select_torch_device(preferred=preferred, model_name=model_name, compute_type=compute_type)
	return info.backend
