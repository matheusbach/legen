from pathlib import Path


def test_numba_constraint_supports_python_312_whisper_runtime():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert '"numba>=0.65.1,<0.66"' in pyproject
    assert "numba>=0.65.1,<0.66" in requirements.splitlines()
