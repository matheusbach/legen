import json
from pathlib import Path

import pytest


def test_numba_constraint_supports_python_312_whisper_runtime():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert '"numba>=0.65.1,<0.66"' in pyproject
    assert "numba>=0.65.1,<0.66" in requirements.splitlines()


@pytest.mark.parametrize("notebook_name", ["legen.ipynb", "legen-beta.ipynb"])
def test_colab_notebooks_allow_nltk_downloads_through_colab_proxy(notebook_name):
    notebook = json.loads(Path(notebook_name).read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )

    assert 'os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"' in source
