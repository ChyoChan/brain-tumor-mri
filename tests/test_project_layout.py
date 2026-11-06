"""Check the entry points that must survive a package/folder reorganization."""

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from brain_mri.utils import project_root

ROOT = Path(__file__).resolve().parents[1]


def test_paths_still_resolve_from_the_core_package():
    assert project_root() == ROOT
    assert (project_root() / "configs/pipeline_config.yaml").is_file()


@pytest.mark.parametrize("script", [
    "download_checkpoints.py", "eval_classification.py", "eval_segmentation.py",
    "prepare_classification_split.py", "run_frontend.py", "stop_frontend.py",
    "train_classification.py", "train_segmentation.py",
])
def test_cli_help_works_outside_the_project(tmp_path, script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("notebook", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook_core_imports_resolve(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(
            "\n" if line.lstrip().startswith(("%", "!")) else line
            for line in cell["source"]
        )
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("brain_mri"):
                module = importlib.import_module(node.module)
                for name in node.names:
                    assert hasattr(module, name.name), (notebook.name, node.module, name.name)
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("brain_mri"):
                        importlib.import_module(name.name)
