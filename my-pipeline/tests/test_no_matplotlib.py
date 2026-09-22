from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = [PROJECT_ROOT / "app.py", *PROJECT_ROOT.joinpath("src").rglob("*.py")]


def _imports_matplotlib(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib" or alias.name.startswith("matplotlib."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "matplotlib" or (node.module and node.module.startswith("matplotlib.")):
                return True
    return False


def test_project_source_does_not_import_matplotlib() -> None:
    offenders = [path.relative_to(PROJECT_ROOT).as_posix() for path in PYTHON_FILES if _imports_matplotlib(path)]
    assert offenders == []


def test_project_dependencies_do_not_declare_matplotlib() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    uv_lock = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "\nmatplotlib" not in requirements
    assert '"matplotlib' not in pyproject
    assert '\nname = "matplotlib"\n' not in uv_lock
