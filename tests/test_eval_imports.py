"""Import boundary tests for evaluators."""

from __future__ import annotations

import ast
from pathlib import Path

import liteness.eval.evaluators as evaluators_pkg

FORBIDDEN_EVALUATOR_IMPORTS = (
    "liteness.loop",
    "liteness.harness",
    "liteness.replay",
    "liteness.providers",
    "liteness.policies",
    "liteness.clickhouse",
)


def _module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_evaluators_do_not_import_runtime_modules() -> None:
    evaluators_dir = Path(evaluators_pkg.__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(evaluators_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _module_names(tree):
            for forbidden in FORBIDDEN_EVALUATOR_IMPORTS:
                if module == forbidden or module.startswith(forbidden + "."):
                    violations.append(f"{path.name}: {module}")
    assert violations == [], f"forbidden imports in evaluators/: {violations}"
