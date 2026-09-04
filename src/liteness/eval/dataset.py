"""Load eval suite YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from liteness.eval.errors import EvalLoadError
from liteness.eval.models import EvalSuite


def load_suite(path: Path | str) -> EvalSuite:
    """Load an eval suite from a YAML file."""
    suite_path = Path(path)
    if not suite_path.exists():
        raise EvalLoadError(f"eval suite not found: {suite_path}")
    try:
        data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvalLoadError(f"invalid YAML in {suite_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalLoadError(f"eval suite must be a mapping: {suite_path}")
    return EvalSuite.from_dict(data, base_dir=suite_path.parent)
