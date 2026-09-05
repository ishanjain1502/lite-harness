"""Load optional .env files into os.environ (without overriding existing vars)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None, *, override: bool = False) -> bool:
    """Parse a dotenv file and set unset environment variables.

    Returns True if a file was found and read. Existing process env wins by
    default so shell exports and CI secrets take precedence over .env.
    """
    if path is None:
        explicit = os.environ.get("LITENESS_ENV_FILE")
        if explicit:
            path = Path(explicit)
        else:
            path = Path.cwd() / ".env"

    if not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return True
