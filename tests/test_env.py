from __future__ import annotations

import os
from pathlib import Path

from liteness.env import load_dotenv


def test_load_dotenv_sets_unset_keys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=from-file\n# comment\nexport GOOGLE_API_KEY=also-from-file\n",
        encoding="utf-8",
    )
    assert load_dotenv(env_file) is True
    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["GOOGLE_API_KEY"] == "also-from-file"


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    load_dotenv(env_file)
    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def test_load_dotenv_missing_file(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "missing.env") is False
