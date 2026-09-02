"""Filesystem plugin tool handler tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from liteness.plugins.filesystem import (
    create_file_handler,
    delete_directory_handler,
    delete_file_handler,
    edit_file_handler,
    read_directory_handler,
    read_file_handler,
)
from liteness.testing import registry_with_plugins


def test_filesystem_plugin_registers_all_tools() -> None:
    names = set(registry_with_plugins("filesystem").names())
    assert names == {
        "read_file",
        "read_directory",
        "create_file",
        "edit_file",
        "delete_file",
        "delete_directory",
    }


def test_read_directory_lists_entries(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    result = read_directory_handler("c1", {"path": str(tmp_path)})

    assert not result.is_error
    assert "a.txt" in result.content
    assert "subdir/" in result.content


def test_read_directory_empty(tmp_path: Path) -> None:
    result = read_directory_handler("c1", {"path": str(tmp_path)})
    assert not result.is_error
    assert "empty" in result.content


def test_read_directory_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    result = read_directory_handler("c1", {"path": str(file_path)})
    assert result.is_error
    assert result.error_code == "NOT_A_DIRECTORY"


def test_create_file_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "new.txt"
    result = create_file_handler(
        "c1", {"path": str(target), "content": "hello world"}
    )
    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "hello world"


def test_create_file_rejects_existing(tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    result = create_file_handler("c1", {"path": str(target), "content": "new"})
    assert result.is_error
    assert result.error_code == "ALREADY_EXISTS"


def test_edit_file_replaces_first_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("foo bar foo", encoding="utf-8")
    result = edit_file_handler(
        "c1",
        {"path": str(target), "old_text": "foo", "new_text": "baz"},
    )
    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "baz bar foo"


def test_edit_file_missing_text(tmp_path: Path) -> None:
    target = tmp_path / "edit.txt"
    target.write_text("hello", encoding="utf-8")
    result = edit_file_handler(
        "c1",
        {"path": str(target), "old_text": "missing", "new_text": "x"},
    )
    assert result.is_error
    assert result.error_code == "TEXT_NOT_FOUND"


def test_delete_file(tmp_path: Path) -> None:
    target = tmp_path / "remove.txt"
    target.write_text("bye", encoding="utf-8")
    result = delete_file_handler("c1", {"path": str(target)})
    assert not result.is_error
    assert not target.exists()


def test_delete_directory_empty(tmp_path: Path) -> None:
    target = tmp_path / "empty_dir"
    target.mkdir()
    result = delete_directory_handler("c1", {"path": str(target)})
    assert not result.is_error
    assert not target.exists()


def test_delete_directory_non_empty_requires_recursive(tmp_path: Path) -> None:
    target = tmp_path / "nested"
    target.mkdir()
    (target / "child.txt").write_text("x", encoding="utf-8")
    result = delete_directory_handler("c1", {"path": str(target)})
    assert result.is_error
    assert result.error_code == "DELETE_ERROR"
    assert target.exists()


def test_delete_directory_recursive(tmp_path: Path) -> None:
    target = tmp_path / "nested"
    target.mkdir()
    (target / "child.txt").write_text("x", encoding="utf-8")
    result = delete_directory_handler(
        "c1", {"path": str(target), "recursive": True}
    )
    assert not result.is_error
    assert not target.exists()


def test_read_file_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "readme.md"
    target.write_text("# hi", encoding="utf-8")
    result = read_file_handler("c1", {"path": str(target)})
    assert result.content == "# hi"
