from pathlib import Path

import pytest

from agent_assistant.tools.agent_executor_files import (
    ExecutorFileTools,
    resolve_project_path,
    truncate_text,
)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目目录"):
        resolve_project_path(tmp_path, "../outside.txt")


@pytest.mark.parametrize("user_path", ["/tmp/outside", "C:\\outside.txt", "\x00"])
def test_invalid_paths_are_rejected(tmp_path: Path, user_path: str) -> None:
    with pytest.raises(ValueError):
        resolve_project_path(tmp_path, user_path)


def test_file_write_read_and_truncation(tmp_path: Path) -> None:
    tools = ExecutorFileTools(tmp_path, max_chars=5)
    assert "写入成功" in tools.write_text("nested/a.txt", "abcdef")
    assert (tmp_path / "nested" / "a.txt").read_text(encoding="utf-8") == "abcdef"
    result = tools.read_text("nested/a.txt")
    assert result.startswith("abcde")
    assert "已截断" in result


def test_search_text_finds_regex_and_truncates(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("alpha\nbeta 42\n", encoding="utf-8")
    found = ExecutorFileTools(tmp_path).search_text(r"beta \d+")
    assert "ok.txt:2:beta 42" in found
    truncated = ExecutorFileTools(tmp_path, max_chars=5).search_text("beta")
    assert "已截断" in truncated


def test_truncate_text_keeps_short_values() -> None:
    assert truncate_text("abc", 5) == "abc"
    assert truncate_text("abcdef", 3).startswith("abc")
    assert "已截断" in truncate_text("abcdef", 3)
