from pathlib import Path
import os
import sys

import pytest

from agent_assistant.tools.filesystem import ProjectFileTools, resolve_project_path


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目目录"):
        resolve_project_path(tmp_path, "../outside.txt")


@pytest.mark.parametrize("user_path", ["/tmp/outside", "C:\\outside.txt", "\x00"])
def test_invalid_paths_are_rejected(tmp_path: Path, user_path: str) -> None:
    with pytest.raises(ValueError):
        resolve_project_path(tmp_path, user_path)


def test_normal_path_is_resolved_under_project(tmp_path: Path) -> None:
    assert resolve_project_path(tmp_path, "docs/readme.txt") == (
        tmp_path / "docs" / "readme.txt"
    ).resolve()


def test_file_write_read_and_truncation(tmp_path: Path) -> None:
    tools = ProjectFileTools(tmp_path, max_chars=5)
    assert "写入成功" in tools.write_text("nested/a.txt", "abcdef")
    assert (tmp_path / "nested" / "a.txt").read_text(encoding="utf-8") == "abcdef"
    result = tools.read_text("nested/a.txt")
    assert result.startswith("abcde")
    assert "已截断" in result


def test_list_directory_is_sorted_relative_and_depth_one(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "a" / "deep").mkdir()
    (tmp_path / "a" / "deep" / "hidden.txt").write_text("x", encoding="utf-8")

    lines = ProjectFileTools(tmp_path).list_directory().splitlines()

    assert lines == ["a/", "a/c.txt", "a/deep/", "b.txt"]


def test_read_invalid_utf8_returns_chinese_error(tmp_path: Path) -> None:
    (tmp_path / "bad.bin").write_bytes(b"\xff\xfe")
    result = ProjectFileTools(tmp_path).read_text("bad.bin")
    assert "UTF-8" in result or "编码" in result


def test_search_text_finds_regex_and_skips_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("alpha\nbeta 42\n", encoding="utf-8")
    (tmp_path / "bad.bin").write_bytes(b"\xff\xfe")

    result = ProjectFileTools(tmp_path).search_text(r"beta \d+")

    assert "ok.txt:2:beta 42" in result
    assert "bad.bin" in result
    assert "跳过" in result


def test_search_invalid_regex_returns_chinese_error(tmp_path: Path) -> None:
    result = ProjectFileTools(tmp_path).search_text("[")
    assert "正则" in result and ("非法" in result or "错误" in result)


def test_directory_and_search_results_are_truncated(tmp_path: Path) -> None:
    (tmp_path / "long-name.txt").write_text("matching content", encoding="utf-8")
    tools = ProjectFileTools(tmp_path, max_chars=5)
    assert "已截断" in tools.list_directory()
    assert "已截断" in tools.search_text("matching")


def test_search_limits_scanned_files(tmp_path: Path) -> None:
    for index in range(1002):
        (tmp_path / f"{index:04}.txt").write_text("needle", encoding="utf-8")
    result = ProjectFileTools(tmp_path, max_chars=100_000).search_text("needle")
    assert "1000" in result and ("限制" in result or "停止" in result)


def test_tool_errors_are_returned_as_text(tmp_path: Path) -> None:
    tools = ProjectFileTools(tmp_path)
    assert "错误" in tools.read_text("../outside.txt")
    assert "错误" in tools.write_text("../outside.txt", "x")


def test_as_tools_returns_expected_langchain_tools(tmp_path: Path) -> None:
    tools = ProjectFileTools(tmp_path).as_tools()
    assert {tool.name for tool in tools} == {
        "read_text",
        "write_text",
        "list_directory",
        "search_text",
    }
    assert all(tool.args_schema is not None for tool in tools)


def _can_create_symlink() -> bool:
    if not hasattr(os, "symlink"):
        return False
    if sys.platform == "win32":
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    return True


@pytest.mark.skipif(not _can_create_symlink(), reason="当前平台或权限不支持符号链接")
def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "link", target_is_directory=True)
    with pytest.raises(ValueError, match="项目目录"):
        resolve_project_path(root, "link/secret.txt")
