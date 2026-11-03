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
    tools = ExecutorFileTools(tmp_path, max_chars=5, approve=lambda _p, _d: True)
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


def test_write_shows_diff_and_can_be_rejected(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old\n", encoding="utf-8")
    seen: list[str] = []
    tools = ExecutorFileTools(
        tmp_path,
        approve=lambda _path, diff: seen.append(diff) or False,
    )

    assert "未执行" in tools.write_text("a.txt", "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert "-old" in seen[0] and "+new" in seen[0]


def test_undo_last_write_refuses_to_overwrite_external_change(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    tools = ExecutorFileTools(tmp_path, approve=lambda _path, _diff: True)
    assert "写入成功" in tools.write_text("a.txt", "agent")
    target.write_text("user", encoding="utf-8")

    assert "又被修改" in tools.undo_last_write()
    assert target.read_text(encoding="utf-8") == "user"


def test_undo_last_write_restores_content(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    tools = ExecutorFileTools(tmp_path, approve=lambda _path, _diff: True)
    tools.write_text("a.txt", "new")

    assert "已撤销" in tools.undo_last_write()
    assert target.read_text(encoding="utf-8") == "old"


def test_write_refuses_change_made_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")

    def approve(_path: str, _diff: str) -> bool:
        target.write_text("external", encoding="utf-8")
        return True

    tools = ExecutorFileTools(tmp_path, approve=approve)
    assert "审批期间文件已变化" in tools.write_text("a.txt", "agent")
    assert target.read_text(encoding="utf-8") == "external"


def test_write_rejects_in_project_symlink_swap_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    other = tmp_path / "b.txt"
    target.write_text("old", encoding="utf-8")
    other.write_text("old", encoding="utf-8")

    def approve(_path: str, _diff: str) -> bool:
        target.unlink()
        try:
            target.symlink_to(other)
        except OSError:
            pytest.skip("当前系统不允许创建符号链接")
        return True

    tools = ExecutorFileTools(tmp_path, approve=approve)
    assert "路径或类型已变化" in tools.write_text("a.txt", "agent")
    assert other.read_text(encoding="utf-8") == "old"


def test_undo_rejects_symlink_replacement(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    target.write_text("old", encoding="utf-8")
    outside.write_text("agent", encoding="utf-8")
    tools = ExecutorFileTools(tmp_path, approve=lambda _path, _diff: True)
    tools.write_text("a.txt", "agent")
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("当前系统不允许创建符号链接")

    result = tools.undo_last_write()
    assert "撤销失败" in result
    assert outside.read_text(encoding="utf-8") == "agent"
