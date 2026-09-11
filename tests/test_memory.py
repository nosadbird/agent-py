from pathlib import Path
from unittest.mock import patch

import pytest

from agent_assistant.memory import FileMemoryStore, MemoryRecord, _atomic_write

def test_short_term_is_trimmed_by_turn(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    for index in range(3):
        store.append_turn("s1", f"u{index}", f"a{index}")
    records = store.recent_turns("s1")
    assert [item.content for item in records] == ["u1", "a1", "u2", "a2"]


def test_corrupt_json_line_does_not_break_valid_memory(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "long_term.txt").write_text(
        '{"id":"1","timestamp":"now","kind":"fact","content":"喜欢中文",'
        '"session_id":null,"tags":[]}\n'
        "broken\n",
        encoding="utf-8",
    )
    store = FileMemoryStore(tmp_path, max_turns=2)
    assert [item.content for item in store.long_term_facts()] == ["喜欢中文"]


def test_append_turn_writes_user_and_assistant_with_same_session(
    tmp_path: Path,
) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5)
    store.append_turn("session-a", "hello", "world")

    records = store.recent_turns("session-a")
    assert len(records) == 2
    assert all(record.session_id == "session-a" for record in records)
    assert records[0].content == "hello"
    assert records[1].content == "world"
    assert records[0].kind == "user"
    assert records[1].kind == "assistant"


def test_trim_is_per_session(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=1)
    store.append_turn("s1", "u0", "a0")
    store.append_turn("s1", "u1", "a1")
    store.append_turn("s2", "other", "reply")

    assert [item.content for item in store.recent_turns("s1")] == ["u1", "a1"]
    assert [item.content for item in store.recent_turns("s2")] == ["other", "reply"]


def test_remember_rejects_blank_content(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    with pytest.raises(ValueError, match="空白"):
        store.remember("   ")
    with pytest.raises(ValueError, match="空白"):
        store.remember("\n\t")


def test_remember_rejects_oversized_content(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2, max_fact_chars=10)
    with pytest.raises(ValueError, match="4000|字符|上限|超过"):
        store.remember("x" * 11)


def test_remember_normalizes_tags(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    record = store.remember("  偏好 Python  ", tags=[" tag1 ", "", "tag2", 123])  # type: ignore[list-item]
    assert record.tags == ["tag1", "tag2"]
    assert record.content == "偏好 Python"
    assert record.kind == "fact"
    assert record.session_id is None


def test_forget_returns_true_when_record_exists(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    record = store.remember("要记住的内容")
    assert store.forget(record.id) is True
    assert store.long_term_facts() == []


def test_forget_returns_false_when_record_missing(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    assert store.forget("nonexistent-id") is False


def test_clear_session_does_not_affect_other_sessions(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5)
    store.append_turn("s1", "u1", "a1")
    store.append_turn("s2", "u2", "a2")

    store.clear_session("s1")

    assert store.recent_turns("s1") == []
    assert [item.content for item in store.recent_turns("s2")] == ["u2", "a2"]


def test_corrupt_short_term_line_is_skipped(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "short_term.txt").write_text(
        '{"id":"1","timestamp":"2026-01-01T00:00:00+00:00","kind":"user",'
        '"content":"valid","session_id":"s1","tags":[]}\n'
        "not-json\n",
        encoding="utf-8",
    )
    store = FileMemoryStore(tmp_path, max_turns=5)
    assert [item.content for item in store.recent_turns("s1")] == ["valid"]


def test_format_context_respects_max_chars(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5, max_context_chars=200)
    store.remember("长期记忆内容" * 5)
    store.append_turn("s1", "用户消息" * 10, "助手回复" * 10)

    context = store.format_context("s1")
    assert len(context) <= 200
    assert "长期记忆" in context
    assert "近期对话" in context


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    before = set(memory_dir.iterdir())

    store = FileMemoryStore(tmp_path, max_turns=2)
    store.append_turn("s1", "u1", "a1")
    store.remember("fact")

    after = set(memory_dir.iterdir())
    assert after - before == {
        memory_dir / "short_term.txt",
        memory_dir / "long_term.txt",
    }


def test_format_context_keeps_headers_when_recent_exceeds_limit(
    tmp_path: Path,
) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5, max_context_chars=80)
    store.append_turn("s1", "用户" * 20, "助手" * 20)

    context = store.format_context("s1")

    assert len(context) <= 80
    assert context.startswith("长期记忆")
    assert "近期对话" in context


def test_format_context_prefers_newest_long_term_facts(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5, max_context_chars=33)
    store.remember("旧事实")
    store.remember("新事实")
    store.append_turn("s1", "u", "a")

    context = store.format_context("s1")

    assert len(context) <= 33
    assert "新事实" in context
    assert "旧事实" not in context


def test_format_context_truncates_on_record_boundaries(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=5, max_context_chars=70)
    store.remember("事实A")
    store.remember("事实B")
    store.append_turn("s1", "用户一", "助手一")
    store.append_turn("s1", "用户二", "助手二")

    context = store.format_context("s1")
    lines = context.splitlines()

    assert len(context) <= 70
    assert all(not line.endswith(":") for line in lines if line not in {"长期记忆", "近期对话"})
    assert not any(line.startswith("用户:") and len(line) < len("用户: 用户二") for line in lines if line.startswith("用户:") and "用户二" in line and line != "用户: 用户二")


def test_atomic_write_cleans_temp_file_when_replace_fails(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    target = memory_dir / "short_term.txt"
    target.write_text("original\n", encoding="utf-8")
    before = set(memory_dir.iterdir())

    with patch.object(Path, "replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            _atomic_write(target, [])

    after = set(memory_dir.iterdir())
    assert after == before
    assert target.read_text(encoding="utf-8") == "original\n"


def test_atomic_write_preserves_replace_error_when_cleanup_unlink_fails(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    target = memory_dir / "long_term.txt"

    with patch.object(Path, "replace", side_effect=OSError("replace failed")):
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink", side_effect=OSError("unlink failed")):
                with pytest.raises(OSError, match="replace failed"):
                    _atomic_write(target, [])


@pytest.mark.parametrize("max_turns", [0, -1])
def test_constructor_rejects_invalid_max_turns(tmp_path: Path, max_turns: int) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        FileMemoryStore(tmp_path, max_turns=max_turns)


@pytest.mark.parametrize("max_fact_chars", [0, -5])
def test_constructor_rejects_invalid_max_fact_chars(
    tmp_path: Path, max_fact_chars: int
) -> None:
    with pytest.raises(ValueError, match="max_fact_chars"):
        FileMemoryStore(tmp_path, max_turns=2, max_fact_chars=max_fact_chars)


def test_constructor_rejects_too_small_max_context_chars(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_context_chars"):
        FileMemoryStore(tmp_path, max_turns=2, max_context_chars=10)


def test_memory_record_is_dataclass() -> None:
    record = MemoryRecord(
        id="id-1",
        timestamp="2026-01-01T00:00:00+00:00",
        kind="fact",
        content="内容",
        session_id=None,
        tags=["tag"],
    )
    assert record.id == "id-1"
    assert record.tags == ["tag"]
