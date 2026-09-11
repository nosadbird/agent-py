"""TXT/JSONL 短期与长期记忆存储。"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_LONG_HEADER = "长期记忆"
_RECENT_HEADER = "近期对话"
_EMPTY_MARKER = "（无）"
_MIN_CONTEXT_CHARS = len(f"{_LONG_HEADER}\n{_EMPTY_MARKER}\n\n{_RECENT_HEADER}\n{_EMPTY_MARKER}")


@dataclass
class MemoryRecord:
    id: str
    timestamp: str
    kind: str
    content: str
    session_id: str | None
    tags: list[str]


_REQUIRED_FIELDS = ("id", "timestamp", "kind", "content", "session_id", "tags")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_dict(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "kind": record.kind,
        "content": record.content,
        "session_id": record.session_id,
        "tags": record.tags,
    }


def _parse_record(data: object) -> MemoryRecord | None:
    if not isinstance(data, dict):
        return None
    for field in _REQUIRED_FIELDS:
        if field not in data:
            return None
    record_id = data["id"]
    timestamp = data["timestamp"]
    kind = data["kind"]
    content = data["content"]
    session_id = data["session_id"]
    tags = data["tags"]
    if not isinstance(record_id, str):
        return None
    if not isinstance(timestamp, str):
        return None
    if not isinstance(kind, str):
        return None
    if not isinstance(content, str):
        return None
    if session_id is not None and not isinstance(session_id, str):
        return None
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        return None
    return MemoryRecord(
        id=record_id,
        timestamp=timestamp,
        kind=kind,
        content=content,
        session_id=session_id,
        tags=tags,
    )


def _read_records(path: Path) -> list[MemoryRecord]:
    if not path.exists():
        return []
    records: list[MemoryRecord] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        record = _parse_record(data)
        if record is not None:
            records.append(record)
    return records


def _atomic_write(path: Path, records: list[MemoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            for record in records:
                temp_file.write(json.dumps(_record_to_dict(record), ensure_ascii=False))
                temp_file.write("\n")
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
        raise


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    normalized: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _build_context_section(header: str, lines: list[str]) -> str:
    section_lines = [header]
    if lines:
        section_lines.extend(lines)
    else:
        section_lines.append(_EMPTY_MARKER)
    return "\n".join(section_lines)


def _format_context_text(long_lines: list[str], recent_lines: list[str]) -> str:
    long_section = _build_context_section(_LONG_HEADER, long_lines)
    recent_section = _build_context_section(_RECENT_HEADER, recent_lines)
    return f"{long_section}\n\n{recent_section}"


class FileMemoryStore:
    """基于 TXT/JSONL 文件的短期与长期记忆存储。"""

    def __init__(
        self,
        root: Path,
        max_turns: int,
        max_fact_chars: int = 4000,
        max_context_chars: int = 12000,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")
        if max_fact_chars <= 0:
            raise ValueError("max_fact_chars 必须大于 0")
        if max_context_chars < _MIN_CONTEXT_CHARS:
            raise ValueError(
                f"max_context_chars 过小，至少需要 {_MIN_CONTEXT_CHARS} 以容纳两个标题和基本换行"
            )
        self._root = root
        self._max_turns = max_turns
        self._max_fact_chars = max_fact_chars
        self._max_context_chars = max_context_chars
        self._memory_dir = root / "memory"
        self._short_term_path = self._memory_dir / "short_term.txt"
        self._long_term_path = self._memory_dir / "long_term.txt"

    def append_turn(self, session_id: str, user: str, assistant: str) -> None:
        records = _read_records(self._short_term_path)
        timestamp = _utc_now_iso()
        records.append(
            MemoryRecord(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                kind="user",
                content=user,
                session_id=session_id,
                tags=[],
            )
        )
        records.append(
            MemoryRecord(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                kind="assistant",
                content=assistant,
                session_id=session_id,
                tags=[],
            )
        )
        trimmed = self._trim_short_term(records)
        _atomic_write(self._short_term_path, trimmed)

    def recent_turns(self, session_id: str) -> list[MemoryRecord]:
        records = _read_records(self._short_term_path)
        return [record for record in records if record.session_id == session_id]

    def clear_session(self, session_id: str) -> None:
        records = _read_records(self._short_term_path)
        remaining = [record for record in records if record.session_id != session_id]
        _atomic_write(self._short_term_path, remaining)

    def remember(self, content: str, tags: list[str] | None = None) -> MemoryRecord:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("记忆内容不能为空白")
        if len(cleaned) > self._max_fact_chars:
            raise ValueError(f"记忆内容超过 {self._max_fact_chars} 字符上限")
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            timestamp=_utc_now_iso(),
            kind="fact",
            content=cleaned,
            session_id=None,
            tags=_normalize_tags(tags),
        )
        records = _read_records(self._long_term_path)
        records.append(record)
        _atomic_write(self._long_term_path, records)
        return record

    def forget(self, record_id: str) -> bool:
        records = _read_records(self._long_term_path)
        remaining = [record for record in records if record.id != record_id]
        if len(remaining) == len(records):
            return False
        _atomic_write(self._long_term_path, remaining)
        return True

    def long_term_facts(self) -> list[MemoryRecord]:
        return _read_records(self._long_term_path)

    def format_context(self, session_id: str) -> str:
        long_term = self.long_term_facts()
        recent = self.recent_turns(session_id)

        long_lines_newest = [f"- {record.content}" for record in reversed(long_term)]
        recent_lines_newest: list[str] = []
        for record in reversed(recent):
            role = "用户" if record.kind == "user" else "助手"
            recent_lines_newest.append(f"{role}: {record.content}")

        selected_recent: list[str] = []
        for line in recent_lines_newest:
            candidate = [line, *selected_recent]
            if len(_format_context_text([], candidate)) <= self._max_context_chars:
                selected_recent = candidate
            else:
                break

        selected_long: list[str] = []
        for line in long_lines_newest:
            candidate = [line, *selected_long]
            if (
                len(_format_context_text(candidate, selected_recent))
                <= self._max_context_chars
            ):
                selected_long = candidate
            else:
                break

        return _format_context_text(selected_long, selected_recent)

    def _trim_short_term(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        keep_count = self._max_turns * 2
        by_session: dict[str, list[MemoryRecord]] = {}
        for record in records:
            if record.session_id is None:
                continue
            by_session.setdefault(record.session_id, []).append(record)

        trimmed: list[MemoryRecord] = []
        for session_records in by_session.values():
            trimmed.extend(session_records[-keep_count:])

        session_order: list[str] = []
        for record in records:
            if record.session_id is not None and record.session_id not in session_order:
                session_order.append(record.session_id)

        ordered: list[MemoryRecord] = []
        kept_ids = {record.id for record in trimmed}
        for session_id in session_order:
            for record in records:
                if record.session_id == session_id and record.id in kept_ids:
                    ordered.append(record)
        return ordered
