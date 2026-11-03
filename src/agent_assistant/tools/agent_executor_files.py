"""agent_executor 专用的项目内文件读写与搜索。"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agent_assistant.agent_events import EventHandler, emit_event

_TRUNCATED = "\n……（已截断）"
_MAX_SEARCH_FILES = 1000


class _ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="项目内相对文件路径")


class _WriteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="项目内相对文件路径")
    content: str = Field(..., description="要写入的 UTF-8 文本")


class _ListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(".", description="项目内相对目录路径")


class _SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(..., description="Python 正则表达式")
    path: str = Field(".", description="项目内相对文件或目录路径")


class _UndoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class _WriteSnapshot:
    relative_path: str
    target: Path
    before: str | None
    after: str


def confirm_file_change(path: str, diff: str) -> bool:
    """展示文件差异，仅 y/yes 视为同意。"""
    print(f"Agent 请求修改文件：{path}\n{diff}")
    try:
        answer = input("是否应用？仅输入 y/yes 表示同意：")
    except (KeyboardInterrupt, EOFError):
        print("文件修改已取消。")
        return False
    return answer.strip().lower() in {"y", "yes"}


def truncate_text(text: str, max_chars: int) -> str:
    """把工具返回截到指定长度，超出部分用固定后缀标明。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATED


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_project_path(root: Path, user_path: str) -> Path:
    """解析项目内相对路径，并拒绝任何形式的目录逃逸。"""
    root_resolved = root.resolve()
    if "\x00" in user_path:
        raise ValueError("路径包含空字节，无法访问项目目录")
    if not user_path:
        raise ValueError("路径不能为空")
    candidate_path = Path(user_path)
    windows_path = PureWindowsPath(user_path)
    if candidate_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("仅允许项目目录内的相对路径")

    candidate = (root_resolved / candidate_path).resolve()
    if not _is_within(candidate, root_resolved):
        raise ValueError("路径超出项目目录")
    return candidate


class ExecutorFileTools:
    """提供受项目根目录约束的 UTF-8 文件操作。"""

    def __init__(
        self,
        root: Path,
        max_chars: int = 12_000,
        approve: Callable[[str, str], bool] | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数")
        self._root = root.resolve()
        self._max_chars = max_chars
        self._approve = approve
        self._on_event = on_event
        self._last_write: _WriteSnapshot | None = None

    def read_text(self, path: str) -> str:
        try:
            target = resolve_project_path(self._root, path)
            if target.is_symlink() or not target.is_file():
                return f"读取错误：不是项目内普通文件：{path}"
            return truncate_text(target.read_text(encoding="utf-8"), self._max_chars)
        except UnicodeDecodeError:
            return f"读取错误：文件不是有效 UTF-8 编码：{path}"
        except (OSError, ValueError) as exc:
            return f"读取错误：{exc}"

    def write_text(self, path: str, content: str) -> str:
        try:
            target = resolve_project_path(self._root, path)
            original_target = target
            if target.exists() and (target.is_symlink() or not target.is_file()):
                return f"写入错误：目标不是项目内普通文件：{path}"
            before = target.read_text(encoding="utf-8") if target.exists() else None
            if before == content:
                return f"文件内容未变化：{path}"
            diff = self._diff(path, before, content)
            emit_event(self._on_event, "file_change", diff)
            if self._approve is not None and not self._approve(path, diff):
                return f"文件修改未执行：{path}"
            target.parent.mkdir(parents=True, exist_ok=True)
            # 审批期间若文件被其他操作改过，则拒绝覆盖。
            target = resolve_project_path(self._root, path)
            if target != original_target or target.is_symlink():
                return f"写入取消：审批期间文件路径或类型已变化：{path}"
            current = target.read_text(encoding="utf-8") if target.exists() else None
            if current != before:
                return f"写入取消：审批期间文件已变化：{path}"
            target.write_text(content, encoding="utf-8")
            relative = target.relative_to(self._root).as_posix()
            self._last_write = _WriteSnapshot(relative, target, before, content)
            return f"写入成功：{relative}"
        except (OSError, ValueError, UnicodeError) as exc:
            return f"写入错误：{exc}"

    def undo_last_write(self) -> str:
        """撤销最近一次写入；若文件后来被改动则拒绝覆盖。"""
        snapshot = self._last_write
        if snapshot is None:
            return "没有可撤销的文件写入"
        try:
            target = resolve_project_path(self._root, snapshot.relative_path)
            if target != snapshot.target or target.is_symlink() or not target.is_file():
                return "撤销失败：文件路径或类型已变化"
            current = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "撤销失败：文件已被其他操作删除"
        except (OSError, ValueError, UnicodeError) as exc:
            return f"撤销失败：{exc}"
        if current != snapshot.after:
            return "撤销失败：文件在 Agent 写入后又被修改"
        relative = snapshot.relative_path
        action = "删除新建文件" if snapshot.before is None else "恢复写入前内容"
        if self._approve is not None and not self._approve(relative, action):
            return "撤销未执行"
        # 审批后再校验一次，防止等待确认期间发生路径替换或内容修改。
        target = resolve_project_path(self._root, relative)
        if target != snapshot.target or target.is_symlink() or not target.is_file():
            return "撤销失败：审批期间文件路径或类型已变化"
        if target.read_text(encoding="utf-8") != snapshot.after:
            return "撤销失败：审批期间文件内容已变化"
        if snapshot.before is None:
            target.unlink()
        else:
            target.write_text(snapshot.before, encoding="utf-8")
        self._last_write = None
        return f"已撤销最近一次写入：{relative}"

    def _diff(self, path: str, before: str | None, after: str) -> str:
        old_lines = [] if before is None else before.splitlines(keepends=True)
        new_lines = after.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return truncate_text(diff or "（内容有变化）", self._max_chars)

    def list_directory(self, path: str = ".") -> str:
        try:
            directory = resolve_project_path(self._root, path)
            if directory.is_symlink() or not directory.is_dir():
                return f"目录错误：不是项目内目录：{path}"
            entries: list[str] = []
            for child in directory.iterdir():
                self._append_directory_entry(entries, child)
                if child.is_dir() and not child.is_symlink():
                    for grandchild in child.iterdir():
                        self._append_directory_entry(entries, grandchild)
            result = "\n".join(sorted(entries))
            return truncate_text(result or "（空目录）", self._max_chars)
        except (OSError, ValueError) as exc:
            return f"目录错误：{exc}"

    def _append_directory_entry(self, entries: list[str], entry: Path) -> None:
        try:
            resolved = entry.resolve()
            if not _is_within(resolved, self._root) or entry.is_symlink():
                return
            relative = entry.relative_to(self._root).as_posix()
            entries.append(relative + ("/" if entry.is_dir() else ""))
        except (OSError, ValueError):
            return

    def search_text(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"正则表达式错误：{exc}"
        try:
            target = resolve_project_path(self._root, path)
            files = self._search_candidates(target)
            lines: list[str] = []
            scanned = 0
            limited = False
            for file_path in files:
                if scanned >= _MAX_SEARCH_FILES:
                    limited = True
                    break
                scanned += 1
                relative = file_path.relative_to(self._root).as_posix()
                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    lines.append(f"跳过非 UTF-8 文件：{relative}")
                    continue
                except OSError as exc:
                    lines.append(f"跳过无法读取文件：{relative}（{exc}）")
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        lines.append(f"{relative}:{line_number}:{line}")
            if limited:
                lines.append(f"已达到扫描文件限制 {_MAX_SEARCH_FILES}，停止扫描")
            result = "\n".join(lines) or "未找到匹配内容"
            return truncate_text(result, self._max_chars)
        except (OSError, ValueError) as exc:
            return f"搜索错误：{exc}"

    def _search_candidates(self, target: Path) -> list[Path]:
        if target.is_symlink():
            return []
        if target.is_file():
            return [target]
        if not target.is_dir():
            raise ValueError("搜索路径不是项目内文件或目录")

        candidates: list[Path] = []
        for current, dir_names, file_names in os.walk(target, followlinks=False):
            current_path = Path(current)
            safe_dirs: list[str] = []
            for name in dir_names:
                directory = current_path / name
                try:
                    if not directory.is_symlink() and _is_within(
                        directory.resolve(), self._root
                    ):
                        safe_dirs.append(name)
                except OSError:
                    continue
            dir_names[:] = safe_dirs
            for name in file_names:
                candidate = current_path / name
                try:
                    if (
                        not candidate.is_symlink()
                        and candidate.is_file()
                        and _is_within(candidate.resolve(), self._root)
                    ):
                        candidates.append(candidate)
                except OSError:
                    continue
        return sorted(candidates, key=lambda item: item.relative_to(self._root).as_posix())

    def as_tools(self) -> list[BaseTool]:
        """组装为可同步或异步调用的 LangChain 工具。"""

        async def aread(path: str) -> str:
            return self.read_text(path)

        async def awrite(path: str, content: str) -> str:
            return self.write_text(path, content)

        async def alist(path: str = ".") -> str:
            return self.list_directory(path)

        async def asearch(pattern: str, path: str = ".") -> str:
            return self.search_text(pattern, path)

        async def aundo() -> str:
            return self.undo_last_write()

        return [
            StructuredTool.from_function(
                func=self.read_text,
                coroutine=aread,
                name="read_text",
                description="读取项目目录内的 UTF-8 文本文件",
                args_schema=_ReadInput,
            ),
            StructuredTool.from_function(
                func=self.write_text,
                coroutine=awrite,
                name="write_text",
                description="向项目目录内写入 UTF-8 文本文件",
                args_schema=_WriteInput,
            ),
            StructuredTool.from_function(
                func=self.list_directory,
                coroutine=alist,
                name="list_directory",
                description="列出项目目录及其下一层内容",
                args_schema=_ListInput,
            ),
            StructuredTool.from_function(
                func=self.search_text,
                coroutine=asearch,
                name="search_text",
                description="用 Python 正则搜索项目内文本文件",
                args_schema=_SearchInput,
            ),
            StructuredTool.from_function(
                func=self.undo_last_write,
                coroutine=aundo,
                name="undo_last_write",
                description="撤销 Agent 最近一次成功的文件写入",
                args_schema=_UndoInput,
            ),
        ]
