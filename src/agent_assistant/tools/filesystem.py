"""限定在项目目录内的文件工具。"""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

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


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
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


class ProjectFileTools:
    """提供受项目根目录约束的 UTF-8 文件操作。"""

    def __init__(self, root: Path, max_chars: int = 12_000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数")
        self._root = root.resolve()
        self._max_chars = max_chars

    def read_text(self, path: str) -> str:
        try:
            target = resolve_project_path(self._root, path)
            if target.is_symlink() or not target.is_file():
                return f"读取错误：不是项目内普通文件：{path}"
            return _truncate(target.read_text(encoding="utf-8"), self._max_chars)
        except UnicodeDecodeError:
            return f"读取错误：文件不是有效 UTF-8 编码：{path}"
        except (OSError, ValueError) as exc:
            return f"读取错误：{exc}"

    def write_text(self, path: str, content: str) -> str:
        try:
            target = resolve_project_path(self._root, path)
            if target.exists() and (target.is_symlink() or not target.is_file()):
                return f"写入错误：目标不是项目内普通文件：{path}"
            target.parent.mkdir(parents=True, exist_ok=True)
            # 创建目录后再次解析，防止父路径经符号链接或 junction 逃逸。
            target = resolve_project_path(self._root, path)
            target.write_text(content, encoding="utf-8")
            return f"写入成功：{target.relative_to(self._root).as_posix()}"
        except (OSError, ValueError, UnicodeError) as exc:
            return f"写入错误：{exc}"

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
            return _truncate(result or "（空目录）", self._max_chars)
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
            return _truncate(result, self._max_chars)
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
        ]
