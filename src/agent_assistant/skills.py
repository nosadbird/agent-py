"""Skills 动态发现与按需加载。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_FRONTMATTER_PATTERN = re.compile(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)(.*)", re.DOTALL)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


class _ListSkillsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _LoadSkillInput(BaseModel):
    name: str = Field(..., min_length=1, description="要加载的技能名称")


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_frontmatter(text: str) -> tuple[dict | None, str | None]:
    match = _FRONTMATTER_PATTERN.match(text)
    if match is None:
        return None, "缺少 YAML frontmatter"
    raw_yaml = match.group(1)
    body = match.group(2)
    try:
        metadata = yaml.safe_load(raw_yaml)
    except yaml.YAMLError:
        return None, "YAML 解析失败"
    if not isinstance(metadata, dict):
        return None, "frontmatter 必须是 YAML 映射"
    return metadata, body


def _is_valid_name(name: str) -> bool:
    return _NAME_PATTERN.fullmatch(name) is not None


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class SkillRegistry:
    """扫描、校验并按需加载 skills/*/SKILL.md。"""

    def __init__(self, root: Path, max_bytes: int = 100_000) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._skills_root = root / "skills"
        self._warnings: list[str] = []
        self._index: dict[str, SkillInfo] = {}
        self._fingerprints: dict[str, str] = {}
        self._scan()

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def list_skills(self) -> list[SkillInfo]:
        return sorted(self._index.values(), key=lambda item: item.name)

    def load_skill(self, name: str) -> str:
        if not _is_valid_name(name):
            return f"非法技能名称：{name}"
        skill = self._index.get(name)
        if skill is None:
            return f"技能不存在：{name}"
        return self._read_skill_file(skill)

    def catalog_text(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "（无可用技能）"
        lines = [f"- {item.name}: {item.description}" for item in skills]
        return "\n".join(lines)

    def as_tools(self) -> list[BaseTool]:
        registry = self

        def list_skills_tool() -> str:
            return registry.catalog_text()

        def load_skill_tool(name: str) -> str:
            return registry.load_skill(name)

        async def alist_skills_tool() -> str:
            return registry.catalog_text()

        async def aload_skill_tool(name: str) -> str:
            return registry.load_skill(name)

        return [
            StructuredTool.from_function(
                func=list_skills_tool,
                coroutine=alist_skills_tool,
                name="list_skills",
                description="列出所有可用技能的名称与描述",
                args_schema=_ListSkillsInput,
            ),
            StructuredTool.from_function(
                func=load_skill_tool,
                coroutine=aload_skill_tool,
                name="load_skill",
                description="按名称加载指定技能的完整 SKILL.md 正文",
                args_schema=_LoadSkillInput,
            ),
        ]

    def _scan(self) -> None:
        self._warnings.clear()
        self._index.clear()
        self._fingerprints.clear()
        if not self._skills_root.is_dir():
            return

        skills_root_resolved = self._skills_root.resolve()
        seen_names: set[str] = set()

        for entry in sorted(self._skills_root.iterdir()):
            dir_name = entry.name
            if not entry.is_dir():
                continue
            if entry.is_symlink():
                self._warnings.append(f"跳过符号链接目录：{dir_name}")
                continue
            if not _is_valid_name(dir_name):
                self._warnings.append(f"跳过非法目录名：{dir_name}")
                continue

            skill_path = entry / "SKILL.md"
            if not skill_path.is_file():
                self._warnings.append(f"技能 {dir_name} 缺少 SKILL.md")
                continue
            if skill_path.is_symlink():
                self._warnings.append(f"跳过符号链接文件：{dir_name}/SKILL.md")
                continue
            if not _is_within_root(skill_path, skills_root_resolved):
                self._warnings.append(f"技能 {dir_name} 路径越界")
                continue

            try:
                size = skill_path.stat().st_size
            except OSError:
                self._warnings.append(f"技能 {dir_name} 无法读取文件大小")
                continue
            if size > self._max_bytes:
                self._warnings.append(
                    f"技能 {dir_name} 文件超过大小限制（{size} > {self._max_bytes}）"
                )
                continue

            try:
                raw = skill_path.read_bytes()
            except OSError:
                self._warnings.append(f"技能 {dir_name} 读取失败")
                continue

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._warnings.append(f"技能 {dir_name} 不是有效的 UTF-8 编码")
                continue

            metadata, body_or_error = _parse_frontmatter(text)
            if metadata is None:
                self._warnings.append(f"技能 {dir_name} {body_or_error}")
                continue

            name = metadata.get("name")
            description = metadata.get("description")
            if name is None:
                self._warnings.append(f"技能 {dir_name} 缺少有效的 name 字段")
                continue
            if not isinstance(name, str) or not name.strip():
                self._warnings.append(f"技能 {dir_name} 缺少有效的 name 字段")
                continue
            if not isinstance(description, str) or not description.strip():
                self._warnings.append(f"技能 {dir_name} 缺少有效的 description 字段")
                continue

            if not _is_valid_name(name):
                self._warnings.append(f"技能 {dir_name} 的 name 字段非法：{name.strip()}")
                continue

            name = name.strip()
            description = description.strip()
            if name in seen_names:
                self._warnings.append(f"跳过重名技能：{name}")
                continue
            if name != dir_name:
                self._warnings.append(
                    f"技能 {dir_name} 的目录名与 frontmatter name 不一致"
                )
                continue

            seen_names.add(name)
            self._index[name] = SkillInfo(
                name=name,
                description=description,
                path=skill_path.resolve(),
            )
            self._fingerprints[name] = _bytes_sha256(raw)

    def _read_skill_file(self, skill: SkillInfo) -> str:
        skills_root_resolved = self._skills_root.resolve()
        if not _is_within_root(skill.path, skills_root_resolved):
            return f"非法技能路径：{skill.name}"
        if not skill.path.is_file():
            return f"技能文件不存在或已被删除：{skill.name}"
        try:
            size = skill.path.stat().st_size
        except OSError:
            return f"技能 {skill.name} 读取失败"
        if size > self._max_bytes:
            return f"技能 {skill.name} 文件超过大小限制"
        try:
            raw = skill.path.read_bytes()
        except OSError:
            return f"技能 {skill.name} 读取失败"

        expected = self._fingerprints.get(skill.name)
        if expected is None or _bytes_sha256(raw) != expected:
            return f"技能文件内容已变更（指纹不一致）：{skill.name}"

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"技能 {skill.name} 不是有效的 UTF-8 编码"

        metadata, body_or_error = _parse_frontmatter(text)
        if metadata is None:
            return f"技能 {skill.name} {body_or_error}"
        return text
