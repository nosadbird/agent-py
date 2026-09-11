from dataclasses import fields
from pathlib import Path
import os
import sys

import pytest

from agent_assistant.skills import SkillInfo, SkillRegistry


def _write_skill(
    root: Path,
    dir_name: str,
    frontmatter: str,
    body: str = "# 正文\n",
) -> Path:
    skill_dir = root / "skills" / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    return skill_path


def test_registry_lists_and_loads_valid_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: writer\ndescription: 写作流程\n---\n\n# 步骤\n先澄清。\n",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path)
    assert [item.name for item in registry.list_skills()] == ["writer"]
    assert "先澄清" in registry.load_skill("writer")


def test_registry_rejects_path_escape(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    assert "非法" in registry.load_skill("../secret")


def test_registry_skips_missing_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "nofm"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text("# 无 frontmatter\n", encoding="utf-8")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("frontmatter" in warning.lower() or "元数据" in warning for warning in registry.warnings)


def test_registry_skips_corrupt_yaml(tmp_path: Path) -> None:
    _write_skill(tmp_path, "bad-yaml", "name: bad-yaml\ndescription: [unclosed")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("YAML" in warning or "yaml" in warning.lower() for warning in registry.warnings)


def test_registry_skips_missing_required_fields(tmp_path: Path) -> None:
    _write_skill(tmp_path, "no-desc", "name: no-desc")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("description" in warning or "描述" in warning for warning in registry.warnings)


def test_registry_skips_invalid_name(tmp_path: Path) -> None:
    invalid_dir = tmp_path / "skills" / "bad name"
    invalid_dir.mkdir(parents=True)
    invalid_dir.joinpath("SKILL.md").write_text(
        "---\nname: bad name\ndescription: 非法名称\n---\n\n正文\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert registry.warnings


def test_registry_skips_directory_name_mismatch(tmp_path: Path) -> None:
    _write_skill(tmp_path, "dir-a", "name: dir-b\ndescription: 名称不一致")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("一致" in warning or "匹配" in warning for warning in registry.warnings)


def test_registry_skips_duplicate_names(tmp_path: Path) -> None:
    _write_skill(tmp_path, "dup", "name: dup\ndescription: 第一个", body="正文一\n")
    _write_skill(tmp_path, "dup2", "name: dup\ndescription: 第二个", body="正文二\n")

    registry = SkillRegistry(tmp_path)

    assert len(registry.list_skills()) == 1
    assert any("重名" in warning or "重复" in warning for warning in registry.warnings)


def test_registry_skips_oversized_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "big", "name: big\ndescription: 过大", body="x" * 200)

    registry = SkillRegistry(tmp_path, max_bytes=50)

    assert registry.list_skills() == []
    assert any("大小" in warning or "超过" in warning for warning in registry.warnings)


def test_registry_skips_invalid_utf8(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "bad-utf8"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_bytes(
        b"---\nname: bad-utf8\ndescription: \xff\xfe\n---\n\nbody\n"
    )

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("UTF-8" in warning or "编码" in warning for warning in registry.warnings)


def test_registry_sorts_valid_skills_by_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "gamma", "name: gamma\ndescription: G")
    _write_skill(tmp_path, "alpha", "name: alpha\ndescription: A")
    _write_skill(tmp_path, "beta", "name: beta\ndescription: B")

    registry = SkillRegistry(tmp_path)

    assert [item.name for item in registry.list_skills()] == ["alpha", "beta", "gamma"]


def test_warnings_returns_copy_not_internal_list(tmp_path: Path) -> None:
    _write_skill(tmp_path, "no-desc", "name: no-desc")

    registry = SkillRegistry(tmp_path)
    warnings = registry.warnings
    assert warnings
    warnings.append("外部注入")

    assert "外部注入" not in registry.warnings


def test_load_skill_missing_returns_error_text(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)

    result = registry.load_skill("missing")

    assert "不存在" in result or "未找到" in result
    assert isinstance(result, str)


def test_catalog_text_contains_name_and_description_only(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )

    registry = SkillRegistry(tmp_path)
    catalog = registry.catalog_text()

    assert "writer" in catalog
    assert "写作流程" in catalog
    assert "先澄清" not in catalog


def test_as_tools_returns_list_and_load_tools(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )

    registry = SkillRegistry(tmp_path)
    tools = registry.as_tools()

    assert len(tools) == 2
    tool_names = {tool.name for tool in tools}
    assert tool_names == {"list_skills", "load_skill"}


def test_as_tools_schema_and_sync_async_invoke(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )

    registry = SkillRegistry(tmp_path)
    list_tool = next(tool for tool in registry.as_tools() if tool.name == "list_skills")
    load_tool = next(tool for tool in registry.as_tools() if tool.name == "load_skill")

    assert load_tool.args_schema is not None
    assert "name" in load_tool.args_schema.model_fields

    catalog = list_tool.invoke({})
    assert "writer" in catalog
    assert "写作流程" in catalog

    loaded = load_tool.invoke({"name": "writer"})
    assert "先澄清" in loaded


@pytest.mark.asyncio
async def test_as_tools_supports_async_invoke(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )

    registry = SkillRegistry(tmp_path)
    list_tool = next(tool for tool in registry.as_tools() if tool.name == "list_skills")
    load_tool = next(tool for tool in registry.as_tools() if tool.name == "load_skill")

    catalog = await list_tool.ainvoke({})
    assert "writer" in catalog

    loaded = await load_tool.ainvoke({"name": "writer"})
    assert "先澄清" in loaded


def test_load_skill_returns_error_when_file_removed_after_index(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )
    registry = SkillRegistry(tmp_path)
    skill_path.unlink()

    result = registry.load_skill("writer")

    assert "不存在" in result or "读取" in result or "失败" in result


def test_skill_info_is_frozen_dataclass(tmp_path: Path) -> None:
    info = SkillInfo(name="writer", description="写作", path=Path("skills/writer/SKILL.md"))
    with pytest.raises(Exception):
        info.name = "other"  # type: ignore[misc]


def test_skill_info_has_only_three_public_fields() -> None:
    assert {field.name for field in fields(SkillInfo)} == {"name", "description", "path"}


def test_skill_info_three_parameter_constructor() -> None:
    path = Path("skills/writer/SKILL.md")
    info = SkillInfo(name="writer", description="写作流程", path=path)

    assert info.name == "writer"
    assert info.description == "写作流程"
    assert info.path == path


def test_registry_skips_name_with_trailing_newline_in_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "newline-name"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        '---\nname: "newline-name\\n"\ndescription: 含换行\n---\n\n正文\n',
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("非法" in warning for warning in registry.warnings)


def test_load_skill_rejects_name_with_trailing_newline(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)

    assert "非法" in registry.load_skill("writer\n")


def test_registry_skips_missing_name_field(tmp_path: Path) -> None:
    _write_skill(tmp_path, "no-name", "description: 缺少 name")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("name" in warning.lower() for warning in registry.warnings)


def test_registry_skips_frontmatter_invalid_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "bad-name", "name: bad@name\ndescription: 非法字符")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("非法" in warning for warning in registry.warnings)


def test_registry_keeps_valid_skills_when_invalid_coexist(tmp_path: Path) -> None:
    _write_skill(tmp_path, "good", "name: good\ndescription: 有效技能")
    _write_skill(tmp_path, "bad-yaml", "name: bad-yaml\ndescription: [unclosed")
    skill_dir = tmp_path / "skills" / "bad name"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: bad name\ndescription: 非法\n---\n\n正文\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)

    assert [item.name for item in registry.list_skills()] == ["good"]
    assert len(registry.warnings) >= 2


def test_registry_scans_only_direct_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "skills" / "nested" / "deep"
    nested.mkdir(parents=True)
    nested.joinpath("SKILL.md").write_text(
        "---\nname: deep\ndescription: 嵌套过深\n---\n\n正文\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("nested" in warning for warning in registry.warnings)


def _can_create_symlink() -> bool:
    if not hasattr(os, "symlink"):
        return False
    if sys.platform == "win32":
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    return True


@pytest.mark.skipif(not _can_create_symlink(), reason="当前平台或权限不支持符号链接测试")
def test_registry_skips_symlink_directory_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("SKILL.md").write_text(
        "---\nname: outside\ndescription: 外部\n---\n\n外部正文\n",
        encoding="utf-8",
    )
    link_dir = tmp_path / "skills" / "linked"
    link_dir.parent.mkdir(parents=True)
    os.symlink(outside, link_dir, target_is_directory=True)

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("符号链接" in warning for warning in registry.warnings)


@pytest.mark.skipif(not _can_create_symlink(), reason="当前平台或权限不支持符号链接测试")
def test_registry_skips_symlink_file_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: escape\ndescription: 外部\n---\n\n外部正文\n",
        encoding="utf-8",
    )
    skill_dir = tmp_path / "skills" / "escape"
    skill_dir.mkdir(parents=True)
    os.symlink(outside, skill_dir / "SKILL.md")

    registry = SkillRegistry(tmp_path)

    assert registry.list_skills() == []
    assert any("符号链接" in warning for warning in registry.warnings)


def test_load_skill_returns_error_when_file_replaced_after_index(tmp_path: Path) -> None:
    skill_path = _write_skill(
        tmp_path,
        "writer",
        "name: writer\ndescription: 写作流程",
        body="# 步骤\n先澄清。\n",
    )
    registry = SkillRegistry(tmp_path)
    skill_path.write_text(
        "---\nname: writer\ndescription: 写作流程\n---\n\n# 步骤\n已替换。\n",
        encoding="utf-8",
    )

    result = registry.load_skill("writer")

    assert result.startswith("技能") or "变更" in result or "不一致" in result or "指纹" in result
    assert "已替换" not in result


def test_as_tools_list_skills_has_empty_schema(tmp_path: Path) -> None:
    _write_skill(tmp_path, "writer", "name: writer\ndescription: 写作流程")

    registry = SkillRegistry(tmp_path)
    list_tool = next(tool for tool in registry.as_tools() if tool.name == "list_skills")

    assert list_tool.args_schema is not None
    assert list(list_tool.args_schema.model_fields.keys()) == []


def test_as_tools_load_skill_name_is_required(tmp_path: Path) -> None:
    _write_skill(tmp_path, "writer", "name: writer\ndescription: 写作流程")

    registry = SkillRegistry(tmp_path)
    load_tool = next(tool for tool in registry.as_tools() if tool.name == "load_skill")

    assert load_tool.args_schema is not None
    name_field = load_tool.args_schema.model_fields["name"]
    assert name_field.is_required()

    with pytest.raises(Exception):
        load_tool.invoke({})


def test_load_skill_detects_crlf_to_lf_byte_replacement(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    original = (
        b"---\r\nname: writer\r\ndescription: \xe5\x86\x99\xe4\xbd\x9c\r\n---\r\n\r\n"
        b"# \xe6\xad\xa5\xe9\xaa\xa4\r\n\xe5\x85\x88\xe6\xbe\x84\xe6\xb8\x85\xe3\x80\x82\r\n"
    )
    skill_path.write_bytes(original)

    registry = SkillRegistry(tmp_path)
    skill_path.write_bytes(original.replace(b"\r\n", b"\n"))

    result = registry.load_skill("writer")

    assert "指纹" in result or "变更" in result or "不一致" in result
    assert "先澄清" not in result


def test_load_skill_detects_lf_to_crlf_byte_replacement(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    original = (
        b"---\nname: writer\ndescription: \xe5\x86\x99\xe4\xbd\x9c\n---\n\n"
        b"# \xe6\xad\xa5\xe9\xaa\xa4\n\xe5\x85\x88\xe6\xbe\x84\xe6\xb8\x85\xe3\x80\x82\n"
    )
    skill_path.write_bytes(original)

    registry = SkillRegistry(tmp_path)
    skill_path.write_bytes(original.replace(b"\n", b"\r\n"))

    result = registry.load_skill("writer")

    assert "指纹" in result or "变更" in result or "不一致" in result
