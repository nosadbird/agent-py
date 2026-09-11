"""主 Agent 组装与调用流程的核心测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import SecretStr

from agent_assistant.agent import AgentAssistant, build_assistant, final_text
from agent_assistant.config import Settings
from agent_assistant.memory import FileMemoryStore
from agent_assistant.skills import SkillRegistry


def _settings(root: Path) -> Settings:
    return Settings(
        project_root=root,
        dashscope_api_key=SecretStr("do-not-leak"),
        agent_max_steps=7,
    )


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: name,
        name=name,
        description=f"{name} tool",
    )


class FakeGraph:
    def __init__(
        self,
        result: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or {}
        self.error = error
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def ainvoke(
        self,
        payload: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((payload, config))
        if self.error is not None:
            raise self.error
        return self.result


class Factory:
    def __init__(self, graph: FakeGraph) -> None:
        self.graph = graph
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeGraph:
        self.calls.append(kwargs)
        return self.graph


def test_final_text_supports_string_content_blocks_and_empty_state() -> None:
    assert final_text({"messages": [HumanMessage("忽略"), AIMessage("完成")]}) == "完成"
    assert (
        final_text(
            {
                "messages": [
                    AIMessage(
                        content=[
                            {"type": "text", "text": "第一段"},
                            {"type": "image", "url": "ignored"},
                            {"type": "text", "text": "第二段"},
                        ]
                    )
                ]
            }
        )
        == "第一段\n第二段"
    )
    assert final_text({}) == ""


@pytest.mark.asyncio
async def test_ask_builds_dynamic_prompt_invokes_graph_and_saves_memory(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: 代码评审\n---\n步骤",
        encoding="utf-8",
    )
    memory = FileMemoryStore(tmp_path, max_turns=3)
    memory.append_turn("s1", "旧问题", "旧答案")
    graph = FakeGraph({"messages": [AIMessage("新答案")]})
    factory = Factory(graph)
    tools = [_tool("sample")]
    assistant = AgentAssistant(
        settings=_settings(tmp_path),
        model=object(),  # type: ignore[arg-type]
        tools=tools,
        memory=memory,
        skills=SkillRegistry(tmp_path),
        agent_factory=factory,
    )

    answer = await assistant.ask("新问题", "s1")

    assert answer == "新答案"
    assert factory.calls[0]["tools"] is tools
    prompt = str(factory.calls[0]["system_prompt"])
    assert "中文" in prompt
    assert "私有思维链" in prompt
    assert "review: 代码评审" in prompt
    assert "旧问题" in prompt
    assert graph.calls == [
        (
            {"messages": [{"role": "user", "content": "新问题"}]},
            {"recursion_limit": 14},
        )
    ]
    assert [record.content for record in memory.recent_turns("s1")][-2:] == [
        "新问题",
        "新答案",
    ]


@pytest.mark.asyncio
async def test_ask_rejects_blank_and_does_not_store_failed_or_empty_response(
    tmp_path: Path,
) -> None:
    memory = FileMemoryStore(tmp_path, max_turns=3)
    blank_assistant = AgentAssistant(
        _settings(tmp_path),
        object(),  # type: ignore[arg-type]
        [],
        memory,
        SkillRegistry(tmp_path),
        agent_factory=Factory(FakeGraph()),
    )

    with pytest.raises(ValueError, match="不能为空"):
        await blank_assistant.ask("  ", "s1")

    assert "没有返回有效" in await blank_assistant.ask("问题", "s1")
    assert memory.recent_turns("s1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("401 do-not-leak"), "鉴权"),
        (RuntimeError("429 do-not-leak"), "频繁"),
        (TimeoutError("do-not-leak"), "超时"),
        (RuntimeError("recursion limit do-not-leak"), "步数"),
        (RuntimeError("other do-not-leak"), "失败"),
    ],
)
async def test_ask_maps_errors_without_leaking_key(
    tmp_path: Path,
    error: Exception,
    expected: str,
) -> None:
    memory = FileMemoryStore(tmp_path, max_turns=3)
    assistant = AgentAssistant(
        _settings(tmp_path),
        object(),  # type: ignore[arg-type]
        [],
        memory,
        SkillRegistry(tmp_path),
        agent_factory=Factory(FakeGraph(error=error)),
    )

    answer = await assistant.ask("问题", "s1")

    assert expected in answer
    assert "do-not-leak" not in answer
    assert memory.recent_turns("s1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_method", ["format_context", "append_turn"])
async def test_ask_degrades_when_memory_io_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_method: str,
) -> None:
    memory = FileMemoryStore(tmp_path, max_turns=3)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("sensitive-path")

    monkeypatch.setattr(memory, failure_method, fail)
    graph = FakeGraph({"messages": [AIMessage("仍然回答")]})
    assistant = AgentAssistant(
        _settings(tmp_path),
        object(),  # type: ignore[arg-type]
        [],
        memory,
        SkillRegistry(tmp_path),
        agent_factory=Factory(graph),
    )

    answer = await assistant.ask("问题", "s1")

    assert "仍然回答" in answer
    assert "警告" in answer
    assert "sensitive-path" not in answer
    assert len(graph.calls) == 1


@pytest.mark.asyncio
async def test_ask_reports_tool_events_without_thoughts(tmp_path: Path) -> None:
    api_key = "do-not-leak"
    state = {
        "messages": [
            AIMessage(
                "private thought must not appear",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {
                            "api_key": api_key,
                            "nested": {"token": "nested-token"},
                            "query": (
                                "password=pw-value "
                                'authorization: "Bearer auth-token" '
                                '{"secret": "json-secret"} '
                                + "x" * 300
                            ),
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    f"result DASHSCOPE_API_KEY={api_key} "
                    "token=result-token Bearer output-token "
                    + "y" * 300
                ),
                tool_call_id="call-1",
                name="lookup",
                status="success",
            ),
            AIMessage("最终答案"),
        ]
    }
    events: list[str] = []
    assistant = AgentAssistant(
        _settings(tmp_path),
        object(),  # type: ignore[arg-type]
        [],
        FileMemoryStore(tmp_path, max_turns=3),
        SkillRegistry(tmp_path),
        agent_factory=Factory(FakeGraph(state)),
    )
    assistant.set_tool_event_callback(events.append)

    assert await assistant.ask("问题", "s1") == "最终答案"
    assert len(events) == 2
    assert "lookup" in events[0] and "参数" in events[0]
    assert "success" in events[1] and "result" in events[1]
    assert all(len(event) < 300 for event in events)
    assert "private thought" not in "\n".join(events)
    joined = "\n".join(events)
    for secret in (
        api_key,
        "nested-token",
        "pw-value",
        "auth-token",
        "json-secret",
        "result-token",
        "output-token",
    ):
        assert secret not in joined
    assert "已隐藏" in joined


@pytest.mark.asyncio
async def test_build_assistant_aggregates_tools_and_keeps_first_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_assistant import agent as agent_module

    builtin = [_tool("same"), _tool("builtin")]
    skill_tools = [_tool("skill")]
    mcp_tools = [_tool("same"), _tool("mcp")]

    class FakeSkills:
        warnings = ["skill warning"]

        def as_tools(self) -> list[StructuredTool]:
            return skill_tools

        def list_skills(self) -> list[object]:
            return []

        def catalog_text(self) -> str:
            return "catalog"

    monkeypatch.setattr(agent_module, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(
        agent_module,
        "build_builtin_tools",
        lambda **kwargs: builtin,
    )
    monkeypatch.setattr(agent_module, "SkillRegistry", lambda root: FakeSkills())
    monkeypatch.setattr(
        agent_module,
        "load_mcp_config_resilient",
        lambda path: (
            {"bad": {"transport": "stdio", "command": "python"}},
            ["config warning"],
        ),
    )

    async def fake_mcp_loader(config: object) -> tuple[list[StructuredTool], list[str]]:
        return mcp_tools, ["MCP bad"]

    monkeypatch.setattr(agent_module, "load_mcp_tools_resilient", fake_mcp_loader)

    assistant = await build_assistant(_settings(tmp_path), lambda command, reason: False)

    assert [tool.name for tool in assistant.tools] == [
        "same",
        "builtin",
        "skill",
        "remember_fact",
        "forget_fact",
        "mcp",
    ]
    assert assistant.tools[0] is builtin[0]
    assert "skill warning" in assistant.warnings
    assert "config warning" in assistant.warnings
    assert "MCP bad" in assistant.warnings
    assert any("same" in warning and "重复" in warning for warning in assistant.warnings)
    assert assistant.memory._root == tmp_path


def test_clear_session_returns_chinese_status_and_degrades_on_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = FileMemoryStore(tmp_path, max_turns=3)
    memory.append_turn("s1", "问题", "回答")
    assistant = AgentAssistant(
        _settings(tmp_path),
        object(),  # type: ignore[arg-type]
        [],
        memory,
        SkillRegistry(tmp_path),
    )

    assert "已清除" in assistant.clear_session("s1")

    assert memory.recent_turns("s1") == []

    def fail(session_id: str) -> None:
        raise OSError("do-not-leak")

    monkeypatch.setattr(memory, "clear_session", fail)
    failure = assistant.clear_session("s1")
    assert "清除失败" in failure
    assert "do-not-leak" not in failure
