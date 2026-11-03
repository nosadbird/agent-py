from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import PrivateAttr

from agent_assistant.agent_executor import (
    _run_config,
    _tool_calls_from_chunks,
    ask_agent_stream,
    events_from_state,
    recursion_limit_for,
)
from agent_assistant.config import Settings
from agent_assistant.middleware import aguard_tool_result, guard_tool_result, handle_tool_errors


class SequenceModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "agent-executor-test"

    def bind_tools(
        self,
        tools: list[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


def test_recursion_limit_is_twice_the_configured_steps(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="test-key",
        agent_max_steps=7,
        _env_file=None,
    )
    assert recursion_limit_for(settings) == 14


def test_run_config_carries_bound_recursion_limit() -> None:
    agent = SimpleNamespace(config={"recursion_limit": 14})
    config = _run_config(agent)
    assert config["configurable"]["thread_id"] == "1"
    assert config["recursion_limit"] == 14


def test_guard_tool_result_truncates_without_asking_for_approval() -> None:
    request = SimpleNamespace(tool_call={"id": "call-1"})

    def handler(_request):
        return "x" * 50

    result = guard_tool_result(request, handler, max_chars=10)
    assert isinstance(result, str)
    assert result.startswith("x" * 10)
    assert "已截断" in result


def test_guard_tool_result_turns_exceptions_into_tool_messages() -> None:
    request = SimpleNamespace(tool_call={"id": "call-2"})

    def handler(_request):
        raise RuntimeError("boom")

    result = guard_tool_result(request, handler, max_chars=100)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-2"
    assert "工具错误" in result.content


@pytest.mark.asyncio
async def test_async_guard_executes_and_handles_errors() -> None:
    request = SimpleNamespace(tool_call={"id": "call-async"})

    async def success(_request):
        return ToolMessage("x" * 20, tool_call_id="call-async")

    result = await aguard_tool_result(request, success, max_chars=5)
    assert isinstance(result, ToolMessage)
    assert "已截断" in result.content

    async def failure(_request):
        raise RuntimeError("boom")

    failed = await aguard_tool_result(request, failure, max_chars=100)
    assert isinstance(failed, ToolMessage)
    assert "工具错误" in failed.content


def test_events_include_tools_but_never_reasoning() -> None:
    state = {
        "messages": [
            HumanMessage("检查"),
            AIMessage(
                "",
                additional_kwargs={"reasoning_content": "不应泄露"},
                tool_calls=[{"name": "read_text", "args": {"path": "a.py"}, "id": "1"}],
            ),
            ToolMessage("文件内容", tool_call_id="1", name="read_text"),
            AIMessage("完成", additional_kwargs={"reasoning_content": "仍不应泄露"}),
        ]
    }

    events = events_from_state(state)

    assert [event.kind for event in events] == ["tool_started", "tool_completed"]
    assert "read_text: a.py" in events[0].text
    assert all("不应泄露" not in event.text for event in events)


def test_tool_call_chunks_are_merged_before_display() -> None:
    pending: dict[int, dict[str, str]] = {}
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "run_", "args": '{"com', "id": "call-1", "index": 0}],
    )
    second = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "shell", "args": 'mand":"pwd"}', "id": None, "index": 0}],
    )

    assert _tool_calls_from_chunks(first, pending) == []
    assert _tool_calls_from_chunks(second, pending) == [
        ("call-1", "run_shell", {"command": "pwd"})
    ]


@pytest.mark.asyncio
async def test_stream_resets_tool_chunks_between_model_rounds(capsys) -> None:
    class FakeAgent:
        config = {}

        async def astream(self, *_args, **_kwargs):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "read_text", "args": '{"path":"a.py"}', "id": "1", "index": 0}
                ],
            )
            yield ToolMessage("first", tool_call_id="1", name="read_text")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "run_shell", "args": '{"command":"pwd"}', "id": "2", "index": 0}
                ],
            )
            yield ToolMessage("second", tool_call_id="2", name="run_shell")
            yield AIMessageChunk(content="done")

    await ask_agent_stream(FakeAgent(), "work")
    output = capsys.readouterr().out
    assert "read_text: a.py" in output
    assert "run_shell: pwd" in output
    assert "done" in output


@pytest.mark.asyncio
async def test_real_async_agent_can_pass_tool_middleware(capsys) -> None:
    async def current() -> str:
        return "2026-01-01"

    tool = StructuredTool.from_function(
        coroutine=current,
        name="current",
        description="返回测试时间",
    )
    model = SequenceModel(
        [
            AIMessage(
                "",
                tool_calls=[{"name": "current", "args": {}, "id": "time-1", "type": "tool_call"}],
            ),
            AIMessage("完成"),
        ]
    )
    agent = create_agent(model=model, tools=[tool], middleware=[handle_tool_errors])

    await ask_agent_stream(agent, "当前时间")

    output = capsys.readouterr().out
    assert "current" in output
    assert "2026-01-01" in output
    assert "完成" in output
