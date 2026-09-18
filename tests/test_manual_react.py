"""手写 ReAct 循环的核心主流程测试。"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import PrivateAttr

from agent_assistant import manual_react_demo
from agent_assistant.manual_react_demo import manual_react


class SequenceModel(BaseChatModel):
    """按顺序返回预设消息，并记录每轮收到的完整消息。"""

    _responses: list[AIMessage] = PrivateAttr()
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[BaseTool] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "sequence-model"

    def bind_tools(
        self,
        tools: list[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        self._bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=self._responses.pop(0))]
        )


def _async_tool(
    name: str,
    coroutine: Any,
) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=f"{name} 测试工具",
    )


@pytest.mark.asyncio
async def test_tool_call_then_final_answer_preserves_id_and_result() -> None:
    async def echo(text: str) -> str:
        return f"结果：{text}"

    tool = _async_tool("echo", echo)
    model = SequenceModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"text": "你好"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("最终回答"),
        ]
    )

    assert await manual_react(model, [tool], "问题") == "最终回答"
    assert model._bound_tools == [tool]
    observation = model._calls[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "call-1"
    assert observation.content == "结果：你好"


@pytest.mark.asyncio
async def test_returns_direct_answer_without_tool_call() -> None:
    model = SequenceModel([AIMessage("直接回答")])

    assert await manual_react(model, [], "问题") == "直接回答"
    assert len(model._calls) == 1


@pytest.mark.asyncio
async def test_unknown_tool_becomes_observation_and_model_can_recover() -> None:
    model = SequenceModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "missing",
                        "args": {},
                        "id": "call-unknown",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("已改用无需工具的回答"),
        ]
    )

    assert await manual_react(model, [], "问题") == "已改用无需工具的回答"
    observation = model._calls[1][-1]
    assert isinstance(observation, ToolMessage)
    assert "未知工具" in str(observation.content)


@pytest.mark.asyncio
async def test_tool_error_becomes_observation_and_model_can_recover() -> None:
    async def fail(value: str) -> str:
        raise RuntimeError("boom")

    tool = _async_tool("fail", fail)
    model = SequenceModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "fail",
                        "args": {"value": "x"},
                        "id": "call-error",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage("工具失败后的回答"),
        ]
    )

    assert await manual_react(model, [tool], "问题") == "工具失败后的回答"
    observation = model._calls[1][-1]
    assert isinstance(observation, ToolMessage)
    assert "执行失败" in str(observation.content)


@pytest.mark.asyncio
async def test_empty_answer_returns_chinese_guidance() -> None:
    model = SequenceModel([AIMessage(content=[])])

    assert "模型没有返回有效内容" in await manual_react(model, [], "问题")


@pytest.mark.asyncio
async def test_model_failure_returns_chinese_guidance_without_details() -> None:
    model = SequenceModel([])

    answer = await manual_react(model, [], "问题")

    assert "模型调用失败" in answer
    assert "pop from empty list" not in answer


@pytest.mark.asyncio
async def test_reaching_two_step_limit_returns_actual_limit() -> None:
    calls = [
        {
            "name": "missing",
            "args": {},
            "id": f"call-{index}",
            "type": "tool_call",
        }
        for index in (1, 2)
    ]
    model = SequenceModel(
        [AIMessage("", tool_calls=[tool_call]) for tool_call in calls]
    )

    answer = await manual_react(model, [], "问题", max_steps=2)

    assert "2" in answer
    assert "上限" in answer
    assert len(model._calls) == 2


def test_demo_main_uses_manual_react_default_step_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DemoSettings:
        shell_timeout_seconds = 3
        shell_max_output_chars = 1000
        agent_max_steps = 99

    captured: dict[str, object] = {}

    async def fake_manual_react(
        model: object,
        tools: object,
        user_input: str,
        max_steps: int = 8,
        system_prompt: str | None = None,
    ) -> str:
        captured["max_steps"] = max_steps
        captured["system_prompt"] = system_prompt
        captured["tool_names"] = (
            [getattr(tool, "name", None) for tool in tools]
            if isinstance(tools, list)
            else tools
        )
        return "回答"

    inputs = iter(["问题", "/exit"])

    monkeypatch.setattr(manual_react_demo, "load_settings", lambda root: DemoSettings())
    monkeypatch.setattr(manual_react_demo, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(manual_react_demo, "build_builtin_tools", lambda **kwargs: [])
    monkeypatch.setattr(manual_react_demo, "manual_react", fake_manual_react)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    manual_react_demo.manual_demo_main()

    assert captured["max_steps"] == 8
    assert "回答" in capsys.readouterr().out


def test_demo_main_initialization_failure_has_chinese_message_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(manual_react_demo, "load_settings", lambda root: object())

    def fail_model(settings: object) -> object:
        raise RuntimeError("do-not-leak")

    monkeypatch.setattr(manual_react_demo, "create_chat_model", fail_model)

    manual_react_demo.manual_demo_main()

    output = capsys.readouterr()
    assert "初始化失败" in output.out
    assert "do-not-leak" not in output.out + output.err
    assert "Traceback" not in output.out + output.err
