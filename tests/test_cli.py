"""交互式 CLI 主流程核心测试。"""

from __future__ import annotations

from typing import Callable

import pytest

from agent_assistant import cli


class FakeSkills:
    def list_skills(self) -> list[object]:
        return [object()]


class FakeAssistant:
    def __init__(
        self,
        ask_effects: list[object] | None = None,
        clear_result: str = "已清除当前会话的短期记忆。",
    ) -> None:
        self.tools = [object(), object()]
        self.skills = FakeSkills()
        self.warnings = ["MCP 已降级"]
        self.ask_effects = list(ask_effects or ["回答"])
        self.asked: list[tuple[str, str]] = []
        self.cleared: list[str] = []
        self.clear_result = clear_result
        self.tool_event_callback: Callable[[str], None] | None = None

    def set_tool_event_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        self.tool_event_callback = callback

    async def ask(self, message: str, session_id: str) -> str:
        self.asked.append((message, session_id))
        if self.tool_event_callback is not None:
            self.tool_event_callback("工具调用：sample")
        effect = self.ask_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return str(effect)

    def clear_session(self, session_id: str) -> str:
        self.cleared.append(session_id)
        return self.clear_result


def _input_sequence(*effects: object) -> Callable[[str], str]:
    pending = list(effects)

    def fake_input(prompt: str) -> str:
        effect = pending.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return str(effect)

    return fake_input


@pytest.mark.asyncio
async def test_exit_does_not_call_agent_and_startup_shows_counts() -> None:
    assistant = FakeAssistant()
    output: list[str] = []

    await cli.run_cli(
        assistant,  # type: ignore[arg-type]
        input_fn=_input_sequence("/exit"),
        output_fn=output.append,
        session_id="fixed",
    )

    assert assistant.asked == []
    joined = "\n".join(output)
    assert "2" in joined
    assert "1" in joined
    assert "/help" in joined
    assert "MCP 已降级" in joined


@pytest.mark.asyncio
async def test_clear_help_and_one_question() -> None:
    assistant = FakeAssistant(["这是回答"])
    output: list[str] = []

    await cli.run_cli(
        assistant,  # type: ignore[arg-type]
        input_fn=_input_sequence("/help", "/clear", "你好", "quit"),
        output_fn=output.append,
        session_id="fixed",
    )

    assert assistant.cleared == ["fixed"]
    assert assistant.asked == [("你好", "fixed")]
    joined = "\n".join(output)
    assert "/clear" in joined
    assert "已清除" in joined
    assert "这是回答" in joined
    assert output.index("工具调用：sample") < output.index("这是回答")


@pytest.mark.asyncio
async def test_clear_failure_is_printed_and_cli_continues() -> None:
    assistant = FakeAssistant(["继续回答"], clear_result="短期记忆清除失败，请稍后重试。")
    output: list[str] = []

    await cli.run_cli(
        assistant,  # type: ignore[arg-type]
        input_fn=_input_sequence("/clear", "继续", "/exit"),
        output_fn=output.append,
        session_id="fixed",
    )

    assert "短期记忆清除失败，请稍后重试。" in output
    assert assistant.asked == [("继续", "fixed")]
    assert "继续回答" in output


@pytest.mark.asyncio
async def test_keyboard_interrupt_during_input_and_turn_continues_then_eof() -> None:
    assistant = FakeAssistant([KeyboardInterrupt(), "恢复回答"])
    output: list[str] = []

    await cli.run_cli(
        assistant,  # type: ignore[arg-type]
        input_fn=_input_sequence(KeyboardInterrupt(), "第一次", "第二次", EOFError()),
        output_fn=output.append,
        session_id="fixed",
    )

    assert assistant.asked == [("第一次", "fixed"), ("第二次", "fixed")]
    assert "恢复回答" in output
    assert sum("已取消" in line for line in output) == 2


def test_shell_approval_requires_explicit_yes() -> None:
    output: list[str] = []
    approval = cli.make_shell_approval(_input_sequence("y"), output.append)

    assert approval("Remove-Item data.txt", "命令包含删除操作") is True
    joined = "\n".join(output)
    assert "Remove-Item data.txt" in joined
    assert "命令包含删除操作" in joined

    denied = cli.make_shell_approval(_input_sequence("ok"), output.append)
    assert denied("unknown", "风险未知") is False


def test_main_missing_api_key_prints_chinese_guidance_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_settings(root: object) -> object:
        raise ValueError("DASHSCOPE_API_KEY missing")

    monkeypatch.setattr(cli, "load_settings", fail_settings)

    cli.main()

    output = capsys.readouterr()
    assert "DASHSCOPE_API_KEY" in output.out
    assert ".env.example" in output.out
    assert "Traceback" not in output.out + output.err


def test_main_build_failure_prints_chinese_guidance_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda root: object())

    async def fail_build(*args: object, **kwargs: object) -> object:
        raise RuntimeError("do-not-leak")

    monkeypatch.setattr(cli, "build_assistant", fail_build)

    cli.main()

    output = capsys.readouterr()
    assert "启动失败" in output.out
    assert "do-not-leak" not in output.out + output.err
    assert "Traceback" not in output.out + output.err
