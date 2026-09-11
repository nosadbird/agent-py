"""Agent Assistant 的交互式命令行入口。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path

from agent_assistant.agent import AgentAssistant, build_assistant
from agent_assistant.config import load_settings

_HELP = (
    "可用命令：\n"
    "  /help   显示帮助\n"
    "  /clear  清除当前会话的短期记忆\n"
    "  /exit   退出（也可输入 exit 或 quit）"
)


def make_shell_approval(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Callable[[str, str], bool]:
    """创建显示完整命令与原因、仅接受 y/yes 的 Shell 审批函数。"""

    def approve(command: str, reason: str) -> bool:
        output_fn(f"检测到危险 Shell 命令：\n{command}")
        output_fn(f"风险原因：{reason}")
        try:
            answer = input_fn("是否执行？仅输入 y/yes 表示同意：")
        except (KeyboardInterrupt, EOFError):
            output_fn("审批已取消，命令未执行。")
            return False
        return answer.strip().lower() in {"y", "yes"}

    return approve


async def run_cli(
    assistant: AgentAssistant,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    session_id: str | None = None,
) -> None:
    """运行可处理中断、清除与退出命令的交互循环。"""
    current_session = session_id or str(uuid.uuid4())
    output_fn(
        f"Agent Assistant 已启动：{len(assistant.tools)} 个工具，"
        f"{len(assistant.skills.list_skills())} 个 Skills。"
    )
    for warning in assistant.warnings:
        output_fn(f"警告：{warning}")
    output_fn("输入 /help 查看命令。")
    assistant.set_tool_event_callback(output_fn)

    while True:
        try:
            raw = input_fn("你：")
        except KeyboardInterrupt:
            output_fn("已取消当前输入，可继续。")
            continue
        except EOFError:
            output_fn("输入结束，已退出。")
            return

        message = raw.strip()
        command = message.lower()
        if command in {"/exit", "exit", "quit"}:
            output_fn("已退出。")
            return
        if command == "/help":
            output_fn(_HELP)
            continue
        if command == "/clear":
            output_fn(assistant.clear_session(current_session))
            continue
        if not message:
            continue

        try:
            answer = await assistant.ask(message, current_session)
        except (KeyboardInterrupt, asyncio.CancelledError):
            output_fn("当前轮已取消，可继续。")
            continue
        output_fn(answer)


def main() -> None:
    """加载项目配置并启动 CLI，配置错误不显示 traceback。"""
    root = Path.cwd()
    try:
        settings = load_settings(root)
    except Exception:
        print(
            "配置加载失败：请参考 .env.example 创建 .env，"
            "并填写有效的 DASHSCOPE_API_KEY。"
        )
        return

    approval = make_shell_approval()

    async def start() -> None:
        assistant = await build_assistant(settings, approval)
        await run_cli(assistant)

    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("已取消并退出。")
    except Exception:
        print("Agent Assistant 启动失败，请检查配置和依赖后重试。")


if __name__ == "__main__":
    main()
