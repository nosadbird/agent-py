"""用 LangChain create_agent 高级 API 的简单示例（框架自动管理工具循环）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent_assistant.checkpointer import build_memory_checkpointer
from agent_assistant.config import Settings, load_settings
from agent_assistant.middleware import handle_tool_errors, summary_messages
from agent_assistant.model import create_chat_model
from agent_assistant.skills import SkillRegistry
from agent_assistant.subagents import call_subagent1
from agent_assistant.tools.agent_executor_files import ExecutorFileTools
from agent_assistant.tools.agent_executor_shell import ExecutorShellRunner, console_approve
from agent_assistant.tools.agent_executor_tools import get_current_datetime

choice = "not_stream"

_EXIT_COMMANDS = frozenset({"/exit", "exit", "quit"})

_SYSTEM_PROMPT_BASE = (
    "你是一个中文本地编码助手。需要时调用工具完成项目内的读写、搜索和命令执行，"
    "再根据工具返回结果继续作答。工具输出可能被截断。"
    "read_text、write_text、list_directory、search_text 只能访问项目目录内的相对路径。"
    "run_shell 在项目根目录执行。只读白名单命令会直接执行；"
    "危险命令必须等待用户审批，不得猜测审批结果。"
    "若任务与 Skills catalog 中的技能相关，先用 load_skill 读取完整说明，再按其步骤执行。"
)


def _build_system_prompt(skills: SkillRegistry) -> str:
    return (
        f"{_SYSTEM_PROMPT_BASE}\n\n"
        f"Skills catalog（需要时使用 load_skill 按需读取）：\n{skills.catalog_text()}"
    )


def recursion_limit_for(settings: Settings) -> int:
    """把配置中的步数上限折成 LangGraph recursion_limit。

    一轮工具调用通常经过模型节点和工具节点，因此按步数的两倍设置。
    """
    return settings.agent_max_steps * 2


def _run_config(agent: Any) -> dict[str, Any]:
    bound = getattr(agent, "config", None)
    config: dict[str, Any] = {"configurable": {"thread_id": "1"}}
    if isinstance(bound, dict):
        limit = bound.get("recursion_limit")
        if isinstance(limit, int) and limit > 0:
            config["recursion_limit"] = limit
    return config


def build_demo_agent(model: BaseChatModel | None = None, settings: Settings | None = None):
    """创建带项目文件、搜索、Shell 和技能工具的 create_agent 实例。"""
    if settings is None:
        settings = load_settings(Path.cwd())
    if model is None:
        model = create_chat_model(settings)
    root = Path.cwd()
    skills = SkillRegistry(root)
    files = ExecutorFileTools(root, max_chars=settings.shell_max_output_chars)
    shell = ExecutorShellRunner(
        root,
        timeout=settings.shell_timeout_seconds,
        max_chars=settings.shell_max_output_chars,
        approve=console_approve,
    )
    graph = create_agent(
        model=model,
        tools=[
            get_current_datetime,
            call_subagent1,
            *files.as_tools(),
            *shell.as_tools(),
            *skills.as_tools(),
        ],
        system_prompt=_build_system_prompt(skills),
        middleware=[handle_tool_errors, summary_messages],
        checkpointer=build_memory_checkpointer(),
    )
    return graph.with_config({"recursion_limit": recursion_limit_for(settings)})


def ask_agent(agent: Any, message: str) -> str:
    """调用 Agent 并返回最终文本。"""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    state = agent.invoke(
        {"messages": [HumanMessage(content=cleaned)]},
        _run_config(agent),
    )
    return final_text(state)


def final_text(state: dict[str, Any]) -> str:
    """从 Agent 返回状态中提取最终文本回答。"""
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "模型没有返回有效内容。"


def ask_agent_stream(agent, message: str) -> str:
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=message)]},
        _run_config(agent),
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")


def main() -> None:
    root = Path.cwd()
    try:
        settings = load_settings(root)
    except Exception:
        print(
            "配置加载失败：请参考 .env.example 创建 .env，"
            "并填写有效的 DASHSCOPE_API_KEY。"
        )
        return

    agent = build_demo_agent(settings=settings)
    print("高级 API 示例已启动（create_agent 自动管理工具循环）。输入 /exit 退出。")
    try:
        while True:
            user_input = input("你：").strip()
            if not user_input:
                continue
            if user_input.lower() in _EXIT_COMMANDS:
                print("已退出。")
                return

            if choice == "stream":
                ask_agent_stream(agent, user_input)
            else:
                print(ask_agent(agent, user_input))
    except (KeyboardInterrupt, EOFError):
        print("已取消并退出。")


if __name__ == "__main__":
    main()
