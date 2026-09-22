"""用 LangChain create_agent 高级 API 的简单示例（框架自动管理工具循环）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent_assistant.config import Settings, load_settings
from agent_assistant.model import create_chat_model

_EXIT_COMMANDS = frozenset({"/exit", "exit", "quit"})
_SYSTEM_PROMPT = (
    "你是一个乐于助人且专业的助手，根据需要选择是否使用工具，尽可能给出准确的回答。"
)


@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数的和，返回 a + b。"""
    return a + b


@tool
def get_current_datetime() -> str:
    """获取当前本地日期和时间，格式为 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_demo_agent(model: BaseChatModel | None = None, settings: Settings | None = None):
    """创建带求和/时间两个工具的 create_agent 实例。"""
    if model is None:
        if settings is None:
            settings = load_settings(Path.cwd())
        model = create_chat_model(settings)
    return create_agent(
        model=model,
        tools=[add_numbers, get_current_datetime],
        system_prompt=_SYSTEM_PROMPT,
    )


def ask_agent(agent: Any, message: str) -> str:
    """调用 Agent 并返回最终文本。"""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    state = agent.invoke({"messages": [HumanMessage(content=cleaned)]})
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
            print(ask_agent(agent, user_input))
    except (KeyboardInterrupt, EOFError):
        print("已取消并退出。")


if __name__ == "__main__":
    main()
