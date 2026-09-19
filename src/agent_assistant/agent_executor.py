"""用 LangChain create_agent 高级 API 的简单示例（框架自动管理工具循环）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent_assistant.config import load_settings
from agent_assistant.model import create_chat_model

_EXIT_COMMANDS = frozenset({"/exit", "exit", "quit"})


@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数的和，返回 a + b。"""
    return a + b


@tool
def get_current_datetime() -> str:
    """获取当前本地日期和时间，格式为 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _final_text(state: dict) -> str:
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

    model = create_chat_model(settings)
    tools = [add_numbers, get_current_datetime]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "你是一个中文助手，根据需要选择是否使用工具，根据工具结果用中文简洁回答。"
        ),
    )

    print("高级 API 示例已启动（create_agent 自动管理工具循环）。输入 /exit 退出。")
    try:
        while True:
            user_input = input("你：").strip()
            if not user_input:
                continue
            if user_input.lower() in _EXIT_COMMANDS:
                print("已退出。")
                return
            state = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            print(_final_text(state))
    except (KeyboardInterrupt, EOFError):
        print("已取消并退出。")


if __name__ == "__main__":
    main()
