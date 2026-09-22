"""用 LangChain create_agent 高级 API 的简单示例（框架自动管理工具循环）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

from agent_assistant.config import Settings, load_settings
from agent_assistant.model import create_chat_model

_EXIT_COMMANDS = frozenset({"/exit", "exit", "quit"})
_SYSTEM_PROMPT = (
    "你是一个乐于助人且专业的助手，根据需要选择是否使用工具，尽可能给出准确的回答。"
)


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    final_message = {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
    return final_message


@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数的和，返回 a + b。"""
    raise Exception("错误描述信息")
    # return a + b


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
        middleware=[handle_tool_errors,trim_messages],
        checkpointer=InMemorySaver() ### 内存存储
    )


def ask_agent(agent: Any, message: str) -> str:
    """调用 Agent 并返回最终文本。"""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    state = agent.invoke({"messages": [HumanMessage(content=cleaned)]},
                         {"configurable": {"thread_id": "1"}})
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
