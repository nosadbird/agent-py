"""Demo agent 的模型前摘要与工具调用确认中间件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain.messages import RemoveMessage
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from agent_assistant.config import load_settings
from agent_assistant.model import create_chat_model
from agent_assistant.tools.agent_executor_files import truncate_text

_DEFAULT_OUTPUT_CHARS = 12_000


@before_model
def summary_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """将较早的对话历史总结为摘要，仅保留最近几条消息，避免超出上下文窗口。"""
    messages = state["messages"]

    # 阈值可以适当调大，避免消息太少时就触发（没有必要总结）
    if len(messages) <= 3:
        return None

    # 保留最近几条消息（奇偶判断保证不截断到"半轮对话"）
    keep_count = 4 if len(messages) % 2 == 0 else 5
    recent_messages = messages[-keep_count:]
    messages_to_summarize = messages[:-keep_count]

    if not messages_to_summarize:
        return None

    # 取出已有摘要（上一次裁剪时生成的），实现摘要的"滚动更新"
    existing_summary = state.get("summary", "")

    # 把需要总结的这部分消息拼接成文本
    conversation_text = "\n".join(
        f"{msg.type}: {msg.content}" for msg in messages_to_summarize
    )

    summary_prompt = (
        "你是一个对话摘要助手。请将下面的新对话内容整合进已有摘要中，"
        "输出一个简洁但完整的摘要，务必保留关键信息（如用户姓名、偏好、"
        "已经讨论过的事项、已经给出的结论等），不要遗漏重要细节。\n\n"
        f"已有摘要：\n{existing_summary or '（无）'}\n\n"
        f"新对话内容：\n{conversation_text}\n\n"
        "请输出更新后的摘要："
    )
    root = Path.cwd()
    try:
        settings = load_settings(root)
    except Exception:
        print(
            "配置加载失败：请参考 .env.example 创建 .env，"
            "并填写有效的 DASHSCOPE_API_KEY。"
        )
        return
    summary_model = create_chat_model(settings)
    summary_response = summary_model.invoke(summary_prompt)
    new_summary = summary_response.content

    summary_message = SystemMessage(
        content=f"[以下是此前对话的摘要，供参考]\n{new_summary}"
    )

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            summary_message,
            *recent_messages,
        ],
        "summary": new_summary,  # 更新持久化的摘要字段
    }


def output_char_limit() -> int:
    """读取配置中的工具输出上限；配置不可用时使用默认值。"""
    try:
        return load_settings(Path.cwd()).shell_max_output_chars
    except Exception:
        return _DEFAULT_OUTPUT_CHARS


def guard_tool_result(request, handler, max_chars: int):
    """执行工具，把异常变成模型可读的消息，并截断过长输出。"""
    try:
        result = handler(request)
    except Exception as exc:
        result = ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(exc)})",
            tool_call_id=request.tool_call["id"],
        )
    return truncate_tool_result(result, max_chars)


def truncate_tool_result(result, max_chars: int):
    """截断字符串或 ToolMessage 文本，其它返回值保持原样。"""
    if isinstance(result, str):
        return truncate_text(result, max_chars)
    if isinstance(result, ToolMessage) and isinstance(result.content, str):
        truncated = truncate_text(result.content, max_chars)
        if truncated == result.content:
            return result
        return result.model_copy(update={"content": truncated})
    return result


@wrap_tool_call
def handle_tool_errors(request, handler):
    """捕获工具异常并截断输出。危险命令的审批由 Shell 工具自己负责。"""
    return guard_tool_result(request, handler, output_char_limit())
