"""显式展示 tool-calling/ReAct 消息循环的教学 Demo。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent_assistant.config import load_settings
from agent_assistant.model import create_chat_model
from agent_assistant.tools import build_builtin_tools

_SYSTEM_PROMPT = (
    "你是一个中文本地助手。需要时调用提供的工具，根据工具返回结果继续作答。"
)
_EMPTY_RESPONSE = "模型没有返回有效内容，请换一种方式重试。"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str) and block.strip():
            parts.append(block.strip())
        elif isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


async def manual_react(
    model: BaseChatModel,
    tools: list[BaseTool],
    user_input: str,
    max_steps: int = 10,
) -> str:
    """运行显式工具调用循环，不输出模型私有推理过程。"""
    cleaned = user_input.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    if max_steps <= 0:
        raise ValueError("max_steps 必须为正整数")

    tool_by_name = {tool.name: tool for tool in tools}
    bound_model = model.bind_tools(tools)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=cleaned),
    ]

    for _ in range(max_steps):
        try:
            response = await bound_model.ainvoke(messages)
            print("--------response start-----------")
            print(response)
            print("--------response end-----------")
        except Exception:
            return "模型调用失败，请稍后重试或检查配置。"
        if not isinstance(response, AIMessage):
            return _EMPTY_RESPONSE
        messages.append(response)

        if not response.tool_calls:
            return _content_text(response.content) or _EMPTY_RESPONSE

        for tool_call in response.tool_calls:
            name = tool_call.get("name")
            call_id = str(tool_call.get("id") or "")
            args = tool_call.get("args")
            tool = tool_by_name.get(name) if isinstance(name, str) else None
            if tool is None:
                observation = f"未知工具：{name or '未提供名称'}"
            elif not isinstance(args, dict):
                observation = f"工具 {name} 参数非法：args 必须是对象"
            else:
                try:
                    observation = str(await tool.ainvoke(args))
                except Exception as exc:
                    observation = f"工具 {name} 执行失败：{type(exc).__name__}"
            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=call_id,
                    name=name if isinstance(name, str) else None,
                )
            )

    return f"已达到最大执行步数上限（{max_steps}），请缩小问题范围后重试。"


def manual_demo_main() -> None:
    """加载本地配置与安全工具，读取一次输入并运行手写循环。"""
    root = Path.cwd()
    try:
        settings = load_settings(root)
    except Exception:
        print(
            "配置加载失败：请参考 .env.example 创建 .env，"
            "并填写有效的 DASHSCOPE_API_KEY。"
        )
        return

    from agent_assistant.cli import make_shell_approval

    try:
        model = create_chat_model(settings)
        tools = build_builtin_tools(
            root=root,
            shell_timeout=settings.shell_timeout_seconds,
            max_chars=settings.shell_max_output_chars,
            approve=make_shell_approval(),
        )
    except Exception:
        print("手写 Demo 初始化失败，请检查模型配置和运行依赖。")
        return
    try:
        ### 改为多次循环
        while True :
            user_input = input("你：")
            answer = asyncio.run(
                manual_react(
                    model,
                    tools,
                    user_input,
                )
            )
            print(answer)
    except (KeyboardInterrupt, EOFError):
        print("已取消并退出。")
        return
    except ValueError as exc:
        print(f"输入无效：{exc}")
        return


if __name__ == "__main__":
    manual_demo_main()
