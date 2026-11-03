"""用 LangChain create_agent 高级 API 的简单示例（框架自动管理工具循环）。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_assistant.agent_events import AgentEvent, EventHandler
from agent_assistant.checkpointer import build_memory_checkpointer
from agent_assistant.config import Settings, load_settings
from agent_assistant.middleware import handle_tool_errors, summary_messages
from agent_assistant.skills import SkillRegistry
from agent_assistant.thinking_output import (
    create_thinking_model,
    reasoning_from_state,
    reasoning_text,
)
from agent_assistant.tools.agent_executor_files import (
    ExecutorFileTools,
    confirm_file_change,
)
from agent_assistant.tools.agent_executor_shell import ExecutorShellRunner, console_approve
from agent_assistant.tools.agent_executor_tools import get_current_datetime, update_plan

choice = "stream"

_EXIT_COMMANDS = frozenset({"/exit", "exit", "quit"})

_SYSTEM_PROMPT_BASE = (
    "你是一个本地智能 Agent 助手。需要时调用工具，再根据工具返回结果完成用户任务。"
    "工具输出可能被截断。多步骤任务先调用 update_plan 公布简短行动计划，简单问答无需计划。"
    "思考过程通过专用字段展示，正式回答仍应简洁清楚。"
    "run_shell 在项目根目录执行；需要审批时必须等待用户决定，不得猜测审批结果。"
    "修改代码后应运行最小相关测试，并如实说明验证是否执行及结果。"
    "若任务与 Skills catalog 中的技能相关，先用 load_skill 读取完整说明，再按其步骤执行。"
)


def _build_system_prompt(skills: SkillRegistry) -> str:
    return (
        f"{_SYSTEM_PROMPT_BASE}\n\n"
        f"Skills catalog（需要时使用 load_skill 按需读取skills）：\n{skills.catalog_text()}"
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


def build_demo_agent(
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    on_event: EventHandler | None = None,
):
    """创建带本地工具和skills的 create_agent 实例。"""
    if settings is None:
        settings = load_settings(Path.cwd())
    if model is None:
        model = create_thinking_model(settings)
    root = Path.cwd()
    skills = SkillRegistry(root)
    files = ExecutorFileTools(
        root,
        max_chars=settings.shell_max_output_chars,
        approve=confirm_file_change,
        on_event=on_event,
    )
    shell = ExecutorShellRunner(
        root,
        timeout=settings.shell_timeout_seconds,
        max_chars=settings.shell_max_output_chars,
        approve=console_approve,
        shell_name=settings.agent_shell,
        bash_path=settings.agent_bash_path,
        on_event=on_event,
    )
    graph = create_agent(
        model=model,
        tools=[
            get_current_datetime,
            update_plan,
            *files.as_tools(),
            *shell.as_tools(),
            *skills.as_tools(),
        ],
        system_prompt=_build_system_prompt(skills),
        middleware=[handle_tool_errors, summary_messages],
        checkpointer=build_memory_checkpointer(),
    )
    return graph.with_config({"recursion_limit": recursion_limit_for(settings)})


@dataclass(frozen=True)
class AgentTurn:
    """一轮对话的最终回答和真实运行事件。"""

    reply: str
    events: tuple[AgentEvent, ...] = ()
    thinking: str = ""


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _tool_summary(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return name
    if name == "run_shell":
        return f"{name}: {args.get('command', '')}"
    if "path" in args:
        return f"{name}: {args['path']}"
    if name == "load_skill":
        return f"{name}: {args.get('name', '')}"
    return name


def events_from_state(state: dict[str, Any]) -> tuple[AgentEvent, ...]:
    """从最终状态提取可审计的工具事件，不包含模型内部推理。"""
    events: list[AgentEvent] = []
    messages = state.get("messages") or []
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index
    for message in messages[start:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                name = str(call.get("name") or "未知工具")
                events.append(AgentEvent("tool_started", _tool_summary(name, call.get("args"))))
        elif isinstance(message, ToolMessage):
            text = _text_content(message.content).strip()
            events.append(AgentEvent("tool_completed", text))
    return tuple(events)


def ask_agent(agent: Any, message: str) -> AgentTurn:
    """调用 Agent，返回最终回答和可观察事件。"""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    state = agent.invoke(
        {"messages": [HumanMessage(content=cleaned)]},
        _run_config(agent),
    )
    return AgentTurn(
        reply=final_text(state),
        events=events_from_state(state),
        thinking=reasoning_from_state(state),
    )


def final_text(state: dict[str, Any]) -> str:
    """从 Agent 返回状态中提取最终文本回答。"""
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "模型没有返回有效内容。"


def _tool_calls_from_chunks(
    token: AIMessage,
    pending: dict[int, dict[str, str]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """合并流式工具调用分片，参数形成完整 JSON 后返回。"""
    completed: list[tuple[str, str, dict[str, Any]]] = []
    for chunk in getattr(token, "tool_call_chunks", []):
        index = chunk.get("index")
        key = index if isinstance(index, int) else len(pending)
        item = pending.setdefault(key, {"id": "", "name": "", "args": ""})
        for field in ("id", "name", "args"):
            value = chunk.get(field)
            if isinstance(value, str):
                item[field] += value
        try:
            args = json.loads(item["args"])
        except json.JSONDecodeError:
            continue
        if item["name"] and isinstance(args, dict):
            completed.append((item["id"], item["name"], args))
    return completed


async def ask_agent_stream(agent: Any, message: str) -> None:
    """分别流式展示思考、真实工具事件和最终回答。"""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("用户输入不能为空")
    started_thinking = False
    thinking_active = False
    started_answer = False
    seen_calls: set[str] = set()
    tool_names: dict[str, str] = {}
    pending_calls: dict[int, dict[str, str]] = {}
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=cleaned)]},
        _run_config(agent),
        stream_mode="messages",
    ):
        token = chunk[0] if isinstance(chunk, tuple) else chunk
        if isinstance(token, AIMessage):
            reasoning = reasoning_text(token)
            if reasoning:
                if not thinking_active:
                    prefix = "\n" if started_thinking else ""
                    print(f"{prefix}思考：", flush=True)
                    started_thinking = True
                    thinking_active = True
                print(reasoning, end="", flush=True)
            calls: list[tuple[str, str, object]] = [
                (str(call.get("id") or ""), str(call.get("name") or "未知工具"), call.get("args"))
                for call in token.tool_calls
            ]
            if getattr(token, "tool_call_chunks", []):
                calls = _tool_calls_from_chunks(token, pending_calls)
            for call_id, name, args in calls:
                if call_id:
                    tool_names[call_id] = name
                marker = call_id or f"{name}:{args}"
                if marker not in seen_calls:
                    seen_calls.add(marker)
                    thinking_active = False
                    print(f"\n● {_tool_summary(name, args)}", flush=True)
            if not token.tool_calls and not getattr(token, "tool_call_chunks", []):
                content = _text_content(token.content)
                if content:
                    if not started_answer:
                        thinking_active = False
                        print("\n回答：", flush=True)
                        started_answer = True
                    print(content, end="", flush=True)
        elif isinstance(token, ToolMessage):
            # 下一次模型调用的工具 index 会从 0 重新开始。
            pending_calls.clear()
            name = token.name or tool_names.get(str(token.tool_call_id), "工具")
            result = _text_content(token.content).strip()
            if result:
                label = "计划" if name == "update_plan" else "结果"
                print(f"\n  └─ {label}：{result}", flush=True)
    print()


def _print_runtime_event(event: AgentEvent) -> None:
    """CLI 只即时打印无法从 LangGraph 消息流获得的终端增量。"""
    if event.kind == "terminal_output":
        print(f"  │ {event.text}", flush=True)


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

    agent = build_demo_agent(settings=settings, on_event=_print_runtime_event)
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
                asyncio.run(ask_agent_stream(agent, user_input))
            else:
                turn = ask_agent(agent, user_input)
                if turn.thinking:
                    print(f"思考：\n{turn.thinking}\n")
                for event in turn.events:
                    print(f"{event.kind}：{event.text}")
                print(f"回答：\n{turn.reply}")
    except (KeyboardInterrupt, EOFError):
        print("已取消并退出。")


if __name__ == "__main__":
    main()
