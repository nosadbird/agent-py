"""LangChain 主 Agent 的提示组装、调用与依赖聚合。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agent_assistant.config import Settings
from agent_assistant.mcp_client import (
    load_mcp_config_resilient,
    load_mcp_tools_resilient,
)
from agent_assistant.memory import FileMemoryStore
from agent_assistant.model import create_chat_model
from agent_assistant.skills import SkillRegistry
from agent_assistant.tools import build_builtin_tools

_EMPTY_RESPONSE = "模型没有返回有效文本，请换一种方式重试。"
_EVENT_SUMMARY_CHARS = 160
_REDACTED = "[已隐藏]"
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|token|secret|password|authorization)$",
    re.IGNORECASE,
)
_INLINE_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (
        ["']?
        \b[\w.-]*(?:api[_-]?key|token|secret|password|authorization)\b
        ["']?\s*(?:=|:)\s*
    )
    (?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}\]]+)
    """
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,'\"}\]]+")


class _RememberFactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(..., description="要长期记住的稳定事实或偏好")
    tags: list[str] | None = Field(None, description="可选分类标签")


class _ForgetFactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(..., description="要删除的长期记忆 ID")


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str) and block.strip():
            parts.append(block.strip())
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


def final_text(state: dict[str, object]) -> str:
    """从 Agent 状态中提取最后一条含文本的 AIMessage。"""
    messages = state.get("messages")
    if not isinstance(messages, (list, tuple)):
        return ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _content_text(message.content)
            if text:
                return text
    return ""


def _system_prompt(skills: SkillRegistry, memory_context: str) -> str:
    return (
        "你是一个使用中文协助用户完成项目任务的本地 Agent 助手。\n"
        "安全边界：文件和 Shell 操作必须遵守工具自身限制；危险操作必须等待用户审批，"
        "不得猜测审批结果，不得泄露密钥、完整环境变量或其他敏感信息。\n"
        "只展示结论、必要步骤和工具结果，不得输出私有思维链。\n\n"
        f"Skills catalog（需要时使用 load_skill 按需读取）：\n{skills.catalog_text()}\n\n"
        f"当前记忆上下文：\n{memory_context}"
    )


def _redact_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: (
                _REDACTED
                if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key)
                else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def _summary(value: object, secret_values: tuple[str, ...] = ()) -> str:
    text = " ".join(str(_redact_value(value)).split())
    for secret in secret_values:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _INLINE_SECRET_PATTERN.sub(lambda match: match.group(1) + _REDACTED, text)
    text = _BEARER_PATTERN.sub(f"Bearer {_REDACTED}", text)
    if len(text) > _EVENT_SUMMARY_CHARS:
        return text[:_EVENT_SUMMARY_CHARS] + "……"
    return text


def _tool_events(
    state: dict[str, object], secret_values: tuple[str, ...] = ()
) -> list[str]:
    messages = state.get("messages")
    if not isinstance(messages, (list, tuple)):
        return []
    events: list[str] = []
    names_by_call_id: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                name = str(tool_call.get("name") or "未知工具")
                call_id = str(tool_call.get("id") or "")
                if call_id:
                    names_by_call_id[call_id] = name
                events.append(
                    "工具调用："
                    f"{name}；参数摘要："
                    f"{_summary(tool_call.get('args', {}), secret_values)}"
                )
        elif isinstance(message, ToolMessage):
            name = message.name or names_by_call_id.get(
                str(message.tool_call_id), "未知工具"
            )
            status = getattr(message, "status", "success")
            events.append(
                f"工具结果：{name}；状态：{status}；结果摘要："
                f"{_summary(message.content, secret_values)}"
            )
    return events


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    response_value = getattr(response, "status_code", None)
    return response_value if isinstance(response_value, int) else None


def _error_message(error: Exception) -> str:
    """把运行异常映射为不回显原始异常的中文提示。"""
    status = _status_code(error)
    name = type(error).__name__.lower()
    text = str(error).lower()
    if status in {401, 403} or "401" in text or "403" in text or "auth" in name:
        return "模型鉴权失败，请检查 API Key、地域、模型和账户权限。"
    if status == 429 or "429" in text or "ratelimit" in name or "rate limit" in text:
        return "请求过于频繁，已触发限流，请稍后重试。"
    if isinstance(error, TimeoutError) or "timeout" in name or "timed out" in text:
        return "模型请求超时，请稍后重试。"
    if (
        "graphrecursion" in name
        or "recursion" in text
        or "recursion" in name
        or "step limit" in text
    ):
        return "Agent 已达到步数上限，请缩小任务范围或继续下一轮。"
    return "Agent 执行失败，请稍后重试或检查配置。"


class AgentAssistant:
    """按会话动态注入记忆与 Skills 的 LangChain Agent。"""

    def __init__(
        self,
        settings: Settings,
        model: BaseChatModel,
        tools: list[BaseTool],
        memory: FileMemoryStore,
        skills: SkillRegistry,
        warnings: list[str] | None = None,
        agent_factory: Callable[..., object] = create_agent,
    ) -> None:
        self.settings = settings
        self.model = model
        self.tools = tools
        self.memory = memory
        self.skills = skills
        self.warnings = list(warnings or [])
        self._agent_factory = agent_factory
        self._tool_event_callback: Callable[[str], None] | None = None

    def set_tool_event_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """设置最终状态中的工具调用/结果事件接收器。"""
        self._tool_event_callback = callback

    async def ask(self, message: str, session_id: str) -> str:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("用户输入不能为空")
        turn_warnings: list[str] = []
        try:
            memory_context = self.memory.format_context(session_id)
        except Exception:
            memory_context = "（记忆读取失败，本轮未使用历史记忆）"
            turn_warnings.append("记忆读取失败，本轮未使用历史记忆。")
        prompt = _system_prompt(self.skills, memory_context)
        graph = self._agent_factory(
            model=self.model,
            tools=self.tools,
            system_prompt=prompt,
        )
        try:
            state = await graph.ainvoke(  # type: ignore[attr-defined]
                {"messages": [{"role": "user", "content": cleaned}]},
                config={"recursion_limit": self.settings.agent_max_steps * 2},
            )
        except Exception as exc:
            return _error_message(exc)
        answer = final_text(state) if isinstance(state, dict) else ""
        if not answer:
            return _EMPTY_RESPONSE
        if isinstance(state, dict) and self._tool_event_callback is not None:
            secret_values = (self.settings.dashscope_api_key.get_secret_value(),)
            for event in _tool_events(state, secret_values):
                try:
                    self._tool_event_callback(event)
                except Exception:
                    break
        try:
            self.memory.append_turn(session_id, cleaned, answer)
        except Exception:
            turn_warnings.append("回答已生成，但记忆写入失败。")
        if turn_warnings:
            return answer + "\n\n" + "\n".join(
                f"警告：{warning}" for warning in turn_warnings
            )
        return answer

    def clear_session(self, session_id: str) -> str:
        try:
            self.memory.clear_session(session_id)
        except Exception:
            return "短期记忆清除失败，请稍后重试。"
        return "已清除当前会话的短期记忆。"


def _memory_tools(memory: FileMemoryStore) -> list[BaseTool]:
    def remember_fact(content: str, tags: list[str] | None = None) -> str:
        record = memory.remember(content, tags)
        return f"已保存长期记忆，ID：{record.id}"

    def forget_fact(record_id: str) -> str:
        if memory.forget(record_id):
            return f"已删除长期记忆：{record_id}"
        return f"未找到长期记忆：{record_id}"

    async def aremember_fact(content: str, tags: list[str] | None = None) -> str:
        return remember_fact(content, tags)

    async def aforget_fact(record_id: str) -> str:
        return forget_fact(record_id)

    return [
        StructuredTool.from_function(
            func=remember_fact,
            coroutine=aremember_fact,
            name="remember_fact",
            description="保存跨会话稳定事实或用户明确要求记住的偏好",
            args_schema=_RememberFactInput,
        ),
        StructuredTool.from_function(
            func=forget_fact,
            coroutine=aforget_fact,
            name="forget_fact",
            description="按 ID 删除一条长期记忆",
            args_schema=_ForgetFactInput,
        ),
    ]


def _deduplicate_tools(
    groups: list[list[BaseTool]],
) -> tuple[list[BaseTool], list[str]]:
    result: list[BaseTool] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for tool in group:
            if tool.name in seen:
                warnings.append(f"工具名称重复，已保留先注册者：{tool.name}")
                continue
            seen.add(tool.name)
            result.append(tool)
    return result, warnings


async def build_assistant(
    settings: Settings,
    approve_shell: Callable[[str, str], bool],
) -> AgentAssistant:
    """创建模型、记忆、Skills、内置与 MCP 工具并组装助手。"""
    root = Path(settings.project_root).resolve()
    model = create_chat_model(settings)
    memory = FileMemoryStore(root, max_turns=settings.short_term_max_turns)
    skills = SkillRegistry(root)
    builtin_tools = build_builtin_tools(
        root=root,
        shell_timeout=settings.shell_timeout_seconds,
        max_chars=settings.shell_max_output_chars,
        approve=approve_shell,
    )

    warnings = list(skills.warnings)
    mcp_config, config_warnings = load_mcp_config_resilient(
        root / "mcp_servers.json"
    )
    warnings.extend(config_warnings)
    mcp_tools, mcp_warnings = await load_mcp_tools_resilient(mcp_config)
    warnings.extend(mcp_warnings)

    tools, duplicate_warnings = _deduplicate_tools(
        [builtin_tools, skills.as_tools(), _memory_tools(memory), mcp_tools]
    )
    warnings.extend(duplicate_warnings)
    return AgentAssistant(settings, model, tools, memory, skills, warnings)
