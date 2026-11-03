"""保留百炼深度思考协议所需的 reasoning_content。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agent_assistant.config import Settings

# 百炼把思考放在这个字段，正式回答仍在 content。
_REASONING_KEY = "reasoning_content"


class _ThinkingModel(ChatOpenAI):
    """普通 ChatOpenAI 会丢掉 reasoning_content，这个子类只负责把它留下来。"""

    def _create_chat_result(self, response: dict | Any, generation_info: dict | None = None):
        # 非流式：整段思考在 message.reasoning_content。
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        choices = payload.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            message = choice.get("message") or {}
            _save_reasoning(generation.message, message.get(_REASONING_KEY))
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        # 流式：思考按增量出现在 delta.reasoning_content，合并时会自动拼接。
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return None
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            _save_reasoning(generation.message, delta.get(_REASONING_KEY))
        return generation

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        # 工具调用的下一轮必须把上一条思考原样带回，否则百炼会拒绝继续。
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        raw_messages = payload.get("messages")
        lc_messages = self._convert_input(input_).to_messages()
        if not isinstance(raw_messages, list) or len(raw_messages) != len(lc_messages):
            return payload
        for lc_message, raw_message in zip(lc_messages, raw_messages):
            if not isinstance(raw_message, dict) or not isinstance(lc_message, AIMessage):
                continue
            reasoning = lc_message.additional_kwargs.get(_REASONING_KEY)
            if isinstance(reasoning, str) and reasoning:
                raw_message[_REASONING_KEY] = reasoning
        return payload


def _save_reasoning(message, reasoning: object) -> None:
    if isinstance(reasoning, str) and reasoning:
        message.additional_kwargs[_REASONING_KEY] = reasoning


def reasoning_text(message: object) -> str:
    """提取一条模型消息中的百炼思考文本。"""
    if not isinstance(message, AIMessage):
        return ""
    reasoning = (message.additional_kwargs or {}).get(_REASONING_KEY)
    return reasoning if isinstance(reasoning, str) else ""


def reasoning_from_state(state: dict[str, Any]) -> str:
    """收集最近一轮用户消息之后的思考文本。"""
    messages = state.get("messages") or []
    start = 0
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            start = index
    parts = [reasoning_text(message).strip() for message in messages[start:]]
    return "\n\n".join(part for part in parts if part)


def create_thinking_model(settings: Settings) -> _ThinkingModel:
    """创建保留并回传 reasoning_content 的百炼模型。"""
    return _ThinkingModel(
        api_key=settings.dashscope_api_key.get_secret_value(),
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_model,
        temperature=settings.model_temperature,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        extra_body={"enable_thinking": True},
    )
