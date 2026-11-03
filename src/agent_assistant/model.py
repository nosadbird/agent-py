"""阿里云百炼 Chat 模型工厂。"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agent_assistant.config import Settings


def create_chat_model(
    settings: Settings,
    *,
    enable_thinking: bool = False,
) -> BaseChatModel:
    """根据配置创建阿里云百炼 OpenAI 兼容 Chat 模型实例。"""
    kwargs: dict[str, Any] = {}
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}
    return ChatOpenAI(
        api_key=settings.dashscope_api_key.get_secret_value(),
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_model,
        temperature=settings.model_temperature,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        **kwargs,
    )
