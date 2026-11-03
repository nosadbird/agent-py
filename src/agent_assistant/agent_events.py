"""agent_executor 链路中可展示的运行事件。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    """一条由真实运行状态产生的用户可见事件。"""

    kind: str
    text: str


EventHandler = Callable[[AgentEvent], None]


def emit_event(handler: EventHandler | None, kind: str, text: str) -> None:
    """发送事件；展示层异常不能中断 Agent。"""
    if handler is None:
        return
    try:
        handler(AgentEvent(kind=kind, text=text))
    except Exception:
        pass
