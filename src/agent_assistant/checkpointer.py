"""Demo agent 的内存 checkpointer。"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver


def build_memory_checkpointer() -> InMemorySaver:
    """创建进程内对话状态存储，进程退出后不保留。"""
    return InMemorySaver()
