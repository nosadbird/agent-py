"""Demo agent 使用的当前时间工具。"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """获取当前本地日期和时间，格式为 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
