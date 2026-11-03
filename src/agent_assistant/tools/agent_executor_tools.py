"""Demo agent 使用的当前时间工具。"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def update_plan(steps: list[str]) -> str:
    """公布或更新多步骤任务的简短执行计划；简单任务无需调用。"""
    cleaned = [step.strip() for step in steps if step.strip()][:8]
    if not cleaned:
        return "计划为空，请提供至少一个步骤。"
    return "任务计划：\n" + "\n".join(
        f"{index}. {step}" for index, step in enumerate(cleaned, start=1)
    )


@tool
def get_current_datetime() -> str:
    """获取当前本地日期和时间，格式为 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
