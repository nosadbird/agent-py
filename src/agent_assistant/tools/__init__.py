"""内置安全工具的统一组装入口。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agent_assistant.tools.common import current_time, safe_calculate
from agent_assistant.tools.filesystem import ProjectFileTools
from agent_assistant.tools.shell import ShellRunner


class _CalculateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(..., description="仅含基础数值运算的表达式")


class _TimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timezone_name: str = Field("Asia/Shanghai", description="IANA 时区名称")


class _ShellInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(..., description="要在项目根目录执行的 Shell 命令")


def build_builtin_tools(
    root: Path,
    shell_timeout: int,
    max_chars: int,
    approve: Callable[[str, str], bool],
) -> list[BaseTool]:
    """组装项目文件、计算、时间与 Shell LangChain 工具。"""
    file_tools = ProjectFileTools(root, max_chars=max_chars)
    shell_runner = ShellRunner(root, shell_timeout, max_chars, approve)

    async def acalculate(expression: str) -> str:
        return safe_calculate(expression)

    async def acurrent_time(timezone_name: str = "Asia/Shanghai") -> str:
        return current_time(timezone_name)

    async def arun_shell(command: str) -> str:
        return shell_runner.run(command)

    common_tools: list[BaseTool] = [
        StructuredTool.from_function(
            func=safe_calculate,
            coroutine=acalculate,
            name="calculate",
            description="安全计算基础数值表达式",
            args_schema=_CalculateInput,
        ),
        StructuredTool.from_function(
            func=current_time,
            coroutine=acurrent_time,
            name="current_time",
            description="获取指定 IANA 时区的当前时间",
            args_schema=_TimeInput,
        ),
        StructuredTool.from_function(
            func=shell_runner.run,
            coroutine=arun_shell,
            name="run_shell",
            description="在项目根目录执行经过风险分类和审批的 Shell 命令",
            args_schema=_ShellInput,
        ),
    ]
    return [*file_tools.as_tools(), *common_tools]


__all__ = [
    "ProjectFileTools",
    "ShellRunner",
    "build_builtin_tools",
    "current_time",
    "safe_calculate",
]
