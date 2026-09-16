from pathlib import Path

import pytest

from agent_assistant.tools import build_builtin_tools
from agent_assistant.tools.common import current_time, safe_calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("-(2 + 3)", "-5"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
    ],
)
def test_safe_calculate_valid_expressions(expression: str, expected: str) -> None:
    assert safe_calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "True",
        "name",
        "[1, 2]",
        "(1).real",
    ],
)
def test_safe_calculate_rejects_non_whitelisted_ast(expression: str) -> None:
    assert "错误" in safe_calculate(expression)


def test_safe_calculate_handles_zero_division_and_bad_syntax() -> None:
    assert "除零" in safe_calculate("1 / 0")
    assert "错误" in safe_calculate("1 +")


def test_safe_calculate_rejects_long_expression_and_large_power() -> None:
    assert "长度" in safe_calculate("1" * 201)
    assert "指数" in safe_calculate("2 ** 101")
    assert "指数" in safe_calculate("2 ** -101")


def test_safe_calculate_limits_result_size() -> None:
    assert "过大" in safe_calculate("10 ** 100")


def test_current_time_returns_offset_iso8601() -> None:
    result = current_time("Asia/Shanghai")
    assert "T" in result
    assert result.endswith("+08:00")


def test_current_time_rejects_invalid_timezone() -> None:
    result = current_time("Invalid/Nowhere")
    assert "时区" in result and ("错误" in result or "无效" in result)


def test_build_builtin_tools_returns_all_tools(tmp_path: Path) -> None:
    tools = build_builtin_tools(
        tmp_path,
        shell_timeout=3,
        max_chars=1000,
        approve=lambda _command, _reason: False,
    )
    assert {tool.name for tool in tools} == {
        "read_text",
        "write_text",
        "list_directory",
        "search_text",
        "calculate",
        "current_time",
        "get_weather",
        "run_shell",
    }
