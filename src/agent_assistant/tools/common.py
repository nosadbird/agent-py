"""安全计算与时间工具。"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MAX_EXPRESSION_LENGTH = 200
_MAX_ABSOLUTE_RESULT = 10**50
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _UnsafeExpression(ValueError):
    pass


def _validate_result(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _UnsafeExpression("结果必须是实数")
    if isinstance(value, float) and not math.isfinite(value):
        raise _UnsafeExpression("结果不是有限数值")
    if abs(value) >= _MAX_ABSOLUTE_RESULT:
        raise _UnsafeExpression("计算结果过大")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _UnsafeExpression("只允许数值常量")
        return _validate_result(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _validate_result(_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise _UnsafeExpression("幂指数绝对值不能超过 100")
        return _validate_result(_BINARY_OPERATORS[type(node.op)](left, right))
    raise _UnsafeExpression("表达式包含不允许的语法")


def safe_calculate(expression: str) -> str:
    """仅计算 AST 白名单内的基础数值表达式。"""
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return f"计算错误：表达式长度不能超过 {_MAX_EXPRESSION_LENGTH}"
    if not expression.strip():
        return "计算错误：表达式不能为空"
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate(tree))
    except ZeroDivisionError:
        return "计算错误：除零"
    except (SyntaxError, _UnsafeExpression, ArithmeticError, ValueError) as exc:
        return f"计算错误：{exc}"


def current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """返回指定 IANA 时区下带 offset 的 ISO 8601 时间。"""
    try:
        return datetime.now(ZoneInfo(timezone_name)).isoformat()
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return f"时区错误：无效时区 {timezone_name}"


def get_weather(city: str) -> str:
    """模拟查询指定城市天气，返回固定文本（不访问真实天气服务）。"""
    cleaned = city.strip()
    if not cleaned:
        return "天气查询错误：城市名称不能为空"
    return (
        f"{cleaned}：晴，气温 22°C，东南风 2 级，湿度 55%。"
        "（模拟数据，非真实天气）"
    )
