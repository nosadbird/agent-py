"""MCP 配置解析与逐服务降级加载。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_VALID_TRANSPORTS = frozenset({"stdio", "http", "streamable_http"})
_SECRET_FIELD_NAMES = frozenset(
    {
        "authorization",
        "token",
        "api_key",
        "api-key",
        "x-api-key",
        "secret",
        "password",
        "cookie",
    }
)


def load_mcp_config(path: Path) -> dict[str, dict[str, object]]:
    """读取并校验 MCP 服务配置；文件不存在时返回空字典。"""
    if not path.is_file():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP 配置文件 JSON 解析失败") from exc

    if not isinstance(data, dict):
        raise ValueError("MCP 配置文件顶层必须是 JSON 对象")

    config: dict[str, dict[str, object]] = {}
    for server_name, server_config in data.items():
        if not isinstance(server_name, str) or not _SERVER_NAME_PATTERN.fullmatch(server_name):
            raise ValueError(f"MCP 服务名非法：{server_name!r}")
        if not isinstance(server_config, dict):
            raise ValueError(f"MCP 服务「{server_name}」的配置必须是对象")
        config[server_name] = _normalize_server_config(server_name, server_config)
    return config


def load_mcp_config_resilient(
    path: Path,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """逐服务校验配置；无效服务只产生脱敏 warning。"""
    if not path.is_file():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, ["MCP 配置文件读取或 JSON 解析失败，已跳过全部 MCP 服务。"]
    if not isinstance(data, dict):
        return {}, ["MCP 配置文件顶层不是对象，已跳过全部 MCP 服务。"]

    config: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for server_name, server_config in data.items():
        if not isinstance(server_name, str) or not _SERVER_NAME_PATTERN.fullmatch(
            server_name
        ):
            warnings.append("MCP 服务名非法，已跳过该服务。")
            continue
        if not isinstance(server_config, dict):
            warnings.append(f"MCP 服务「{server_name}」配置无效，已跳过。")
            continue
        try:
            config[server_name] = _normalize_server_config(server_name, server_config)
        except (TypeError, ValueError):
            warnings.append(f"MCP 服务「{server_name}」配置无效，已跳过。")
    return config, warnings


def _normalize_server_config(
    server_name: str,
    server_config: dict[str, object],
) -> dict[str, object]:
    transport = server_config.get("transport")
    if not isinstance(transport, str) or transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"MCP 服务「{server_name}」的 transport 必须是 stdio、http 或 streamable_http"
        )

    normalized: dict[str, object] = {"transport": transport}
    if transport == "stdio":
        command = server_config.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"MCP 服务「{server_name}」的 stdio command 不能为空")
        normalized["command"] = command.strip()

        args = server_config.get("args", [])
        if args is None:
            args = []
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError(f"MCP 服务「{server_name}」的 args 必须是字符串数组")
        normalized["args"] = args
    else:
        url = server_config.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"MCP 服务「{server_name}」的 HTTP url 不能为空")
        normalized["url"] = url.strip()

    for key, value in server_config.items():
        if key in {"transport", "command", "args", "url"}:
            continue
        normalized[key] = value
    return normalized


def _collect_secret_values(server_config: dict[str, object]) -> list[str]:
    secrets: list[str] = []
    for key, value in server_config.items():
        key_lower = key.lower()
        if key_lower in _SECRET_FIELD_NAMES and isinstance(value, str) and value:
            secrets.append(value)
        if key_lower == "headers" and isinstance(value, dict):
            for header_name, header_value in value.items():
                if not isinstance(header_value, str) or not header_value:
                    continue
                if header_name.lower() in _SECRET_FIELD_NAMES or header_name.lower().startswith(
                    "x-"
                ):
                    secrets.append(header_value)
    return secrets


def _sanitize_warning_message(message: str, secrets: list[str]) -> str:
    sanitized = message
    for secret in sorted(secrets, key=len, reverse=True):
        sanitized = sanitized.replace(secret, "***")
    return sanitized


def _format_service_warning(
    server_name: str,
    error: Exception,
    server_config: dict[str, object],
) -> str:
    sanitized = _sanitize_warning_message(str(error), _collect_secret_values(server_config))
    return f"MCP 服务「{server_name}」加载失败：{sanitized}"


async def load_mcp_tools_resilient(
    config: dict[str, dict[str, object]],
    client_factory: Callable[..., object] = MultiServerMCPClient,
) -> tuple[list[BaseTool], list[str]]:
    """逐个服务加载 MCP 工具；单服务失败时记录中文 warning 并继续。"""
    tools: list[BaseTool] = []
    warnings: list[str] = []

    for server_name, server_config in config.items():
        try:
            client = client_factory(
                connections={server_name: server_config},
                tool_name_prefix=True,
                handle_tool_errors=True,
            )
            server_tools = await client.get_tools()
            tools.extend(server_tools)
        except Exception as exc:
            warnings.append(_format_service_warning(server_name, exc, server_config))

    return tools, warnings
