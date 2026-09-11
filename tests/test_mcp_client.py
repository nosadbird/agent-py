"""MCP 配置解析与逐服务降级加载测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from agent_assistant.mcp_client import (
    load_mcp_config,
    load_mcp_config_resilient,
    load_mcp_tools_resilient,
)
from agent_assistant.mcp_servers.demo_server import add, text_stats


def _write_config(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_mcp_config_accepts_valid_stdio_and_http(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp_servers.json"
    _write_config(
        config_path,
        {
            "demo": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "agent_assistant.mcp_servers.demo_server"],
            },
            "remote": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {"Authorization": "Bearer secret-token"},
            },
            "streamable": {
                "transport": "streamable_http",
                "url": "http://localhost:9000/mcp",
            },
        },
    )

    config = load_mcp_config(config_path)

    assert config["demo"]["transport"] == "stdio"
    assert config["demo"]["command"] == "python"
    assert config["demo"]["args"] == ["-m", "agent_assistant.mcp_servers.demo_server"]
    assert config["remote"]["transport"] == "http"
    assert config["remote"]["url"] == "http://localhost:8000/mcp"
    assert config["streamable"]["transport"] == "streamable_http"


def test_load_mcp_config_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_mcp_config(tmp_path / "missing.json") == {}


def test_resilient_config_keeps_valid_service_and_redacts_invalid_one(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "mcp_servers.json"
    secret = "should-not-appear"
    _write_config(
        config_path,
        {
            "healthy": {"transport": "stdio", "command": "python"},
            "broken": {
                "transport": "http",
                "url": "",
                "headers": {"Authorization": secret},
            },
        },
    )

    config, warnings = load_mcp_config_resilient(config_path)

    assert list(config) == ["healthy"]
    assert len(warnings) == 1
    assert "broken" in warnings[0]
    assert secret not in warnings[0]


@pytest.mark.parametrize(
    ("payload", "expected_fragment"),
    [
        ("not-json", "JSON"),
        ([], "对象"),
        ({"bad name": {"transport": "stdio", "command": "python"}}, "服务名"),
        ({"demo": "stdio"}, "配置"),
        ({"demo": {"transport": "ws"}}, "transport"),
        ({"demo": {"transport": "stdio"}}, "command"),
        ({"demo": {"transport": "stdio", "command": "  "}}, "command"),
        ({"demo": {"transport": "stdio", "command": "python", "args": "bad"}}, "args"),
        ({"demo": {"transport": "http"}}, "url"),
        ({"demo": {"transport": "http", "url": ""}}, "url"),
    ],
)
def test_load_mcp_config_rejects_invalid_payload(
    tmp_path: Path,
    payload: object,
    expected_fragment: str,
) -> None:
    config_path = tmp_path / "mcp_servers.json"
    if isinstance(payload, str):
        config_path.write_text(payload, encoding="utf-8")
    else:
        _write_config(config_path, payload)

    with pytest.raises(ValueError, match=expected_fragment):
        load_mcp_config(config_path)


class FakeClient:
    created: list[dict[str, object]] = []

    def __init__(self, connections: dict[str, dict[str, object]], **kwargs: object) -> None:
        self.connections = connections
        self.kwargs = kwargs
        FakeClient.created.append({"connections": connections, "kwargs": kwargs})

    async def get_tools(self, server_name: str | None = None) -> list[BaseTool]:
        (name,) = self.connections
        if name == "bad":
            raise OSError("offline")
        return [
            StructuredTool.from_function(
                func=lambda: name,
                name=f"{name}_tool",
                description=f"{name} 工具",
            )
        ]


@pytest.fixture(autouse=True)
def _reset_fake_client() -> None:
    FakeClient.created.clear()


@pytest.mark.asyncio
async def test_one_failed_server_does_not_hide_healthy_tools() -> None:
    tools, warnings = await load_mcp_tools_resilient(
        {
            "ok": {"transport": "stdio", "command": "python", "args": []},
            "bad": {"transport": "http", "url": "http://localhost:8000/mcp"},
        },
        client_factory=FakeClient,
    )

    assert [tool.name for tool in tools] == ["ok_tool"]
    assert any("bad" in warning for warning in warnings)
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_resilient_loader_uses_expected_client_options() -> None:
    await load_mcp_tools_resilient(
        {"demo": {"transport": "stdio", "command": "python", "args": []}},
        client_factory=FakeClient,
    )

    assert len(FakeClient.created) == 1
    entry = FakeClient.created[0]
    assert entry["kwargs"]["tool_name_prefix"] is True
    assert entry["kwargs"]["handle_tool_errors"] is True
    assert list(entry["connections"].keys()) == ["demo"]


@pytest.mark.asyncio
async def test_warning_contains_service_name_but_not_secret() -> None:
    secret = "super-secret-token-value"
    _, warnings = await load_mcp_tools_resilient(
        {
            "bad": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {"Authorization": f"Bearer {secret}"},
            }
        },
        client_factory=FakeClient,
    )

    assert len(warnings) == 1
    warning = warnings[0]
    assert "bad" in warning
    assert secret not in warning


def test_demo_add_and_text_stats() -> None:
    assert add(1.5, 2.5) == 4.0
    assert text_stats("hello world\nline two") == {
        "char_count": 20,
        "non_whitespace_count": 17,
        "word_count": 4,
        "line_count": 2,
    }
