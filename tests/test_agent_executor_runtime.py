from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from agent_assistant.agent_executor import _run_config, recursion_limit_for
from agent_assistant.config import Settings
from agent_assistant.middleware import guard_tool_result


def test_recursion_limit_is_twice_the_configured_steps(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="test-key",
        agent_max_steps=7,
        _env_file=None,
    )
    assert recursion_limit_for(settings) == 14


def test_run_config_carries_bound_recursion_limit() -> None:
    agent = SimpleNamespace(config={"recursion_limit": 14})
    config = _run_config(agent)
    assert config["configurable"]["thread_id"] == "1"
    assert config["recursion_limit"] == 14


def test_guard_tool_result_truncates_without_asking_for_approval() -> None:
    request = SimpleNamespace(tool_call={"id": "call-1"})

    def handler(_request):
        return "x" * 50

    result = guard_tool_result(request, handler, max_chars=10)
    assert isinstance(result, str)
    assert result.startswith("x" * 10)
    assert "已截断" in result


def test_guard_tool_result_turns_exceptions_into_tool_messages() -> None:
    request = SimpleNamespace(tool_call={"id": "call-2"})

    def handler(_request):
        raise RuntimeError("boom")

    result = guard_tool_result(request, handler, max_chars=100)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-2"
    assert "工具错误" in result.content
