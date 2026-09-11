from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_assistant.config import Settings
from agent_assistant.model import create_chat_model


def test_create_chat_model_maps_settings(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="secret-key",
        dashscope_base_url="https://example.com/v1",
        dashscope_model="qwen-max",
        model_temperature=0.5,
        model_timeout_seconds=45,
        model_max_retries=2,
        _env_file=None,
    )
    mock_chat = MagicMock()

    with patch(
        "agent_assistant.model.ChatOpenAI", return_value=mock_chat
    ) as mock_cls:
        result = create_chat_model(settings)

    mock_cls.assert_called_once_with(
        api_key="secret-key",
        base_url="https://example.com/v1",
        model="qwen-max",
        temperature=0.5,
        timeout=45,
        max_retries=2,
    )
    assert result is mock_chat
