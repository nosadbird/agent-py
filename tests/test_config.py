from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_assistant.config import Settings, load_settings


def test_settings_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, _env_file=None)


@pytest.mark.parametrize("api_key", ["", "   ", "\t\n"])
def test_settings_rejects_blank_api_key(tmp_path: Path, api_key: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            project_root=tmp_path,
            dashscope_api_key=api_key,
            _env_file=None,
        )


def test_settings_has_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="test-key",
        _env_file=None,
    )
    assert settings.agent_max_steps == 12
    assert settings.short_term_max_turns == 20
    assert settings.shell_timeout_seconds == 30


def test_load_settings_reads_project_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=env-file-key\n"
        "DASHSCOPE_MODEL=custom-model\n",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.dashscope_api_key.get_secret_value() == "env-file-key"
    assert settings.dashscope_model == "custom-model"
