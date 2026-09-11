"""应用配置加载与校验。"""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量与 .env 文件加载的运行配置。"""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path
    dashscope_api_key: SecretStr
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"
    model_temperature: float = 0.2
    model_timeout_seconds: int = 60
    model_max_retries: int = 3
    agent_max_steps: int = 12
    short_term_max_turns: int = 20
    shell_timeout_seconds: int = 30
    shell_max_output_chars: int = 12_000

    @field_validator("dashscope_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API Key 不能为空或纯空白")
        return value

    @field_validator(
        "model_timeout_seconds",
        "agent_max_steps",
        "short_term_max_turns",
        "shell_timeout_seconds",
        "shell_max_output_chars",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("必须为正整数")
        return value

    @field_validator("model_max_retries")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("不能为负数")
        return value

    @field_validator("model_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0 or value > 2:
            raise ValueError("温度须在 0 到 2 之间")
        return value


def load_settings(project_root: Path) -> Settings:
    """从项目根目录加载 .env 并返回规范化后的配置。"""
    root = project_root.resolve()
    return Settings(project_root=root, _env_file=root / ".env")
