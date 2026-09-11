"""采用默认拒绝策略的 Shell 风险分类与执行。"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_TRUNCATED = "\n……（已截断）"
_SENSITIVE_ENV_PARTS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "API_KEY")
_CONNECTOR_PATTERN = re.compile(r"(?:&&|\|\||[;&|]|\r|\n)")
_COMMAND_SUBSTITUTION_PATTERN = re.compile(r"(?:\$\(|`)")
_REDIRECTION_PATTERN = re.compile(r"(?:>{1,2}|<{1,2})")
_DANGEROUS_GIT_OPTION_PATTERN = re.compile(
    r"(?:^|\s)(?:-c(?:\s|$|[^\s=]+=)|--config-env(?:=|\s)|--exec-path(?:=|\s|$)|"
    r"--paginate(?:\s|$))",
    re.IGNORECASE,
)
_GIT_COMMAND_PATTERN = re.compile(
    r"^\s*git\s+(status|log|diff|show|branch)\b(.*)$",
    re.IGNORECASE,
)
_DANGEROUS_GIT_ENV = {
    "GIT_EXTERNAL_DIFF",
    "GIT_DIFF_OPTS",
    "GIT_PAGER",
    "GIT_EDITOR",
    "GIT_SEQUENCE_EDITOR",
    "GIT_ASKPASS",
    "GIT_EXEC_PATH",
}
_DANGEROUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:remove-item|rm|del|erase|rmdir|rd|unlink|shred)\b",
            re.IGNORECASE,
        ),
        "命令包含删除操作",
    ),
    (
        re.compile(
            r"\b(?:move-item|move|mv|copy-item|copy|cp|rename-item|ren)\b",
            re.IGNORECASE,
        ),
        "命令可能移动、覆盖或复制文件",
    ),
    (
        re.compile(
            r"\b(?:chmod|chown|icacls|takeown|set-acl|set-itemproperty|reg(?:\.exe)?)\b",
            re.IGNORECASE,
        ),
        "命令可能修改权限或注册表",
    ),
    (
        re.compile(
            r"\b(?:start-service|stop-service|restart-service|sc(?:\.exe)?|"
            r"start-process|stop-process|taskkill|kill|diskpart|format)\b",
            re.IGNORECASE,
        ),
        "命令可能修改服务、进程或磁盘",
    ),
    (
        re.compile(
            r"\b(?:pip|pip3|npm|yarn|pnpm|winget|choco|apt|apt-get|yum|dnf)\s+"
            r"(?:install|uninstall|remove|update|upgrade)\b",
            re.IGNORECASE,
        ),
        "命令包含包安装或修改",
    ),
    (
        re.compile(
            r"\b(?:invoke-webrequest|invoke-restmethod|curl|wget|bitsadmin)\b",
            re.IGNORECASE,
        ),
        "命令包含网络访问或下载",
    ),
    (
        re.compile(r"\b(?:invoke-expression|iex|eval)\b", re.IGNORECASE),
        "命令包含动态代码执行",
    ),
    (
        re.compile(r"\b(?:sudo|runas)\b", re.IGNORECASE),
        "命令包含提权执行",
    ),
)


@dataclass(frozen=True)
class ShellRisk:
    is_risky: bool
    reason: str


def _safe_read_only_command(command: str) -> bool:
    normalized = " ".join(command.strip().split())
    simple_patterns = (
        r"(?:pwd|get-location)(?:\s+[-\w]+)*",
        r"(?:dir|ls|get-childitem)(?:\s+.+)?",
        r"(?:type|cat|get-content)\s+.+",
        r"python\s+--version",
        r"python\s+-m\s+pytest(?:\s+.*)?",
        r"git\s+(?:status|log|diff|show)(?:\s+.*)?",
    )
    if any(re.fullmatch(pattern, normalized, re.IGNORECASE) for pattern in simple_patterns):
        lowered = normalized.lower()
        unsafe_git_options = (
            "--output",
            "--ext-diff",
            "--textconv",
            "--exec",
        )
        return not any(option in lowered for option in unsafe_git_options)

    branch_match = re.fullmatch(r"git\s+branch(?:\s+(.*))?", normalized, re.IGNORECASE)
    if branch_match is None:
        return False
    arguments = branch_match.group(1)
    if not arguments:
        return True
    # 仅允许明确的查询参数；裸分支名会创建分支，因此不能自动执行。
    return (
        re.fullmatch(r"-(?:a|r|v|vv|av|rv|ar|vr)", arguments, re.IGNORECASE)
        is not None
        or re.fullmatch(
            r"--(?:all|remotes|verbose|show-current)", arguments, re.IGNORECASE
        )
        is not None
        or re.fullmatch(
            r"--(?:list|contains|no-contains|merged|no-merged)(?:\s+.+)?",
            arguments,
            re.IGNORECASE,
        )
        is not None
    )


def classify_shell_command(command: str) -> ShellRisk:
    """判定命令是否可无需审批执行；无法确认时一律标记危险。"""
    if not command.strip():
        return ShellRisk(True, "命令为空")
    if _CONNECTOR_PATTERN.search(command):
        return ShellRisk(True, "命令包含连接符或多个命令，无法确认只读")
    if _COMMAND_SUBSTITUTION_PATTERN.search(command):
        return ShellRisk(True, "命令包含命令替换语法，可能执行额外操作")
    if _REDIRECTION_PATTERN.search(command):
        return ShellRisk(True, "命令包含输入输出重定向，可能写入文件")
    if command.lstrip().lower().startswith("git ") and _DANGEROUS_GIT_OPTION_PATTERN.search(
        command
    ):
        return ShellRisk(True, "Git 命令包含可改变配置或执行路径的全局选项")
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return ShellRisk(True, reason)
    if _safe_read_only_command(command):
        return ShellRisk(False, "命令属于只读白名单")
    return ShellRisk(True, "命令不在只读白名单中")


def _sanitized_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in _SENSITIVE_ENV_PARTS)
        and key.upper() not in _DANGEROUS_GIT_ENV
        and not key.upper().startswith("GIT_CONFIG_")
    }
    # 空通用 pager 配合 Git --no-pager，避免配置启动交互式外部程序。
    environment["PAGER"] = ""
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _harden_automatic_git_command(command: str) -> str:
    match = _GIT_COMMAND_PATTERN.fullmatch(command)
    if match is None:
        return command
    subcommand = match.group(1).lower()
    arguments = match.group(2)
    if subcommand in {"diff", "log", "show"}:
        return (
            f"git --no-pager {subcommand} --no-ext-diff --no-textconv"
            f"{arguments}"
        )
    return f"git --no-pager {subcommand}{arguments}"


def _redact_environment_secrets(text: str) -> str:
    result = text
    for key, value in os.environ.items():
        if value and any(part in key.upper() for part in _SENSITIVE_ENV_PARTS):
            result = result.replace(value, "[已隐藏敏感信息]")
    return result


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATED


def _shell_command(command: str) -> list[str]:
    if sys.platform == "win32":
        return [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
    return ["/bin/sh", "-c", command]


class ShellRunner:
    """在固定项目目录中执行经分类和审批的 Shell 命令。"""

    def __init__(
        self,
        root: Path,
        timeout: int,
        max_chars: int,
        approve: Callable[[str, str], bool],
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须为正整数")
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数")
        self._root = root.resolve()
        self._timeout = timeout
        self._max_chars = max_chars
        self._approve = approve

    def run(self, command: str) -> str:
        if not command.strip():
            return "Shell 错误：空命令，未执行"
        risk = classify_shell_command(command)
        if risk.is_risky:
            try:
                approved = self._approve(command, risk.reason)
            except Exception as exc:
                return f"Shell 审批错误：{exc}，未执行"
            if not approved:
                return f"危险命令未执行：{risk.reason}"
        try:
            execution_command = (
                _harden_automatic_git_command(command) if not risk.is_risky else command
            )
            completed = subprocess.run(
                _shell_command(execution_command),
                cwd=self._root,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=_sanitized_environment(),
            )
        except subprocess.TimeoutExpired:
            return f"Shell 执行超时（{self._timeout} 秒）"
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Shell 启动错误：{exc}"

        output = (
            f"退出码：{completed.returncode}\n"
            f"stdout：\n{completed.stdout or ''}\n"
            f"stderr：\n{completed.stderr or ''}"
        )
        return _truncate(_redact_environment_secrets(output), self._max_chars)
