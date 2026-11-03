"""agent_executor 专用的 Shell：只读命令直接执行，危险命令才审批。"""

from __future__ import annotations

import asyncio
import codecs
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agent_assistant.agent_events import EventHandler, emit_event
from agent_assistant.tools.agent_executor_files import truncate_text

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


class _ShellInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(..., description="要在项目根目录执行的 Shell 命令")


@dataclass(frozen=True)
class ShellRisk:
    is_risky: bool
    reason: str


def console_approve(command: str, reason: str) -> bool:
    """在终端展示完整命令和原因，仅 y/yes 视为同意。"""
    print(f"检测到危险 Shell 命令：\n{command}")
    print(f"风险原因：{reason}")
    try:
        answer = input("是否执行？仅输入 y/yes 表示同意：")
    except (KeyboardInterrupt, EOFError):
        print("审批已取消，命令未执行。")
        return False
    return answer.strip().lower() in {"y", "yes"}


def _safe_read_only_command(command: str) -> bool:
    normalized = " ".join(command.strip().split())
    simple_patterns = (
        r"(?:pwd|get-location)(?:\s+[-\w]+)*",
        r"(?:dir|ls|get-childitem)(?:\s+.+)?",
        r"(?:type|cat|get-content)\s+.+",
        r"python\s+--version",
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


@dataclass(frozen=True)
class ShellBackend:
    """一个已解析的本地 Shell 后端。"""

    name: str
    executable: str

    def command(self, text: str) -> list[str]:
        if self.name == "powershell":
            return [self.executable, "-NoProfile", "-NonInteractive", "-Command", text]
        if self.name == "bash":
            return [self.executable, "--noprofile", "--norc", "-c", text]
        return [self.executable, "-c", text]


def resolve_shell_backend(requested: str = "auto", bash_path: str | None = None) -> ShellBackend:
    """解析 Shell；用户明确指定时找不到就报错。"""
    name = requested.lower().strip()
    if name not in {"auto", "powershell", "bash", "sh"}:
        raise ValueError(f"不支持的 Shell 后端：{requested}")
    if name == "auto":
        if sys.platform == "win32":
            return ShellBackend("powershell", "powershell.exe")
        bash = shutil.which("bash")
        return ShellBackend("bash", bash) if bash else ShellBackend("sh", shutil.which("sh") or "/bin/sh")
    if name == "powershell":
        executable = shutil.which("powershell.exe") or shutil.which("pwsh")
    elif name == "bash":
        executable = bash_path or shutil.which("bash")
    else:
        executable = shutil.which("sh")
    if not executable or (bash_path and name == "bash" and not Path(executable).is_file()):
        raise RuntimeError(f"未找到 Shell：{name}")
    return ShellBackend(name, executable)


class ExecutorShellRunner:
    """在固定项目目录中执行经分类的 Shell 命令，仅危险命令走审批。"""

    def __init__(
        self,
        root: Path,
        timeout: int,
        max_chars: int,
        approve: Callable[[str, str], bool],
        shell_name: str = "auto",
        bash_path: str | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须为正整数")
        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数")
        self._root = root.resolve()
        self._timeout = timeout
        self._max_chars = max_chars
        self._approve = approve
        self._backend = resolve_shell_backend(shell_name, bash_path)
        self._on_event = on_event

    def _prepare(self, command: str) -> tuple[ShellRisk, str] | str:
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
        execution = _harden_automatic_git_command(command) if not risk.is_risky else command
        return risk, execution

    def run(self, command: str) -> str:
        prepared = self._prepare(command)
        if isinstance(prepared, str):
            return prepared
        _risk, execution_command = prepared
        try:
            completed = subprocess.run(
                self._backend.command(execution_command),
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
        result = truncate_text(_redact_environment_secrets(output), self._max_chars)
        emit_event(self._on_event, "tool_completed", f"退出码：{completed.returncode}")
        return result

    async def arun(self, command: str) -> str:
        """异步执行并按行发送终端输出事件。"""
        prepared = self._prepare(command)
        if isinstance(prepared, str):
            return prepared
        _risk, execution_command = prepared
        emit_event(self._on_event, "tool_started", f"运行：{command}")
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        process = await asyncio.create_subprocess_exec(
            *self._backend.command(execution_command),
            cwd=self._root,
            env=_sanitized_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creation_flags,
            start_new_session=sys.platform != "win32",
        )
        chunks: list[str] = []
        captured_chars = 0
        output_limited = False
        started = time.monotonic()

        async def read_stream(stream: asyncio.StreamReader | None, name: str) -> None:
            nonlocal captured_chars, output_limited
            if stream is None:
                return
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while data := await stream.read(4096):
                text = _redact_environment_secrets(decoder.decode(data))
                remaining = self._max_chars - captured_chars
                if remaining <= 0:
                    output_limited = True
                    continue
                visible = text[:remaining]
                captured_chars += len(visible)
                output_limited = output_limited or len(visible) < len(text)
                chunks.append(f"{name}：{visible}")
                emit_event(self._on_event, "terminal_output", f"{name}：{visible.rstrip()}")
            tail = _redact_environment_secrets(decoder.decode(b"", final=True))
            if tail and captured_chars < self._max_chars:
                visible = tail[: self._max_chars - captured_chars]
                chunks.append(f"{name}：{visible}")
                emit_event(self._on_event, "terminal_output", f"{name}：{visible.rstrip()}")

        stdout_task = asyncio.create_task(read_stream(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(read_stream(process.stderr, "stderr"))

        async def wait_and_drain() -> None:
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task)

        try:
            await asyncio.wait_for(wait_and_drain(), timeout=self._timeout)
        except asyncio.TimeoutError:
            await self._terminate(process)
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            return f"Shell 执行超时（{self._timeout} 秒）"
        except asyncio.CancelledError:
            await self._terminate(process)
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise

        elapsed = time.monotonic() - started
        emit_event(self._on_event, "tool_completed", f"退出码：{process.returncode}，耗时：{elapsed:.2f} 秒")
        output = f"退出码：{process.returncode}\n" + "".join(chunks)
        if output_limited:
            output += "\n……（已截断）"
        return truncate_text(output, self._max_chars)

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if sys.platform == "win32":
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=3,
                )
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            return

        process_group = process.pid
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            pass
        # 父进程可能先退出，仍需清理同组中忽略 SIGTERM 的子进程。
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.returncode is None:
            await process.wait()

    def as_tools(self) -> list[BaseTool]:
        """组装为可同步或异步调用的 LangChain 工具。"""
        return [
            StructuredTool.from_function(
                func=self.run,
                coroutine=self.arun,
                name="run_shell",
                description=(
                    "在项目根目录执行 Shell 命令。只读白名单命令直接执行，"
                    "其余命令需要用户审批后才会运行。"
                ),
                args_schema=_ShellInput,
            )
        ]
