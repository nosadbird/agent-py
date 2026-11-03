import asyncio
import subprocess
from pathlib import Path

import pytest

from agent_assistant.tools import agent_executor_shell as shell_module
from agent_assistant.tools.agent_executor_shell import (
    ExecutorShellRunner,
    classify_shell_command,
    console_approve,
    resolve_shell_backend,
)


def test_read_only_command_is_safe() -> None:
    risk = classify_shell_command("python --version")
    assert risk.is_risky is False
    assert classify_shell_command("python -m pytest -q").is_risky is True


def test_bash_backend_uses_no_startup_files(tmp_path: Path) -> None:
    bash = tmp_path / "bash.exe"
    bash.write_text("", encoding="utf-8")
    backend = resolve_shell_backend("bash", str(bash))
    assert backend.command("pwd") == [
        str(bash),
        "--noprofile",
        "--norc",
        "-c",
        "pwd",
    ]


def test_remove_command_requires_approval(tmp_path: Path) -> None:
    assert classify_shell_command("Remove-Item important.txt").is_risky
    runner = ExecutorShellRunner(tmp_path, 3, 1000, approve=lambda _c, _r: False)
    assert "未执行" in runner.run("Remove-Item important.txt")


def test_safe_command_does_not_request_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approvals: list[tuple[str, str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "ok", ""),
    )
    runner = ExecutorShellRunner(
        tmp_path,
        3,
        1000,
        approve=lambda command, reason: approvals.append((command, reason)) or False,
    )
    assert "ok" in runner.run("python --version")
    assert approvals == []


def test_rejected_command_never_starts_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess 不应启动")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    runner = ExecutorShellRunner(tmp_path, 3, 1000, approve=lambda _c, _r: False)
    assert "未执行" in runner.run("Remove-Item important.txt")


def test_shell_output_is_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "x" * 100, ""),
    )
    result = ExecutorShellRunner(tmp_path, 1, 20, lambda _c, _r: True).run("unknown")
    assert "已截断" in result
    assert len(result) < 100


def test_console_approve_accepts_only_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert console_approve("rm x", "删除") is True
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert console_approve("rm x", "删除") is False


def test_windows_uses_noninteractive_powershell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", fake_run)
    ExecutorShellRunner(tmp_path, 3, 1000, lambda _c, _r: True).run("Get-Location")
    assert captured["command"] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-Location",
    ]


@pytest.mark.asyncio
async def test_async_shell_streams_output(tmp_path: Path) -> None:
    events: list[str] = []
    runner = ExecutorShellRunner(
        tmp_path,
        5,
        1000,
        approve=lambda _c, _r: True,
        shell_name="powershell",
        on_event=lambda event: events.append(event.text),
    )

    result = await runner.arun("Write-Output hello")

    assert "退出码：0" in result
    assert any("hello" in event for event in events)


@pytest.mark.asyncio
async def test_async_shell_truncates_long_line(tmp_path: Path) -> None:
    runner = ExecutorShellRunner(
        tmp_path,
        5,
        100,
        approve=lambda _c, _r: True,
        shell_name="powershell",
    )

    result = await runner.arun("[Console]::Write('x' * 70000)")

    assert "退出码：0" in result
    assert "已截断" in result
    assert len(result) < 1000
