# LangChain ReAct Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个由阿里云百炼驱动、支持内置工具、MCP、Skills、TXT 长短期记忆和危险 Shell 审批的 Python 3.11 CLI ReAct Agent，并附手写 ReAct 教学示例。

**Architecture:** 主入口基于 LangChain 1.x `create_agent`，通过清晰的配置、记忆、技能、工具和 MCP 模块组合运行时能力。所有文件访问限制在项目根目录，外部依赖通过接口注入以便测试；手写 demo 使用 `bind_tools` 显式演示工具调用循环。

**Tech Stack:** Python 3.11、LangChain 1.x、LangGraph、langchain-openai、langchain-mcp-adapters、MCP/FastMCP、Pydantic Settings、Rich、PyYAML、pytest。

## Global Constraints

- Python 必须为 `>=3.11,<3.12`。
- 默认模型接口为阿里云百炼 OpenAI 兼容 API。
- 只提供 CLI，不增加 Web 或 FastAPI。
- 记忆必须保存到项目内 `.txt` 文件。
- 所有关键注释、用户提示和 README 使用中文。
- 不输出模型私有思维链，只输出工具动作与观察摘要。
- Shell 危险命令必须由用户确认，并始终设置超时与输出上限。
- MCP 单服务失败必须降级，不能阻断其他功能。

---

### Task 1: 项目骨架、配置与模型工厂

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/agent_assistant/__init__.py`
- Create: `src/agent_assistant/config.py`
- Create: `src/agent_assistant/model.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings(project_root: Path)`, `load_settings(project_root: Path) -> Settings`
- Produces: `create_chat_model(settings: Settings) -> BaseChatModel`

- [ ] **Step 1: 写配置失败测试**

```python
from pathlib import Path
import pytest
from pydantic import ValidationError
from agent_assistant.config import Settings


def test_settings_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(project_root=tmp_path, _env_file=None)


def test_settings_has_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="test-key",
        _env_file=None,
    )
    assert settings.agent_max_steps == 12
    assert settings.short_term_max_turns == 20
    assert settings.shell_timeout_seconds == 30
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL，提示 `ModuleNotFoundError: agent_assistant`。

- [ ] **Step 3: 创建项目元数据和最小配置实现**

`Settings` 使用 `pydantic_settings.BaseSettings`，字段名严格为：

```python
class Settings(BaseSettings):
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
```

`create_chat_model` 构造 `ChatOpenAI(api_key=..., base_url=..., model=..., timeout=..., max_retries=...)`，不得记录 SecretStr 的明文。

- [ ] **Step 4: 验证配置测试通过**

Run: `python -m pytest tests/test_config.py -v`

Expected: 2 passed。

---

### Task 2: TXT 短期与长期记忆

**Files:**
- Create: `src/agent_assistant/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `MemoryRecord(id, timestamp, kind, content, session_id, tags)`
- Produces: `FileMemoryStore(root: Path, max_turns: int)`
- Produces: `append_turn(session_id: str, user: str, assistant: str) -> None`
- Produces: `recent_turns(session_id: str) -> list[MemoryRecord]`
- Produces: `remember(content: str, tags: list[str]) -> MemoryRecord`
- Produces: `forget(record_id: str) -> bool`
- Produces: `long_term_facts() -> list[MemoryRecord]`

- [ ] **Step 1: 写记忆行为失败测试**

```python
from pathlib import Path
from agent_assistant.memory import FileMemoryStore


def test_short_term_is_trimmed_by_turn(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path, max_turns=2)
    for index in range(3):
        store.append_turn("s1", f"u{index}", f"a{index}")
    records = store.recent_turns("s1")
    assert [item.content for item in records] == ["u1", "a1", "u2", "a2"]


def test_corrupt_json_line_does_not_break_valid_memory(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "long_term.txt").write_text(
        '{"id":"1","timestamp":"now","kind":"fact","content":"喜欢中文","session_id":null,"tags":[]}\n'
        "broken\n",
        encoding="utf-8",
    )
    store = FileMemoryStore(tmp_path, max_turns=2)
    assert [item.content for item in store.long_term_facts()] == ["喜欢中文"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_memory.py -v`

Expected: FAIL，提示 `agent_assistant.memory` 不存在。

- [ ] **Step 3: 实现 JSONL 解析和原子写入**

实现 `_read_records(path)` 逐行 `json.loads`，仅跳过 `JSONDecodeError` 和校验失败行；实现 `_atomic_write(path, records)`，先写 `path.with_suffix(".tmp")`，再调用 `replace`。每轮写入两个 `kind="short"` 记录，裁剪时按 session 保留最后 `max_turns * 2` 条，同时保留其他 session 的有效记录。

长期记忆单条最多 4000 字符，总注入最多 12000 字符；`forget` 仅删除匹配 ID 的长期记录。

- [ ] **Step 4: 运行记忆测试**

Run: `python -m pytest tests/test_memory.py -v`

Expected: 全部通过。

---

### Task 3: Skills 动态发现与按需加载

**Files:**
- Create: `src/agent_assistant/skills.py`
- Create: `skills/example-summary/SKILL.md`
- Test: `tests/test_skills.py`

**Interfaces:**
- Produces: `SkillInfo(name: str, description: str, path: Path)`
- Produces: `SkillRegistry(root: Path, max_bytes: int = 100_000)`
- Produces: `list_skills() -> list[SkillInfo]`
- Produces: `load_skill(name: str) -> str`
- Produces: `as_tools() -> list[BaseTool]`

- [ ] **Step 1: 写 Skill 解析失败测试**

```python
from pathlib import Path
from agent_assistant.skills import SkillRegistry


def test_registry_lists_and_loads_valid_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: writer\ndescription: 写作流程\n---\n\n# 步骤\n先澄清。\n",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path)
    assert [item.name for item in registry.list_skills()] == ["writer"]
    assert "先澄清" in registry.load_skill("writer")


def test_registry_rejects_path_escape(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path)
    assert "非法" in registry.load_skill("../secret")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_skills.py -v`

Expected: FAIL，提示 `agent_assistant.skills` 不存在。

- [ ] **Step 3: 实现注册表与 LangChain 工具**

仅接受 `^[A-Za-z0-9_-]+$` 的目录和 frontmatter `name`。使用 `yaml.safe_load`；无效 YAML、缺字段、重名和超过大小限制的文件加入 warnings 并跳过。工具返回字符串而不是抛出路径或解析异常，便于模型自我修正。

示例 Skill 名为 `example-summary`，正文要求先提取主题、再列关键事实、最后给出三句内摘要。

- [ ] **Step 4: 运行 Skill 测试**

Run: `python -m pytest tests/test_skills.py -v`

Expected: 全部通过。

---

### Task 4: 安全内置工具

**Files:**
- Create: `src/agent_assistant/tools/__init__.py`
- Create: `src/agent_assistant/tools/filesystem.py`
- Create: `src/agent_assistant/tools/common.py`
- Create: `src/agent_assistant/tools/shell.py`
- Test: `tests/tools/test_filesystem.py`
- Test: `tests/tools/test_common.py`
- Test: `tests/tools/test_shell.py`

**Interfaces:**
- Produces: `ProjectFileTools(root: Path, max_chars: int)`
- Produces: `resolve_project_path(root: Path, user_path: str) -> Path`
- Produces: `safe_calculate(expression: str) -> str`
- Produces: `ShellRisk(is_risky: bool, reason: str)`
- Produces: `classify_shell_command(command: str) -> ShellRisk`
- Produces: `ShellRunner(root, timeout, max_chars, approve).run(command) -> str`
- Produces: `build_builtin_tools(...) -> list[BaseTool]`

- [ ] **Step 1: 写路径与 Shell 安全失败测试**

```python
from pathlib import Path
import pytest
from agent_assistant.tools.filesystem import resolve_project_path
from agent_assistant.tools.shell import ShellRunner, classify_shell_command


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="项目目录"):
        resolve_project_path(tmp_path, "../outside.txt")


def test_remove_command_requires_approval(tmp_path: Path) -> None:
    assert classify_shell_command("Remove-Item important.txt").is_risky
    runner = ShellRunner(tmp_path, 3, 1000, approve=lambda _c, _r: False)
    assert "未执行" in runner.run("Remove-Item important.txt")
```

计算测试必须覆盖 `1 + 2 * 3 == 7` 和拒绝 `__import__('os')`。文件测试覆盖正常读写、目录列表、正则搜索、绝对路径和符号链接逃逸。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/tools -v`

Expected: FAIL，提示工具模块不存在。

- [ ] **Step 3: 实现最小安全工具**

`resolve_project_path` 使用 `Path.resolve()` 后调用 `candidate.relative_to(root.resolve())` 验证边界。写文件自动创建项目内父目录，所有文本强制 UTF-8。

计算器遍历 AST，只允许 `Expression`、`Constant`、`UnaryOp`、`BinOp` 及 `+ - * / // % **`，限制表达式长度和幂指数。

Shell 分类同时检查 PowerShell、cmd 与常见 Unix 风险词，包括删除、覆盖、格式化磁盘、权限修改、结束进程、下载后管道执行和项目外路径；未命中明确只读白名单时按危险处理。使用 `subprocess.run(..., cwd=root, timeout=..., capture_output=True, text=True, shell=True)`，返回退出码、stdout、stderr 的截断文本。

- [ ] **Step 4: 运行工具测试**

Run: `python -m pytest tests/tools -v`

Expected: 全部通过。

---

### Task 5: MCP Client 与本地 FastMCP 示例

**Files:**
- Create: `src/agent_assistant/mcp_client.py`
- Create: `src/agent_assistant/mcp_servers/__init__.py`
- Create: `src/agent_assistant/mcp_servers/demo_server.py`
- Create: `mcp_servers.json`
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Produces: `load_mcp_config(path: Path) -> dict[str, dict[str, object]]`
- Produces: `async load_mcp_tools_resilient(config, client_factory=...) -> tuple[list[BaseTool], list[str]]`

- [ ] **Step 1: 写 MCP 降级失败测试**

```python
import pytest
from agent_assistant.mcp_client import load_mcp_tools_resilient


class FakeClient:
    def __init__(self, connections, **kwargs):
        self.connections = connections

    async def get_tools(self, server_name=None):
        if server_name == "bad":
            raise OSError("offline")
        return [f"{server_name}_tool"]


@pytest.mark.asyncio
async def test_one_failed_server_does_not_hide_healthy_tools() -> None:
    tools, warnings = await load_mcp_tools_resilient(
        {"ok": {"transport": "stdio"}, "bad": {"transport": "http"}},
        client_factory=FakeClient,
    )
    assert tools == ["ok_tool"]
    assert any("bad" in warning for warning in warnings)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_mcp_client.py -v`

Expected: FAIL，提示 MCP 模块不存在。

- [ ] **Step 3: 实现逐服务隔离加载**

校验 transport 只能是 `stdio`、`http` 或 `streamable_http`；逐个以单服务配置创建 `MultiServerMCPClient(tool_name_prefix=True, handle_tool_errors=True)` 并调用 `get_tools()`。异常仅转成中文 warning。

FastMCP 示例暴露 `add(a: float, b: float) -> float` 与 `text_stats(text: str) -> dict[str, int]`。默认 JSON 通过当前 Python 解释器模块路径 `python -m agent_assistant.mcp_servers.demo_server` 启动。

- [ ] **Step 4: 运行 MCP 单元测试**

Run: `python -m pytest tests/test_mcp_client.py -v`

Expected: 全部通过。

---

### Task 6: 主 Agent 与 CLI

**Files:**
- Create: `src/agent_assistant/agent.py`
- Create: `src/agent_assistant/cli.py`
- Test: `tests/test_agent.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `AgentAssistant(settings, model, tools, memory, skills)`
- Produces: `async ask(message: str, session_id: str) -> str`
- Produces: `async build_assistant(settings: Settings) -> AgentAssistant`
- Produces: `main() -> None`

- [ ] **Step 1: 写主循环失败测试**

```python
import pytest
from langchain_core.messages import AIMessage
from agent_assistant.agent import final_text


def test_final_text_returns_last_ai_message() -> None:
    state = {"messages": [AIMessage(content="完成")]}
    assert final_text(state) == "完成"


def test_final_text_handles_empty_response() -> None:
    assert "没有返回" in final_text({"messages": []})
```

CLI 测试使用注入的 `input_fn` 和 `output_fn` 验证 `/exit` 不调用 Agent、`/clear` 清理当前 session、一次 `KeyboardInterrupt` 只取消当前轮。

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_agent.py tests/test_cli.py -v`

Expected: FAIL，提示 Agent/CLI 模块不存在。

- [ ] **Step 3: 实现 Agent 组装和调用**

系统提示包含：角色、仅在需要时调用工具、工具失败后可修正、危险操作规则、Skill 目录、长期事实和最近会话。`create_agent` 接收模型、聚合工具和 `ToolRetryMiddleware`、`ModelRetryMiddleware`、`ToolCallLimitMiddleware`、`ModelCallLimitMiddleware`；若当前 LangChain 小版本的中间件签名不同，以已安装官方 API 为准并用兼容测试锁定行为。

`ask` 调用 `agent.ainvoke({"messages": [{"role": "user", "content": message}]}, config={"recursion_limit": settings.agent_max_steps * 2})`。仅成功获得非空最终回答后写入短期记忆。鉴权异常、限流、超时、递归超限与未知异常映射为不泄露密钥的中文消息。

- [ ] **Step 4: 实现 Rich CLI**

支持 `/help`、`/clear` 和 `/exit`；启动时显示成功加载的内置、Skill、MCP 工具数量及降级告警。工具事件只展示工具名称和截断摘要。危险 Shell 审批只接受 `y` 或 `yes`。

- [ ] **Step 5: 运行 Agent 与 CLI 测试**

Run: `python -m pytest tests/test_agent.py tests/test_cli.py -v`

Expected: 全部通过。

---

### Task 7: 手写 ReAct 示例、文档与整体验收

**Files:**
- Create: `src/agent_assistant/manual_react_demo.py`
- Create: `README.md`
- Test: `tests/test_manual_react.py`

**Interfaces:**
- Produces: `async manual_react(model, tools, user_input: str, max_steps: int = 8) -> str`
- Produces: `manual_demo_main() -> None`

- [ ] **Step 1: 写手写循环失败测试**

```python
import pytest
from langchain_core.messages import AIMessage
from agent_assistant.manual_react_demo import manual_react


class SequenceModel:
    def __init__(self):
        self.index = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        responses = [
            AIMessage(content="", tool_calls=[
                {"name": "echo", "args": {"text": "hi"}, "id": "call-1"}
            ]),
            AIMessage(content="完成"),
        ]
        result = responses[self.index]
        self.index += 1
        return result


@pytest.mark.asyncio
async def test_manual_react_executes_tool_then_answers(echo_tool) -> None:
    assert await manual_react(SequenceModel(), [echo_tool], "开始") == "完成"
```

补充未知工具与循环超限测试：

```python
class InfiniteToolModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="", tool_calls=[
            {"name": "missing", "args": {}, "id": "missing-call"}
        ])


@pytest.mark.asyncio
async def test_manual_react_stops_at_limit() -> None:
    result = await manual_react(InfiniteToolModel(), [], "开始", max_steps=2)
    assert "2 轮" in result


class EmptyModel:
    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="")


@pytest.mark.asyncio
async def test_manual_react_handles_empty_answer() -> None:
    result = await manual_react(EmptyModel(), [], "开始")
    assert "没有返回" in result
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_manual_react.py -v`

Expected: FAIL，提示教学模块不存在。

- [ ] **Step 3: 实现显式 tool-calling 循环**

建立 `{tool.name: tool}` 映射。每次 `ainvoke` 后先追加 `AIMessage`；对每个 tool call 调用 `await tool.ainvoke(args)`，捕获异常并追加带相同 `tool_call_id` 的 `ToolMessage`。没有 tool calls 时提取文本返回；达到上限返回“已达到 8 轮工具调用上限”。

- [ ] **Step 4: 编写中文 README**

README 给出 PowerShell 命令：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m agent_assistant.cli
python -m agent_assistant.manual_react_demo
python -m pytest -v
```

同时解释百炼地域地址、MCP 配置、Skill 格式、两种记忆文件、Shell 风险和常见错误。

- [ ] **Step 5: 运行完整自动化验证**

Run: `python -m pytest -v`

Expected: 全部测试通过且无 warning。

Run: `python -m compileall -q src tests`

Expected: exit code 0。

- [ ] **Step 6: 执行无密钥启动冒烟测试**

Run: `python -m agent_assistant.cli`

Expected: 中文提示缺少 `DASHSCOPE_API_KEY` 并指向 `.env.example`，不显示 Python traceback。

- [ ] **Step 7: 使用用户自己的百炼 Key 执行联网冒烟测试**

Run: `python -m agent_assistant.cli`

Input: `请读取 README.md，调用示例 MCP 的 text_stats，然后总结项目。`

Expected: 展示文件工具和带服务前缀的 MCP 工具调用，最终给出中文总结；短期记忆文件新增本轮记录。

---

## Plan Self-Review

- 设计中的百炼、ReAct、内置工具、MCP、Skills、TXT 双层记忆、Shell 审批、异常兜底、手写 demo、中文文档和 Windows 运行要求均有对应任务。
- 所有跨任务接口名称一致，测试通过构造参数替换网络、终端和审批依赖。
- 首版未引入 Web、向量数据库、多用户或分布式组件。
- 没有未定义的后续占位需求；真实百炼联网验收明确依赖用户提供有效 Key。
