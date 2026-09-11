# Agent Assistant

一个面向 Python 3.11 的中文本地 Agent 教学项目。项目使用 LangChain/LangGraph
连接阿里云百炼 OpenAI 兼容接口，提供安全文件工具、需审批的 Shell 工具、Skills、
文件记忆和可降级的 MCP 扩展；另附一个显式 tool-calling/ReAct 循环，便于理解模型、
工具和消息之间的协作方式。

> 本项目会把模型生成的工具参数交给本地工具执行。运行前请阅读“安全说明”，不要把它
> 当作沙箱，也不要在不可信项目目录或含敏感文件的目录中运行。

## 架构与目录

```text
.
├─ src/agent_assistant/
│  ├─ agent.py                 # A：LangChain create_agent 主实现
│  ├─ manual_react_demo.py     # B：手写 tool-calling/ReAct 教学循环
│  ├─ cli.py                   # 多轮交互 CLI、/help、/clear、/exit
│  ├─ config.py / model.py     # 配置校验与百炼模型工厂
│  ├─ memory.py                # 短期/长期 JSONL 文件记忆
│  ├─ skills.py                # Skills 发现、校验和按需加载
│  ├─ mcp_client.py            # MCP 配置与逐服务降级加载
│  ├─ mcp_servers/demo_server.py
│  └─ tools/                   # 文件、计算、时间、Shell 工具
├─ skills/example-summary/SKILL.md
├─ tests/
├─ mcp_servers.json
├─ .env.example
└─ pyproject.toml
```

### A/B 两种 ReAct 实现

- **A：`agent_assistant.cli` + `agent.py`**  
  生产式主流程。由 LangChain `create_agent`/LangGraph 管理工具循环，支持多轮 CLI、
  Skills、短期和长期记忆、内置工具及 MCP 工具；适合日常使用和后续扩展。
- **B：`manual_react_demo.py`**  
  教学式单轮入口。代码显式执行 `bind_tools → AIMessage → tool_calls →
  ToolMessage → 下一轮模型调用`，只加载内置安全工具，不接入记忆、Skills 或 MCP，
  便于阅读和调试协议。

两者都不会打印或构造 `Thought`，也不展示模型私有思维链。手写版本中的“ReAct”指
可观察的工具调用与 observation 循环，不代表输出隐藏推理过程。

## Windows PowerShell 安装

要求 CPython 3.11。项目的依赖约束不支持 3.12。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

如果 PowerShell 拒绝执行激活脚本，可仅为当前进程调整策略：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 配置百炼

复制模板：

```powershell
Copy-Item .env.example .env
```

至少填写：

```dotenv
DASHSCOPE_API_KEY=你的百炼_API_Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

- 不要提交 `.env` 或真实密钥；`.gitignore` 已忽略 `.env`。
- `DASHSCOPE_MODEL` 必须是当前账户和地域已开通的模型。
- Base URL 随地域和服务入口可能不同，不能把其他地域的 Key 与地址混用。请以阿里云
  百炼控制台中当前地域的 OpenAI 兼容接口说明为准。
- `.env.example` 还提供温度、超时、重试、Agent 步数、记忆轮数和 Shell 输出限制。

## 运行

### 主 CLI

模块入口：

```powershell
python -m agent_assistant.cli
```

安装 editable package 后也可使用 console script：

```powershell
agent-assistant
```

CLI 支持：

- `/help`：显示命令；
- `/clear`：只清除当前 CLI 会话的短期记忆；
- `/exit`、`exit`、`quit`：退出。

### 手写 ReAct Demo

```powershell
python -m agent_assistant.manual_react_demo
# 或
agent-react-demo
```

它读取一次用户输入，固定使用默认最多 8 轮工具循环并退出，不读取 `AGENT_MAX_STEPS`。
未知工具、非法参数
或工具异常会作为 `ToolMessage` observation 交还模型，使模型有机会自我修正。

## 内置工具与安全说明

主 CLI 和手写 Demo 都可加载以下内置工具：

- `read_text`：读取项目内 UTF-8 文本；
- `write_text`：写入项目内 UTF-8 文本；
- `list_directory`：列出项目目录；
- `search_text`：使用 Python 正则搜索文本；
- `calculate`：计算受限的基础数值表达式；
- `current_time`：读取指定 IANA 时区时间；
- `run_shell`：在项目根目录运行 Shell 命令。

主 CLI 另外聚合 `list_skills`、`load_skill`、`remember_fact`、`forget_fact` 和 MCP
工具。若名称重复，保留先注册的工具并显示 warning。

### Shell 审批

Shell 采用默认拒绝策略：只有明确识别的只读白名单命令可自动运行；删除、覆盖、安装、
下载、提权、重定向、命令连接符以及无法确认的命令都会显示**完整命令和风险原因**。
仅输入 `y` 或 `yes` 才批准执行，其他输入、中断或 EOF 都视为拒绝。

Shell 执行环境会移除常见密钥变量并限制输出，但它**不是操作系统沙箱**：

- 已批准命令仍以当前 Windows 用户权限运行；
- 工具的文本路径检查不能约束 Shell 自身；
- 不要批准看不懂的命令，不要在管理员终端中运行；
- 先备份重要文件，并使用权限受限的测试目录。

## 添加 Skill

每个 Skill 位于 `skills/<name>/SKILL.md`。目录名和 frontmatter 的 `name` 必须一致，
名称仅使用字母、数字、下划线和连字符，文件使用 UTF-8：

```markdown
---
name: review-code
description: 检查 Python 变更并输出可执行的评审意见
---

# 代码评审

1. 阅读相关实现和测试。
2. 按严重程度列出有证据的问题。
3. 给出最小修复建议。
```

启动时只把名称和描述放入 catalog；模型需要时通过 `load_skill` 按需读取全文。运行期间
若文件被修改，指纹检查会拒绝加载；修改 Skill 后请重启 CLI。

最短扩展步骤：复制示例目录，修改目录名、`name`、`description` 和正文，重启后观察
启动计数或让 Agent 调用 `list_skills`。无需修改 Python 注册代码。

## 记忆文件

运行主 CLI 后，记忆位于：

- `memory/short_term.txt`：短期对话，每条用户/助手消息一行；
- `memory/long_term.txt`：跨会话事实，每条事实一行。

文件扩展名虽为 `.txt`，内容语义是 **JSON Lines（JSONL）**：每个非空行都是独立
JSON 对象，包含 `id`、`timestamp`、`kind`、`content`、`session_id`、`tags`。
损坏或字段不完整的行会被忽略。短期记忆按 `SHORT_TERM_MAX_TURNS` 为每个 session
保留最近轮次；长期事实由 `remember_fact` 写入，可用返回的 ID 调用 `forget_fact`。

`/clear` 只删除当前随机 session ID 对应的短期记录，不清除其他会话或长期事实。
记忆可能含用户内容，分享项目之前应自行检查并移除 `memory/`。如不希望版本控制记忆，
可在本地 `.gitignore` 中加入 `memory/`。

手写 Demo 不读写上述记忆。

## MCP

项目根目录的 `mcp_servers.json` 是服务映射。内置 stdio 示例：

```json
{
  "demo": {
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "agent_assistant.mcp_servers.demo_server"]
  }
}
```

该服务暴露 `add` 和 `text_stats`。客户端启用了服务名前缀，因此 Agent 侧工具名会带
服务标识，避免多个 MCP 服务重名。

HTTP/streamable HTTP 示例（地址和令牌仅为占位符）：

```json
{
  "remote": {
    "transport": "streamable_http",
    "url": "https://example.invalid/mcp",
    "headers": {
      "Authorization": "Bearer replace-me"
    }
  }
}
```

也可把 `transport` 写为 `http`。不要提交真实 token。配置按服务逐个加载：单个服务
连接失败只产生中文 warning，其余服务和内置工具仍可使用；但 JSON 顶层整体格式错误时，
主 CLI 会降级为不使用任何 MCP 工具。

当前加载方式适合无状态工具发现和调用。依赖 server session、会话 cookie、持续订阅或
其他状态的 MCP 服务，需要在应用生命周期中持有对应客户端/session；不能假设每次
临时加载都能延续状态。接入此类服务时应先改造 `mcp_client.py` 的生命周期管理。

最短扩展步骤：在 `mcp_servers.json` 新增唯一服务名，按 transport 填写
`command/args` 或 `url/headers`，重启 CLI，检查启动 warning 和可用工具。

## 扩展新工具

1. 在 `src/agent_assistant/tools/` 中实现普通函数或受状态约束的类。
2. 使用 Pydantic `BaseModel` 定义严格参数 schema，建议 `extra="forbid"`。
3. 用 `StructuredTool.from_function` 同时提供同步函数和 async coroutine。
4. 在 `build_builtin_tools()` 返回列表中注册唯一名称。
5. 先补工具的成功、拒绝和核心错误路径测试，再运行全量测试。

工具返回值应是可安全展示的短文本；不要在异常或 observation 中回显密钥。

## 测试与离线验收

核心手写循环：

```powershell
python -m pytest tests/test_manual_react.py -v
```

完整测试和语法编译：

```powershell
python -m pytest -q
python -m compileall -q src tests
```

测试使用假的顺序模型和本地工具，不需要 API Key，不会请求百炼。验证“缺 Key 时中文提示
且无 traceback”时，即使机器已有环境变量或 `.env`，也可临时用纯空白覆盖：

```powershell
$saved = $env:DASHSCOPE_API_KEY
$env:DASHSCOPE_API_KEY = " "
python -m agent_assistant.cli
$env:DASHSCOPE_API_KEY = $saved
```

预期输出包含 `.env.example`、`DASHSCOPE_API_KEY`，且不出现 `Traceback`。这条冒烟
测试不应进入模型调用。

## 常见问题

### 配置加载失败或缺少 Key

确认命令在项目根目录执行，`.env` 文件名正确且 Key 不是空白。不要给值额外加中文引号。

### 401/403 鉴权失败

检查 Key 是否有效、Base URL 地域是否匹配、模型是否开通以及账户权限。不要把完整 Key
贴入 issue、日志或聊天。

### 模型不存在或无权限

把 `DASHSCOPE_MODEL` 改为百炼控制台当前地域已开通的模型名。模型展示名不一定等于 API
调用名。

### MCP 服务加载失败

- stdio：先在同一虚拟环境运行模块，确认 `python` 指向项目 Python；
- HTTP：检查 URL、transport、代理、证书和认证 header；
- JSON：检查双引号、逗号和顶层对象结构；
- 有状态服务：确认是否需要持久 MCP session，而不是重复创建短连接。

MCP warning 不代表主 Agent 整体不可用；先确认内置工具是否正常。

### Windows 命令或编码问题

- 使用 PowerShell 示例，不要把 Bash 的 `source` 当作激活命令；
- 若 console script 找不到，重新激活 `.venv` 并执行 `pip install -e ".[dev]"`；
- `.py`、`SKILL.md` 和被文件工具读取的文本应保存为 UTF-8；
- 路径参数使用项目内相对路径；绝对路径和 `..` 越界会被拒绝；
- 执行策略阻止激活时使用本页的 `Set-ExecutionPolicy -Scope Process Bypass`。

## 当前边界

- 真实百炼连通性、模型权限和不同模型的 tool-calling 兼容性取决于外部服务；
- Shell 风险分类与审批减少误操作，但不提供强隔离；
- 文件记忆适合单机教学，不提供多进程事务、加密或用户级权限隔离；
- 手写 Demo 是单轮教学入口，完整交互能力请使用主 CLI。
