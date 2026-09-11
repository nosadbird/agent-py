# Python LangChain ReAct Agent 设计说明

## 1. 目标

从空项目实现一个基于 Python 3.11 与 LangChain 1.x 的命令行 Agent 助手。主实现采用 LangChain `create_agent` 提供的标准 ReAct 工具调用循环，并额外提供一个手写工具调用循环示例，帮助理解“模型决策 → 执行工具 → 返回观察结果 → 再次决策”的工作机制。

首版必须支持：

- 阿里云百炼 OpenAI 兼容接口与支持 tool calling 的 Qwen 模型；
- 自主选择并调用内置工具和 MCP 工具；
- 从项目目录动态发现并按需加载 Skills；
- 基于项目内 TXT 文件的短期、长期记忆；
- 项目文件读写、搜索、计算、时间与 Shell 命令工具；
- 危险 Shell 命令人工确认；
- 模型、工具、MCP、配置和记忆文件异常的可恢复兜底；
- Windows PowerShell 下可直接安装和运行；
- 中文注释、中文用户提示和较详细的 README。

## 2. 范围与非目标

首版只提供交互式 CLI，不提供 Web 页面、FastAPI、多用户、账号权限系统或分布式任务调度。记忆文件只服务于本地单用户场景，不实现向量数据库和语义检索。MCP Client 支持 stdio 与 HTTP 配置，并附带一个本地 stdio FastMCP Server 示例；不实现 MCP 网关或服务市场。

Agent 会显示工具名称、参数摘要、执行状态和结果摘要，但不要求模型暴露私有思维链。ReAct 在此指可观测的“Action/Observation”工具循环，而不是打印隐藏推理内容。

## 3. 总体架构

主程序使用 `langchain.agents.create_agent` 创建 Agent。百炼通过 `langchain-openai` 的 `ChatOpenAI` 接入，API Key、Base URL、模型名、温度、超时和重试参数全部来自环境变量。LangChain Agent 负责标准 ReAct 循环；项目代码负责工具聚合、上下文组装、记忆持久化、安全审批、生命周期和错误展示。

运行时工具由三个来源组成：

1. 内置工具：项目文件、文本搜索、计算、时间和 Shell；
2. Skills 工具：列出技能与按名称加载 `SKILL.md`；
3. MCP 工具：通过 `langchain-mcp-adapters` 从配置的多个 MCP Server 动态加载。

另设独立教学入口，直接调用同一个 ChatModel 的 `bind_tools`，显式处理 `AIMessage.tool_calls`、执行工具、添加 `ToolMessage` 并再次请求模型，最多循环 8 次。该示例复用配置与基础工具，但不复制完整主程序的中间件能力。

## 4. 模块边界

建议使用 `src/agent_assistant` 包，职责如下：

- `config.py`：读取并校验环境变量，提供类型化配置；
- `model.py`：创建百炼 ChatModel，集中处理连接参数；
- `agent.py`：聚合工具、组装系统提示、创建并调用主 Agent；
- `cli.py`：交互循环、日志展示、退出与中断恢复；
- `memory.py`：TXT/JSONL 记忆解析、裁剪、追加与原子写入；
- `skills.py`：扫描、校验、列出并加载 `skills/*/SKILL.md`；
- `mcp_client.py`：解析 MCP 配置、连接多个 Server、降级加载工具；
- `tools/filesystem.py`：限制在项目根目录内的读写、列目录与搜索；
- `tools/shell.py`：风险识别、用户审批、超时执行和输出截断；
- `tools/common.py`：安全计算与当前时间；
- `manual_react_demo.py`：手写 ReAct/tool-calling 教学循环；
- `mcp_servers/demo_server.py`：本地 FastMCP 示例服务。

所有外部依赖均通过构造参数注入到核心类，Shell 审批函数、ChatModel 与 MCP 加载器可以在测试中替换，避免依赖真实网络和交互终端。

## 5. 配置

`.env.example` 提供以下配置：

- `DASHSCOPE_API_KEY`：百炼 API Key；
- `DASHSCOPE_BASE_URL`：百炼所在地域的 OpenAI 兼容地址；
- `DASHSCOPE_MODEL`：支持工具调用的 Qwen 模型名称；
- `MODEL_TEMPERATURE`：默认 `0.2`；
- `MODEL_TIMEOUT_SECONDS`：默认 `60`；
- `MODEL_MAX_RETRIES`：默认 `3`；
- `AGENT_MAX_STEPS`：默认 `12`；
- `SHORT_TERM_MAX_TURNS`：默认 `20`；
- `SHELL_TIMEOUT_SECONDS`：默认 `30`；
- `SHELL_MAX_OUTPUT_CHARS`：默认 `12000`。

密钥只从环境变量读取，不写入记忆、日志、配置示例或异常详情。项目提供 `pyproject.toml`，约束 Python 为 `>=3.11,<3.12`，并明确列出 LangChain、LangGraph、langchain-openai、langchain-mcp-adapters、MCP、Pydantic Settings、python-dotenv、Rich、PyYAML 与 pytest 等依赖。

## 6. Skills 设计

每个 Skill 位于 `skills/<skill-name>/SKILL.md`。文件顶部使用 YAML frontmatter，至少包含 `name` 与 `description`，正文包含适用场景、执行步骤和约束。启动阶段只把名称和描述组成技能目录放入系统提示，避免一次性占用过多上下文。

Agent 通过两个 LangChain 工具使用技能：

- `list_skills()`：返回所有有效技能的名称与描述；
- `load_skill(name)`：安全读取指定技能全文。

技能名称只允许字母、数字、下划线和连字符。路径解析必须阻止跳出 `skills` 根目录。缺少元数据、YAML 损坏、重名或文件过大的 Skill 会被跳过并记录中文告警，不阻断 Agent 启动。项目附带一个示例 Skill，用于演示结构和按需加载行为。

## 7. 记忆设计

记忆目录包含：

- `memory/short_term.txt`：JSONL 格式的最近对话轮次；
- `memory/long_term.txt`：JSONL 格式的长期事实。

短期记录包含时间、会话 ID、角色与文本。每轮成功响应后追加用户消息与最终回答，只向模型注入当前会话最近 `SHORT_TERM_MAX_TURNS` 轮；超过上限时采用原子替换方式裁剪文件。

长期记录包含时间、内容和可选标签。Agent 获得 `remember_fact(content, tags)` 和 `forget_fact(id)` 工具，只有用户明确要求记住，或模型判断是跨会话稳定偏好时才写入。启动与每轮调用时读取有效长期事实并注入系统上下文。

写文件时先写同目录临时文件，再使用原子替换，降低进程中断导致文件损坏的概率。读取时逐行解析；损坏行会跳过并告警，其余记忆继续可用。记忆内容设置单条和总字符上限，避免文件无限增长与上下文溢出。

## 8. 内置工具与安全边界

文件工具包括读取文本、写入文本、列目录和正则搜索。所有用户或模型提供的路径先相对项目根目录解析，再验证最终绝对路径仍位于根目录内；拒绝 `..`、绝对路径逃逸和符号链接逃逸。读取、搜索和 Shell 输出均有字符上限。

计算工具使用 Python AST 白名单解析，只允许数字、括号和常见算术运算，不使用 `eval`。时间工具返回带时区的 ISO 8601 时间。

Shell 工具通过 `subprocess` 在项目根目录执行。命令先经过风险分类：

- 常见只读命令可自动执行；
- 删除、覆盖、权限修改、进程终止、磁盘/网络配置、安装脚本管道、访问项目外路径等命令判定为危险；
- 无法可靠分类的命令按危险处理。

危险命令在 CLI 显示完整命令、风险原因并要求明确输入确认；拒绝或非交互环境下返回“未执行”的 Observation。执行设置超时、捕获退出码，并截断过长输出。风险分类不是安全沙箱，README 必须明确本工具只适用于受信任的本地单用户环境。

## 9. MCP 设计

`mcp_servers.json` 保存多个 Server 配置，支持：

- stdio：`command` 与 `args`；
- HTTP：`url` 与 `transport`。

`MultiServerMCPClient` 使用工具名前缀，避免不同 Server 同名冲突，并启用工具错误转消息，使模型有机会根据失败结果自我修正。启动时逐个加载服务：单个服务配置错误、进程启动失败或网络不可达只产生告警并跳过该服务，内置工具和其他 MCP 服务仍可使用。

本地示例使用 FastMCP，通过 stdio 暴露简单、无副作用的文本统计与加法工具。默认配置可直接启动此示例。首版采用适合无状态工具的短连接方式，并在 README 说明：需要跨调用状态的 MCP Server 应改为显式持久会话。

## 10. 主 ReAct 数据流

1. CLI 加载配置；缺少 API Key 时打印修复方式并退出；
2. 加载有效 Skills、短期记忆、长期记忆和内置工具；
3. 尝试加载各 MCP Server 工具，并汇总成功与失败结果；
4. 创建包含角色、安全边界、技能目录和记忆摘要的系统提示；
5. 将用户消息提交给 Agent；
6. 模型选择直接回答或生成结构化工具调用；
7. 工具执行并把结果作为 Observation 返回模型；
8. 重复步骤 6 至 7，直到模型返回最终回答或达到步数上限；
9. CLI 展示最终回答并持久化本轮短期记忆；
10. 用户输入 `/clear` 清除当前短期记忆，输入 `/exit` 正常退出。

只展示工具行为，不展示模型私有思维链。

## 11. 异常与恢复

- 配置错误：一次性汇总缺失或非法字段，给出 `.env.example` 指引；
- 模型限流、超时和暂时网络错误：最多 3 次指数退避，仍失败则返回中文提示；
- 鉴权、余额或模型不存在：不盲目重试，明确提示检查 Key、地域、模型和账户；
- 工具参数错误与运行异常：转换为短文本 Observation，由模型修正参数或改用其他工具；
- Agent 步数超限：停止循环，提示用户缩小任务或继续；
- MCP 服务失败：隔离到单个服务并降级运行；
- Skill 或记忆损坏：跳过损坏项并保留有效内容；
- Shell 超时：终止子进程并返回已超时；
- Ctrl+C：取消当前轮但保留 CLI；连续退出命令才结束程序；
- 空模型响应：作为可重试异常处理，最终给出明确提示。

日志不得包含 API Key、完整环境变量或未经截断的大段工具输出。

## 12. 手写 ReAct 教学示例

`manual_react_demo.py` 使用 ChatModel 的 `bind_tools` 注册基础工具，并维护 `SystemMessage`、`HumanMessage`、`AIMessage` 与 `ToolMessage` 列表。每轮：

1. 请求模型；
2. 若没有 `tool_calls`，输出最终答案并结束；
3. 若存在调用，按调用 ID 执行对应工具；
4. 捕获工具异常，将失败文本写入 `ToolMessage`；
5. 把模型消息和所有工具观察结果追加到消息列表；
6. 进入下一轮。

循环硬限制为 8 轮，未知工具、非法 JSON 参数、工具异常和空回答都有明确兜底。示例代码用中文注释解释每一步，但不实现主程序的完整 MCP 生命周期、长期记忆和 Shell 审批，避免教学代码失焦。

## 13. 测试策略

测试以 pytest 为主，遵循先失败测试、后最小实现：

- 配置：默认值、缺少 Key、非法数值；
- 文件工具：正常读写、路径穿越、绝对路径、符号链接逃逸、输出截断；
- Shell：安全命令自动执行、危险命令确认/拒绝、未知命令保守处理、超时；
- 记忆：追加、读取、裁剪、原子替换、损坏 JSONL 跳过；
- Skills：有效解析、错误 YAML、重名、非法名称、路径逃逸；
- MCP：配置解析、部分服务失败时降级、工具名前缀；
- Agent：假模型直接回答、单次工具调用、多次工具调用、工具异常恢复、步数上限；
- 手写示例：无工具结束、工具调用后结束、未知工具与循环超限；
- CLI：退出、清除记忆和 Ctrl+C 恢复。

真实百炼和 MCP 网络调用不进入默认单元测试；README 提供独立冒烟测试命令。最终验收要求全部测试通过，CLI 能连接百炼，能自主读取项目文件、加载示例 Skill、调用本地 MCP 工具，并在危险 Shell 命令前等待确认。

## 14. 文档与使用体验

README 使用中文，覆盖：

- Python 3.11 虚拟环境创建；
- 依赖安装；
- 从 `.env.example` 创建 `.env` 并配置百炼；
- 模型与地域 Base URL 的注意事项；
- 启动主 CLI、启动手写 demo、运行测试；
- 增加 Skill、内置工具和 MCP Server 的方法；
- 短期与长期记忆文件格式；
- Shell 权限风险与使用边界；
- 常见鉴权、模型、MCP 和 Windows 命令问题。

首版完成标准是：全新 Windows PowerShell 环境按 README 操作即可运行；核心行为有自动化测试；外部服务故障不会造成无堆栈说明的崩溃；代码结构清晰，关键流程配有中文注释。
