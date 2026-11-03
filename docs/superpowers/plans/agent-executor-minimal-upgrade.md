# agent_executor 最小升级计划

## 目标与边界

只改 `agent_executor.py` 主链路及其直接依赖，不改 `agent.py`、`cli.py`、`manual_react_demo.py`、MCP 或旧记忆实现。

目标按顺序完成：

1. 用可观察事件替代原始思维链展示。
2. 把现有 `run_shell` 升级为可选 Bash、实时输出、可取消的终端工具。
3. 补上文件 diff、写入确认、撤销和修改后验证提示，形成最小编码闭环。

保持简单：只保留一个 `run_shell` 工具；不引入事件总线框架、数据库、任务队列或完整沙箱。

## 第一阶段：可观察过程

### 1. 增加一个很小的事件模型

新增 `src/agent_assistant/agent_events.py`：

- `AgentEvent(kind, text)`，只保存事件类型和可展示文本。
- `EventHandler` 回调类型。
- 事件类型只覆盖当前需要：状态、计划、工具开始、工具结果、终端输出、文件变更和回答。

不设计复杂 JSON schema。CLI 直接打印事件；以后 API 要做 SSE 时可复用。

### 2. 保留深度思考协议，但不展示原始思考

修改 `thinking_output.py`：

- 保留百炼 `reasoning_content` 的接收、合并和下一轮回传逻辑，确保深度思考加工具调用仍能工作。
- 删除面向用户聚合和格式化思维链的职责。
- 对外提供一个简单的思考模型创建函数，避免类名继续暗示“思考输出”。

修改 `agent_executor.py`：

- 不再打印或返回原始 `reasoning_content`。
- 为兼容当前 `api.py`，暂时保留 `AgentTurn.thinking` 字段，但固定为空字符串，避免扩大本次范围。
- 流式执行改为展示真实的工具开始、工具结果和最终回答。
- 工具参数只展示安全摘要：文件工具显示路径，Shell 显示完整命令，写文件不显示整段 content。

### 3. 增加简单计划工具

修改 `agent_executor_tools.py`：

- 新增 `update_plan(steps: list[str])` 无副作用工具。
- 返回编号后的计划文本，由现有工具流自然展示。
- system prompt 只要求多步骤任务使用它，简单问答不使用。
- 计划只描述准备执行的动作，不要求模型解释内部推理。

## 第二阶段：升级 run_shell

### 4. 保持一个工具名，支持多个后端

修改 `config.py` 和 `.env.example`：

- 增加 `AGENT_SHELL=auto|powershell|bash|sh`，默认 `auto`。
- 增加可选 `AGENT_BASH_PATH`，Windows 用户可显式指定 Git Bash。
- `auto` 保持现有行为：Windows 用 PowerShell，Linux/macOS 优先 Bash，找不到时用 sh。

修改 `agent_executor_shell.py`：

- 抽出一个小型 Shell 后端解析函数。
- Bash 使用 `--noprofile --norc -c`，不加载用户启动脚本。
- 继续使用参数数组和 `shell=False`，工作目录仍固定为项目根目录。
- 找不到用户明确指定的后端时直接报清晰错误，不静默切换。

### 5. 增加实时输出、超时和取消

- 保留当前同步 `run()`，避免破坏非流式调用和已有测试。
- 新增真正的异步 `arun()`，使用 `asyncio.create_subprocess_exec`。
- stdout/stderr 按行发送 `terminal_output` 事件，同时累积受长度限制的最终工具结果。
- 超时或取消时终止进程；在 Windows/Unix 上尽量终止同一进程组，避免遗留子进程。
- `agent_executor.py` 的流式入口改用 `agent.astream()`，使 LangChain 调用异步 Shell 实现。
- 从免审批白名单中移除 `python -m pytest`，因为测试会执行仓库代码。

不在本阶段实现 PTY、交互式命令、后台常驻进程或完整 OS sandbox。

## 第三阶段：最小编码闭环

### 6. 文件写入前展示 diff 并确认

修改 `agent_executor_files.py`：

- `write_text` 写入前用 `difflib.unified_diff` 生成 diff。
- diff 通过 `file_change` 事件展示，并按现有输出上限截断。
- 新文件和覆盖文件都调用一次写入确认；拒绝后不修改磁盘。
- 写入仍使用现有项目路径和符号链接防护。

在 `agent_executor.py` 中增加与 Shell 审批风格一致的简短文件确认函数，并在构建工具时传入。

### 7. 提供一次安全撤销

- `ExecutorFileTools` 仅保存最近一次成功写入的内存快照，不增加持久化数据库。
- 注册 `undo_last_write` 工具。
- 撤销前校验文件仍是 Agent 上次写入的版本；若用户或其他程序随后改过文件，则拒绝覆盖。
- 新建文件的撤销是删除该文件，已有文件的撤销是恢复旧内容，两者都需要用户确认。

### 8. 引导修改后验证

修改 system prompt：

- 修改代码后运行最小相关测试或检查。
- 最终回答只陈述真实发生过的修改和验证结果。
- 未运行、失败或被拒绝的验证必须明确说明。

这里不新增自动工作流引擎，复用 `update_plan`、`run_shell` 和事件记录完成闭环。

## 测试与验收

更新或新增定向测试：

- `tests/test_thinking_output.py`：reasoning 仍能保存和回传，但没有用户展示接口。
- `tests/test_agent_executor_runtime.py`：不泄露 reasoning；正确展示计划、工具与回答事件。
- `tests/tools/test_agent_executor_shell.py`：后端选择、Bash 参数、pytest 需审批、实时输出、超时和取消。
- `tests/tools/test_agent_executor_files.py`：diff、拒绝不写入、确认后写入、撤销成功、外部改动后拒绝撤销。
- 为 `update_plan` 增加一个小型单元测试。

验收命令：

1. 先运行上述定向测试。
2. 运行 `python -m compileall -q src tests`。
3. 运行 `python -m pytest -q`；当前全量测试已有 `tests/test_agent.py:117` 的旧链路失败，本任务不修改它，报告时会明确区分新回归和既有失败。
4. 手工冒烟：执行一个“读文件 → 更新计划 → 改文件 → 审批 → 跑测试 → 总结”的任务，确认界面只显示可观察事件，不显示原始思维链。
