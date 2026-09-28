from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode  # LangGraph已封装好的工具执行节点

from agent_assistant.config import Settings, load_settings
from pathlib import Path
from agent_assistant.model import create_chat_model


# ========== 1. 定义状态（State） ==========
# State是整个图运行过程中传递、累积的数据结构
class AgentState(TypedDict):
    # add_messages是一个reducer：新消息会"追加"到列表，而不是覆盖
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ========== 2. 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    return f"{city}今天晴天，25度"

tools = [get_weather]


# ========== 3. 定义模型（绑定工具） ==========
settings = load_settings(Path.cwd())
model = create_chat_model(settings)
model_with_tools = model.bind_tools(tools)  # 关键：让LLM知道有哪些工具可用


# ========== 4. 定义节点函数 ==========

# 4.1 LLM节点：调用模型，返回新消息（可能包含tool_calls）
def call_model(state: AgentState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    # 返回的内容会通过reducer自动追加到state["messages"]中
    return {"messages": [response]}

# 4.2 工具节点：LangGraph内置的ToolNode，会自动读取最后一条消息里的tool_calls并执行对应工具
tool_node = ToolNode(tools)


# ========== 5. 定义条件路由函数 ==========
# 判断LLM返回的最后一条消息里，是否包含tool_calls
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:  # 如果LLM决定调用工具
        return "tools"
    return END  # 否则直接结束


# ========== 6. 构建图 ==========
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边：agent节点执行完后，根据should_continue的返回值决定去哪
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# 添加普通边：工具执行完后，一定回到agent节点，让LLM根据工具结果继续思考
workflow.add_edge("tools", "agent")

# 编译成可执行对象
app = workflow.compile()


# ========== 7. 运行 ==========
result = app.invoke({
    "messages": [
        SystemMessage(content="你是一个旅游助手，帮用户查询天气。"),
        ("user", "上海今天适合旅游吗？"),
    ]
})

print(result["messages"][-1].content)