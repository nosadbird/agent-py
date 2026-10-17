"""Demo agent 调用的子 Agent。"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from agent_assistant.config import load_settings
from agent_assistant.model import create_chat_model

settings = load_settings(Path.cwd())
sub_model = create_chat_model(settings)
subagent1 = create_agent(model=sub_model)


@tool(
    "行程规划助手",
    description="你是一个行程规划的助手，帮用户合理的规划形成",
)
def call_subagent1(query: str):
    result = subagent1.invoke({"messages": [{"role": "user", "content": query}]})
    print("subAgent:" + result["messages"][-1].content)
    return result["messages"][-1].content
