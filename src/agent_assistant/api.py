"""FastAPI：对外暴露 create_agent 简单示例接口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_assistant.agent_executor import ask_agent, build_demo_agent
from agent_assistant.config import load_settings


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")


class ChatResponse(BaseModel):
    reply: str


def create_app() -> FastAPI:
    state: dict[str, Any] = {"agent": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            settings = load_settings(Path.cwd())
            state["agent"] = build_demo_agent(settings=settings)
        except Exception as exc:
            raise RuntimeError(
                "Agent 初始化失败：请检查 .env 中的 DASHSCOPE_API_KEY 等配置。"
            ) from exc
        yield

    app = FastAPI(title="Agent Demo API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        agent = state["agent"]
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent 尚未就绪")
        try:
            reply = ask_agent(agent, body.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Agent 执行失败：{type(exc).__name__}"
            ) from exc
        return ChatResponse(reply=reply)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "agent_assistant.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
