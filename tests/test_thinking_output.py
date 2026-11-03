from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent_assistant.config import Settings
from agent_assistant.thinking_output import create_thinking_model


def test_model_keeps_and_sends_reasoning(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dashscope_api_key="secret-key",
        dashscope_base_url="https://example.com/v1",
        dashscope_model="qwen-plus",
        _env_file=None,
    )
    model = create_thinking_model(settings)
    result = model._create_chat_result(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "最终回答",
                        "reasoning_content": "先比较两种做法",
                    }
                }
            ]
        }
    )
    assert result.generations[0].message.additional_kwargs["reasoning_content"] == "先比较两种做法"

    chunk = model._convert_chunk_to_generation_chunk(
        {"choices": [{"delta": {"reasoning_content": "先"}, "index": 0}]},
        AIMessageChunk,
        None,
    )
    assert chunk.message.additional_kwargs["reasoning_content"] == "先"

    payload = model._get_request_payload(
        [
            HumanMessage("继续"),
            AIMessage(content="回答", additional_kwargs={"reasoning_content": "因为上次的结论"}),
        ]
    )
    assert payload["messages"][1]["reasoning_content"] == "因为上次的结论"
    assert payload["extra_body"]["enable_thinking"] is True
