"""Tests for the Anthropic Messages API translation layer."""

import base64
import json

import pytest
from fastapi import HTTPException

from api.models import bedrock as bedrock_module
from api.models.messages import BedrockMessagesModel, _StreamSession, resolve_model
from api.schema import AnthropicCountTokensRequest, AnthropicMessagesRequest

CLAUDE = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
QWEN = "qwen.qwen3-coder-480b-a35b-v1:0"


@pytest.fixture(autouse=True)
def model_list(monkeypatch):
    monkeypatch.setattr(
        bedrock_module,
        "bedrock_model_list",
        {
            "anthropic.claude-sonnet-4-5-20250929-v1:0": {"modalities": ["TEXT", "IMAGE"]},
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"modalities": ["TEXT", "IMAGE"]},
            CLAUDE: {"modalities": ["TEXT", "IMAGE"]},
            "us.anthropic.claude-opus-4-6-v1": {"modalities": ["TEXT", "IMAGE"]},
            "global.anthropic.claude-opus-4-5-20251101-v1:0": {"modalities": ["TEXT", "IMAGE"]},
            QWEN: {"modalities": ["TEXT"]},
        },
    )


def build(model=CLAUDE, betas=None, **kwargs):
    kwargs.setdefault("max_tokens", 1024)
    kwargs.setdefault("messages", [{"role": "user", "content": "Hi"}])
    request = AnthropicMessagesRequest(model=model, **kwargs)
    return BedrockMessagesModel().build_converse_args(request, betas)


def sse_events(payload: bytes) -> list[dict]:
    events = []
    for block in payload.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n")
        data = json.loads(data_line[len("data: ") :])
        assert event_line[len("event: ") :] == data["type"]
        events.append(data)
    return events


def stream(chunks: list[dict]) -> list[dict]:
    session = _StreamSession("claude-sonnet-4-5")
    payload = session.start()
    for chunk in chunks:
        payload += b"".join(session.handle(chunk))
    payload += b"".join(session.finish())
    return sse_events(payload)


# Model names


@pytest.mark.parametrize(
    "name, expected",
    [
        ("claude-sonnet-4-5", CLAUDE),
        ("claude-sonnet-4-5-20250929", CLAUDE),
        ("claude-opus-4-6", "us.anthropic.claude-opus-4-6-v1"),
        # Must not pick up claude-opus-4-5 or claude-opus-4-6.
        ("claude-opus-4", "claude-opus-4"),
        (QWEN, QWEN),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ],
)
def test_resolve_model(name, expected):
    assert resolve_model(name) == expected


def test_unknown_model_is_rejected():
    with pytest.raises(HTTPException) as exc:
        build(model="claude-nonexistent-9")
    assert exc.value.status_code == 400


# Requests


def test_basic_request():
    args = build(system="Be brief.", temperature=0.5, stop_sequences=["END"])

    assert args["modelId"] == CLAUDE
    assert args["system"] == [{"text": "Be brief."}]
    assert args["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert args["inferenceConfig"] == {"maxTokens": 1024, "temperature": 0.5, "stopSequences": ["END"]}
    assert "toolConfig" not in args


def test_cache_control_becomes_cache_points():
    args = build(
        system=[
            {"type": "text", "text": "billing header"},
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ],
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        tools=[{"name": "Bash", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
    )

    assert args["system"] == [
        {"text": "billing header"},
        {"text": "You are Claude Code."},
        {"cachePoint": {"type": "default", "ttl": "1h"}},
    ]
    assert args["messages"][0]["content"] == [{"text": "Hi"}, {"cachePoint": {"type": "default"}}]
    assert args["toolConfig"]["tools"][1] == {"cachePoint": {"type": "default"}}


def test_cache_control_is_dropped_for_models_without_caching():
    args = build(
        model=QWEN,
        system=[{"type": "text", "text": "Sys", "cache_control": {"type": "ephemeral"}}],
    )

    assert args["system"] == [{"text": "Sys"}]


def test_tool_round_trip_keeps_thinking_signatures():
    args = build(
        messages=[
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Use ls.", "signature": "sig-1"},
                    {"type": "redacted_thinking", "data": base64.b64encode(b"secret").decode()},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": True,
                        "content": [{"type": "text", "text": "permission denied"}],
                    }
                ],
            },
        ],
        tools=[{"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}}],
        tool_choice={"type": "any"},
    )

    assert args["messages"][1]["content"] == [
        {"reasoningContent": {"reasoningText": {"text": "Use ls.", "signature": "sig-1"}}},
        {"reasoningContent": {"redactedContent": b"secret"}},
        {"toolUse": {"toolUseId": "toolu_1", "name": "Bash", "input": {"command": "ls"}}},
    ]
    assert args["messages"][2]["content"] == [
        {"toolResult": {"toolUseId": "toolu_1", "content": [{"text": "permission denied"}], "status": "error"}}
    ]
    assert args["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "Bash",
                    "description": "Run a command",
                    "inputSchema": {"json": {"type": "object"}},
                }
            }
        ],
        "toolChoice": {"any": {}},
    }


def test_unsigned_thinking_is_not_replayed():
    args = build(
        model=QWEN,
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": ""}, {"type": "text", "text": "Hello"}]},
            {"role": "user", "content": "Again"},
        ],
    )

    assert args["messages"][1]["content"] == [{"text": "Hello"}]


def test_tool_history_without_tools_gets_placeholder_specs():
    args = build(
        messages=[
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}]},
            {"role": "user", "content": "Summarise the conversation."},
        ],
    )

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == ["Bash"]


def test_server_tools_are_dropped():
    args = build(tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}])

    assert "toolConfig" not in args


def test_consecutive_turns_are_merged_and_blank_text_dropped():
    args = build(
        messages=[
            {"role": "user", "content": "One"},
            {"role": "user", "content": [{"type": "text", "text": "  "}, {"type": "text", "text": "Two"}]},
        ]
    )

    assert args["messages"] == [{"role": "user", "content": [{"text": "One"}, {"text": "Two"}]}]


def test_mid_conversation_system_messages_join_a_user_turn():
    args = build(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": [{"type": "text", "text": "Env info"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "system", "content": "Reminder"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ],
        tools=[{"name": "Bash", "input_schema": {"type": "object"}}],
    )

    assert args["messages"][0]["content"] == [{"text": "Hi"}, {"text": "Env info"}]
    # Tool results have to stay first in their turn.
    assert args["messages"][2]["content"] == [
        {"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}},
        {"text": "Reminder"},
    ]


def test_images_and_documents():
    png = base64.b64encode(b"png-bytes").decode()
    pdf = base64.b64encode(b"%PDF").decode()
    args = build(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                    {"type": "document", "title": "Q3 report.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf}},
                ],
            }
        ]
    )

    image, document = args["messages"][0]["content"]
    assert image == {"image": {"format": "png", "source": {"bytes": b"png-bytes"}}}
    assert document == {"document": {"format": "pdf", "name": "Q3 report pdf 1", "source": {"bytes": b"%PDF"}}}


def test_images_are_rejected_for_text_only_models():
    with pytest.raises(HTTPException) as exc:
        build(
            model=QWEN,
            messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}],
        )
    assert exc.value.status_code == 400


def test_claude_fields_and_betas_are_forwarded():
    args = build(
        max_tokens=32000,
        top_p=0.9,
        top_k=5,
        thinking={"type": "adaptive", "display": "omitted"},
        output_config={"effort": "high"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        betas=["claude-code-20250219", "interleaved-thinking-2025-05-14", "advisor-tool-2026-03-01"],
    )

    assert args["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "top_k": 5,
        # Only the betas Bedrock accepts.
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
    }
    # top_p cannot be combined with thinking.
    assert "topP" not in args["inferenceConfig"]


def test_claude_fields_are_dropped_for_other_models():
    args = build(
        model=QWEN,
        thinking={"type": "enabled", "budget_tokens": 2048},
        output_config={"effort": "high"},
        betas=["interleaved-thinking-2025-05-14"],
    )

    assert "additionalModelRequestFields" not in args


def test_count_tokens_request_has_no_inference_config():
    request = AnthropicCountTokensRequest(model=CLAUDE, messages=[{"role": "user", "content": "Hi"}])
    args = BedrockMessagesModel().build_converse_args(request)

    assert "inferenceConfig" not in args


def test_count_tokens_falls_back_to_an_estimate(monkeypatch):
    def unsupported(**kwargs):
        raise RuntimeError("The provided model doesn't support counting tokens.")

    monkeypatch.setattr("api.models.messages.bedrock_runtime.count_tokens", unsupported)
    request = AnthropicCountTokensRequest(model=QWEN, messages=[{"role": "user", "content": "Hello there"}])

    import asyncio

    assert asyncio.run(BedrockMessagesModel().count_tokens(request)) > 0


# Responses


def test_response_content_conversion():
    content = BedrockMessagesModel.content(
        [
            {"reasoningContent": {"reasoningText": {"text": "think", "signature": "sig"}}},
            {"reasoningContent": {"redactedContent": b"secret"}},
            {"text": "Hello"},
            {"toolUse": {"toolUseId": "t1", "name": "Bash", "input": {"command": "ls"}}},
        ]
    )

    assert content == [
        {"type": "thinking", "thinking": "think", "signature": "sig"},
        {"type": "redacted_thinking", "data": base64.b64encode(b"secret").decode()},
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
    ]


@pytest.mark.parametrize(
    "bedrock_reason, expected",
    [("tool_use", "tool_use"), ("max_tokens", "max_tokens"), ("guardrail_intervened", "refusal"), (None, "end_turn")],
)
def test_stop_reasons(bedrock_reason, expected):
    assert BedrockMessagesModel.stop(bedrock_reason, None)[0] == expected


def test_stream_events():
    events = stream(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Let me"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "Running ls"}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"contentBlockStart": {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "t1", "name": "Bash"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '{"command":'}}}},
            {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": ' "ls"}'}}}},
            {"contentBlockStop": {"contentBlockIndex": 2}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 100}}},
        ]
    )

    assert [e["type"] for e in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1]["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}
    assert events[3]["delta"] == {"type": "signature_delta", "signature": "sig"}
    assert events[5]["index"] == 1 and events[5]["content_block"]["type"] == "text"
    assert events[8]["content_block"] == {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
    partial = "".join(e["delta"]["partial_json"] for e in events[9:11])
    assert json.loads(partial) == {"command": "ls"}
    assert events[12]["delta"]["stop_reason"] == "tool_use"
    assert events[12]["usage"]["input_tokens"] == 10
    assert events[12]["usage"]["cache_read_input_tokens"] == 100


def test_stream_closes_blocks_bedrock_left_open():
    events = stream([{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}}])

    assert [e["type"] for e in events][-3:] == ["content_block_stop", "message_delta", "message_stop"]


def test_tool_result_images_stay_inside_for_claude():
    args = build(
        messages=[
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}],
                    }
                ],
            },
        ],
        tools=[{"name": "Read", "input_schema": {"type": "object"}}],
    )

    assert args["messages"][1]["content"] == [
        {"toolResult": {"toolUseId": "t1", "content": [{"image": {"format": "png", "source": {"bytes": b"\x00"}}}]}}
    ]


def test_tool_result_images_are_moved_beside_the_result_for_other_models(monkeypatch):
    gpt = "global.openai.gpt-6-luna"
    bedrock_module.bedrock_model_list[gpt] = {"modalities": ["TEXT", "IMAGE"]}
    args = build(
        model=gpt,
        messages=[
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "chart.png"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
                        ],
                    }
                ],
            },
        ],
        tools=[{"name": "Read", "input_schema": {"type": "object"}}],
    )

    result, image = args["messages"][1]["content"]
    assert [part["text"] for part in result["toolResult"]["content"]] == [
        "chart.png",
        "[1 attachment(s) follow this tool result]",
    ]
    assert image == {"image": {"format": "png", "source": {"bytes": b"\x00"}}}
