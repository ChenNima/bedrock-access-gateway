"""Tests for the Responses API translation layer."""

import json

import pytest
from fastapi import HTTPException

from api.models.responses import BedrockResponsesModel
from api.schema import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    ResponsesRequest,
)

MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"


class FakeChatModel:
    """Stands in for BedrockModel so no Bedrock call is made."""

    def __init__(self, response=None, chunks=None):
        self.response = response
        self.chunks = chunks or []
        self.chat_request = None
        self.stream = None

    def validate(self, chat_request):
        pass

    async def invoke(self, chat_request, stream=False):
        self.chat_request = chat_request
        self.stream = stream
        if stream:
            return {"stream": self.chunks}
        return self.response


def build(**kwargs):
    """Convert a Responses request into the internal chat request."""
    model = BedrockResponsesModel(chat_model=FakeChatModel())
    return model.build_chat_request(ResponsesRequest(model=MODEL, **kwargs))


def sse_events(payload: bytes) -> list[dict]:
    """Parse an SSE payload into the list of event data objects."""
    events = []
    for block in payload.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        data = json.loads(lines[1][len("data: ") :])
        # The event name always matches the type inside the payload.
        assert lines[0][len("event: ") :] == data["type"]
        events.append(data)
    return events


async def collect(generator) -> list[dict]:
    payload = b""
    async for chunk in generator:
        payload += chunk
    return sse_events(payload)


def test_string_input_becomes_a_user_message():
    chat_request = build(input="Hello!")

    assert chat_request.messages == [UserMessage(content="Hello!")]


def test_instructions_become_a_system_message():
    chat_request = build(instructions="Be brief.", input="Hello!")

    assert chat_request.messages[0] == SystemMessage(content="Be brief.")
    assert isinstance(chat_request.messages[1], UserMessage)


def test_message_items_carry_text_and_images():
    chat_request = build(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,Zm8="},
                ],
            }
        ]
    )

    content = chat_request.messages[0].content
    assert content[0].text == "What is this?"
    assert content[1].image_url.url == "data:image/png;base64,Zm8="


def test_role_only_items_and_developer_role():
    chat_request = build(
        input=[
            {"role": "developer", "content": "Follow the rules."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
            {"role": "user", "content": "Again"},
        ]
    )

    roles = [message.role for message in chat_request.messages]
    assert roles == ["developer", "user", "assistant", "user"]
    assert chat_request.messages[2] == AssistantMessage(content="Hello")


def test_function_call_round_trip_keeps_the_call_id():
    chat_request = build(
        input=[
            {"role": "user", "content": "weather?"},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "tooluse_abc",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
            {"type": "function_call_output", "call_id": "tooluse_abc", "output": "18C"},
        ]
    )

    assistant = chat_request.messages[1]
    assert assistant.tool_calls[0].id == "tooluse_abc"
    assert assistant.tool_calls[0].function.name == "get_weather"
    assert chat_request.messages[2] == ToolMessage(tool_call_id="tooluse_abc", content="18C")


def test_blank_function_call_arguments_stay_valid_json():
    chat_request = build(
        input=[
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "t1", "name": "ping", "arguments": ""},
        ]
    )

    assert json.loads(chat_request.messages[1].tool_calls[0].function.arguments) == {}


def test_structured_function_call_output_is_flattened():
    chat_request = build(
        input=[
            {"role": "user", "content": "go"},
            {
                "type": "function_call_output",
                "call_id": "t1",
                "output": [{"type": "output_text", "text": "done"}],
            },
        ]
    )

    assert chat_request.messages[1].content == "done"


def test_reasoning_items_are_dropped():
    # Bedrock only accepts a reasoning block back with the signature it issued, which the
    # Responses wire format does not carry.
    chat_request = build(
        input=[
            {"role": "user", "content": "Hi"},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}]},
        ]
    )

    assert [message.role for message in chat_request.messages] == ["user"]


def test_flat_function_tools_are_converted():
    chat_request = build(
        input="Hi",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
            {"type": "web_search"},
        ],
    )

    assert len(chat_request.tools) == 1
    assert chat_request.tools[0].function.name == "get_weather"
    assert chat_request.tools[0].function.description == "Get weather"


def test_tool_choice_shapes():
    assert build(input="Hi").tool_choice == "auto"
    assert build(input="Hi", tool_choice="required").tool_choice == "required"
    # Bedrock has no "none", so it degrades to letting the model decide.
    assert build(input="Hi", tool_choice="none").tool_choice == "auto"
    assert build(input="Hi", tool_choice={"type": "function", "name": "f"}).tool_choice == {
        "function": {"name": "f"}
    }


def test_reasoning_effort_is_mapped_and_gets_a_token_budget():
    chat_request = build(input="Hi", reasoning={"effort": "minimal"})

    assert chat_request.reasoning_effort == "low"
    # Claude rejects reasoning without an explicit maxTokens.
    assert chat_request.max_tokens is not None

    assert build(input="Hi", reasoning={"effort": "high"}, max_output_tokens=1000).max_tokens == 1000
    assert build(input="Hi", reasoning={"effort": "none"}).reasoning_effort is None


def test_empty_input_is_rejected():
    with pytest.raises(HTTPException) as exc:
        build(instructions="Be brief.", input=[])

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_non_streaming_response_maps_every_block_type():
    chat_model = FakeChatModel(
        response={
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
                        {"text": "Here you go"},
                        {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather", "input": {"city": "Paris"}}},
                    ]
                }
            },
            "usage": {"inputTokens": 8, "outputTokens": 4, "totalTokens": 12, "cacheReadInputTokens": 3},
            "stopReason": "tool_use",
        }
    )
    model = BedrockResponsesModel(chat_model=chat_model)

    response = await model.respond(ResponsesRequest(model=MODEL, input="Hi"))

    assert response.object == "response"
    assert response.id.startswith("resp_")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["reasoning", "message", "function_call"]
    assert response.output[0].summary[0].text == "thinking"
    assert response.output[1].content[0].text == "Here you go"
    assert response.output[2].call_id == "tooluse_1"
    assert json.loads(response.output[2].arguments) == {"city": "Paris"}
    assert response.usage.input_tokens == 8
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 12
    assert response.usage.input_tokens_details.cached_tokens == 3
    assert response.usage.output_tokens_details.reasoning_tokens > 0


@pytest.mark.asyncio
async def test_truncated_response_is_incomplete():
    chat_model = FakeChatModel(
        response={
            "output": {"message": {"content": [{"text": "half a sen"}]}},
            "usage": {"outputTokens": 4, "totalTokens": 12},
            "stopReason": "max_tokens",
        }
    )

    response = await BedrockResponsesModel(chat_model=chat_model).respond(
        ResponsesRequest(model=MODEL, input="Hi")
    )

    assert response.status == "incomplete"
    assert response.incomplete_details == {"reason": "max_output_tokens"}


@pytest.mark.asyncio
async def test_streaming_emits_reasoning_text_and_tool_call_events():
    chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": " there"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 2,
                "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '{"city":'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '"Paris"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 2}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"outputTokens": 4, "totalTokens": 12}}},
    ]
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))

    events = await collect(model.respond_stream(ResponsesRequest(model=MODEL, input="Hi", stream=True)))

    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-1] == "response.completed"
    # The signature delta carries nothing a client can use, so it emits no event.
    assert types.count("response.reasoning_summary_text.delta") == 1
    assert types.count("response.output_text.delta") == 2
    assert types.count("response.output_item.done") == 3

    # sequence_number is monotonic across the whole stream.
    assert [event["sequence_number"] for event in events] == list(range(1, len(events) + 1))

    text_deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert text_deltas == "Hello there"

    arguments = [e for e in events if e["type"] == "response.function_call_arguments.done"][0]
    assert json.loads(arguments["arguments"]) == {"city": "Paris"}

    done = [e for e in events if e["type"] == "response.output_item.done"]
    assert [e["item"]["type"] for e in done] == ["reasoning", "message", "function_call"]
    # One output item per Bedrock content block, indexed in the order the blocks opened.
    assert [e["output_index"] for e in done] == [0, 1, 2]

    final = events[-1]["response"]
    assert [item["type"] for item in final["output"]] == ["reasoning", "message", "function_call"]
    assert final["usage"]["total_tokens"] == 12
    assert final["status"] == "completed"


@pytest.mark.asyncio
async def test_streaming_reports_a_truncated_response_as_incomplete():
    chunks = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "half a sen"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "max_tokens"}},
        {"metadata": {"usage": {"outputTokens": 4, "totalTokens": 12}}},
    ]
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))

    events = await collect(model.respond_stream(ResponsesRequest(model=MODEL, input="Hi", stream=True)))

    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.asyncio
async def test_streaming_failure_becomes_a_response_failed_event():
    class BoomChatModel(FakeChatModel):
        async def invoke(self, chat_request, stream=False):
            raise RuntimeError("bedrock is down")

    model = BedrockResponsesModel(chat_model=BoomChatModel())

    events = await collect(model.respond_stream(ResponsesRequest(model=MODEL, input="Hi", stream=True)))

    # response.created still comes first, otherwise a client has nothing to attach the
    # failure to and the OpenAI SDK raises instead of surfacing the error.
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "error",
        "response.failed",
    ]
    assert "bedrock is down" in events[2]["message"]
    assert events[3]["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_unclosed_blocks_are_closed_before_completion():
    # Bedrock always sends contentBlockStop, but a dropped one must not leave a half-open item.
    chunks = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"outputTokens": 1, "totalTokens": 2}}},
    ]
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))

    events = await collect(model.respond_stream(ResponsesRequest(model=MODEL, input="Hi", stream=True)))

    assert [event["type"] for event in events][-2:] == ["response.output_item.done", "response.completed"]
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hi"


@pytest.mark.asyncio
async def test_response_echoes_the_request_settings():
    chat_model = FakeChatModel(
        response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"outputTokens": 1, "totalTokens": 3},
            "stopReason": "end_turn",
        }
    )
    request = ResponsesRequest(
        model=MODEL,
        input="Hi",
        instructions="Be brief.",
        max_output_tokens=100,
        temperature=0.5,
        top_p=0.9,
        metadata={"trace": "1"},
        reasoning={"effort": "low"},
    )

    response = await BedrockResponsesModel(chat_model=chat_model).respond(request)

    assert response.model == MODEL
    assert response.instructions == "Be brief."
    assert response.max_output_tokens == 100
    assert response.temperature == 0.5
    assert response.top_p == 0.9
    assert response.metadata == {"trace": "1"}
    assert response.reasoning.effort == "low"
    assert response.store is False
