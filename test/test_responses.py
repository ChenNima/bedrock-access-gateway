"""Tests for the Responses API translation layer."""

import json
import re

import pytest
from fastapi import HTTPException

from api.models.responses import BedrockResponsesModel
from api.schema import (
    AssistantMessage,
    ResponsesRequest,
    SystemMessage,
    ToolMessage,
    UserMessage,
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
    tools = [{"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}}]
    assert build(input="Hi").tool_choice == "auto"
    assert build(input="Hi", tools=tools, tool_choice="required").tool_choice == "required"
    # Bedrock has no "none", so it degrades to letting the model decide.
    assert build(input="Hi", tool_choice="none").tool_choice == "auto"
    assert build(input="Hi", tools=tools, tool_choice={"type": "function", "name": "f"}).tool_choice == {
        "function": {"name": "f"}
    }
    # allowed_tools and hosted tool choices have no Bedrock counterpart.
    assert build(input="Hi", tools=tools, tool_choice={"type": "allowed_tools", "tools": []}).tool_choice == "auto"


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


def test_tool_history_without_tools_still_gets_a_tool_config(monkeypatch):
    # Codex compacts context by summarising a tool-using history and sends no tools.
    from api.models.bedrock import BedrockModel

    model = BedrockModel()
    monkeypatch.setattr(model, "_resolve_to_foundation_model", lambda model_id: model_id)
    chat_request = build(
        input=[
            {"type": "message", "role": "user", "content": "list files"},
            {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            {"type": "message", "role": "user", "content": "Summarize the conversation."},
        ]
    )

    args = model._parse_request(chat_request)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == ["shell"]
    assert "toolChoice" not in args["toolConfig"]


def test_plain_history_without_tools_has_no_tool_config(monkeypatch):
    from api.models.bedrock import BedrockModel

    model = BedrockModel()
    monkeypatch.setattr(model, "_resolve_to_foundation_model", lambda model_id: model_id)

    args = model._parse_request(build(input="hello"))

    assert "toolConfig" not in args


# --- Namespaces, name mapping and tool_choice ---------------------------------------------

TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def function(name, **extra):
    return {"type": "function", "name": name, "description": f"{name} tool", "parameters": SCHEMA, **extra}


def namespace(name, *members):
    return {"type": "namespace", "name": name, "description": f"{name} server", "tools": list(members)}


def bedrock_args(chat_request, monkeypatch):
    from api.models.bedrock import BedrockModel

    model = BedrockModel()
    monkeypatch.setattr(model, "_resolve_to_foundation_model", lambda model_id: model_id)
    return model._parse_request(chat_request)


def tool_names(chat_request):
    return [tool.function.name for tool in chat_request.tools]


def assert_valid_and_unique(names):
    assert all(TOOL_NAME.match(name) for name in names), names
    assert len(set(names)) == len(names)


def test_namespace_only_request_builds_a_tool_config_and_required_maps_to_any(monkeypatch):
    chat_request = build(
        input="check in",
        tools=[namespace("mcp__chorus", function("chorus_checkin"))],
        tool_choice="required",
    )

    args = bedrock_args(chat_request, monkeypatch)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == ["mcp__chorus__chorus_checkin"]
    assert args["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == SCHEMA
    assert args["toolConfig"]["toolChoice"] == {"any": {}}


def test_same_name_in_two_namespaces_gets_two_bedrock_names():
    chat_request = build(
        input="Hi",
        tools=[namespace("mcp__a", function("search")), namespace("mcp__b", function("search"))],
    )

    assert tool_names(chat_request) == ["mcp__a__search", "mcp__b__search"]


def test_long_and_invalid_names_are_sanitised_and_hashed():
    long_name = "x" * 70
    chat_request = build(
        input="Hi",
        tools=[
            function(long_name),
            namespace("mcp__srv", function("read.file"), function("y" * 60)),
            function("has space"),
        ],
    )

    names = tool_names(chat_request)
    assert_valid_and_unique(names)
    assert names[0].startswith("x" * 55 + "_")
    assert names[1].startswith("mcp__srv__read_file_")
    assert names[3].startswith("has_space_")


def test_top_level_and_namespace_collision_stays_unique():
    chat_request = build(
        input="Hi",
        tools=[function("mcp__chorus__chorus_checkin"), namespace("mcp__chorus", function("chorus_checkin"))],
    )

    names = tool_names(chat_request)
    assert_valid_and_unique(names)
    # The first declaration keeps the natural name.
    assert names[0] == "mcp__chorus__chorus_checkin"
    assert names[1] != names[0]


def test_hashed_names_resolve_their_own_collisions():
    from api.models.responses import ToolRegistry, _hashed_tool_name

    registry = ToolRegistry()
    taken = _hashed_tool_name("mcp__srv", "read.file")
    registry.add_declaration(function(taken))
    registry.add_declaration(namespace("mcp__srv", function("read.file")))

    names = list(registry.by_bedrock)
    assert_valid_and_unique(names)
    assert names[0] == taken


def test_name_mapping_is_deterministic():
    kwargs = dict(
        input="Hi",
        tools=[
            function("z" * 80),
            namespace("mcp__a", function("search"), function("bad/name")),
            namespace("mcp__b", function("search")),
            function("mcp__a__search"),
        ],
    )

    first, second = tool_names(build(**kwargs)), tool_names(build(**kwargs))

    assert first == second
    assert_valid_and_unique(first)


def test_named_tool_choice_with_namespace_selects_the_mapped_tool(monkeypatch):
    chat_request = build(
        input="Hi",
        tools=[namespace("mcp__a", function("search")), namespace("mcp__b", function("search"))],
        tool_choice={"type": "function", "name": "search", "namespace": "mcp__b"},
    )

    args = bedrock_args(chat_request, monkeypatch)

    assert args["toolConfig"]["toolChoice"] == {"tool": {"name": "mcp__b__search"}}


@pytest.mark.parametrize(
    "tools, tool_choice",
    [
        (None, "required"),
        # Hosted tools are dropped, so nothing is left to require.
        ([{"type": "web_search"}], "required"),
        ([function("search")], {"type": "function", "name": "missing"}),
        ([namespace("mcp__a", function("search"))], {"type": "function", "name": "search"}),
        ([namespace("mcp__a", function("search"))], {"type": "function", "name": "search", "namespace": "mcp__b"}),
    ],
)
@pytest.mark.asyncio
async def test_unsatisfiable_tool_choice_is_rejected_before_invoking_bedrock(tools, tool_choice):
    chat_model = FakeChatModel()
    model = BedrockResponsesModel(chat_model=chat_model)
    request = ResponsesRequest(model=MODEL, input="Hi", tools=tools, tool_choice=tool_choice)

    with pytest.raises(HTTPException) as exc:
        await model.respond(request)

    assert exc.value.status_code == 400
    assert exc.value.detail["param"] == "tool_choice"
    assert exc.value.detail["message"]
    assert chat_model.chat_request is None


def test_history_tool_does_not_satisfy_a_named_tool_choice():
    # A call replayed from history only gets a placeholder, which is not a declared tool.
    with pytest.raises(HTTPException):
        build(
            input=[
                {"role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "old", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
            tools=[function("new")],
            tool_choice={"type": "function", "name": "old"},
        )


def test_unsatisfiable_tool_choice_returns_an_openai_error_body(monkeypatch):
    from fastapi.testclient import TestClient

    from api import app as app_module
    from api.models import bedrock as bedrock_module
    from api.setting import API_ROUTE_PREFIX

    monkeypatch.setattr(bedrock_module, "bedrock_model_list", {MODEL: {"modalities": ["TEXT"]}})

    def fail_converse(**kwargs):
        raise AssertionError("Bedrock must not be called")

    monkeypatch.setattr(bedrock_module.bedrock_runtime, "converse", fail_converse)
    client = TestClient(app_module.app)

    response = client.post(
        f"{API_ROUTE_PREFIX}/responses",
        headers={"Authorization": "Bearer test-api-key"},
        json={"model": MODEL, "input": "Hi", "tool_choice": "required"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert isinstance(error["message"], str) and error["message"]
    assert error == {
        "message": error["message"],
        "type": "invalid_request_error",
        "param": "tool_choice",
        "code": None,
    }

    # A body that fails validation gets the same shape, without a param.
    response = client.post(
        f"{API_ROUTE_PREFIX}/responses",
        headers={"Authorization": "Bearer test-api-key"},
        json={"model": MODEL, "input": "Hi", "temperature": 5},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] is None


# --- Namespaced output and replay ------------------------------------------------------------

NAMESPACED_TOOLS = [function("shell"), namespace("mcp__chorus", function("chorus_checkin"))]


@pytest.mark.asyncio
async def test_non_streaming_tool_uses_map_back_to_namespace_and_name():
    chat_model = FakeChatModel(
        response={
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"toolUseId": "tooluse_1", "name": "mcp__chorus__chorus_checkin", "input": {}}},
                        {"toolUse": {"toolUseId": "tooluse_2", "name": "shell", "input": {"q": "ls"}}},
                    ]
                }
            },
            "usage": {"outputTokens": 4, "totalTokens": 12},
            "stopReason": "tool_use",
        }
    )
    request = ResponsesRequest(model=MODEL, input="Hi", tools=NAMESPACED_TOOLS)

    response = await BedrockResponsesModel(chat_model=chat_model).respond(request)

    items = response.model_dump()["output"]
    assert items[0]["type"] == "function_call"
    assert items[0]["name"] == "chorus_checkin"
    assert items[0]["namespace"] == "mcp__chorus"
    assert items[0]["call_id"] == "tooluse_1"
    assert items[1]["name"] == "shell"
    assert items[1]["call_id"] == "tooluse_2"
    assert "namespace" not in items[1]
    assert "namespace" not in json.loads(response.model_dump_json())["output"][1]


@pytest.mark.asyncio
async def test_streaming_tool_uses_map_back_to_namespace_and_name():
    chunks = [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "mcp__chorus__chorus_checkin"}},
            }
        },
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_2", "name": "shell"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"q":"ls"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"outputTokens": 4, "totalTokens": 12}}},
    ]
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))
    request = ResponsesRequest(model=MODEL, input="Hi", tools=NAMESPACED_TOOLS, stream=True)

    events = await collect(model.respond_stream(request))

    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = [e["item"] for e in events if e["type"] == "response.output_item.added"]
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    for items in (added, done, events[-1]["response"]["output"]):
        assert (items[0]["namespace"], items[0]["name"], items[0]["call_id"]) == (
            "mcp__chorus",
            "chorus_checkin",
            "tooluse_1",
        )
        assert (items[1]["name"], items[1]["call_id"]) == ("shell", "tooluse_2")
        assert "namespace" not in items[1]


def test_namespaced_function_call_replay_pairs_tool_use_and_result(monkeypatch):
    chat_request = build(
        input=[
            {"role": "user", "content": "check in"},
            {
                "type": "function_call",
                "call_id": "tooluse_1",
                "namespace": "mcp__chorus",
                "name": "chorus_checkin",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "tooluse_1", "output": "checked in"},
        ],
        tools=NAMESPACED_TOOLS,
    )

    messages = bedrock_args(chat_request, monkeypatch)["messages"]

    tool_use = messages[1]["content"][0]["toolUse"]
    tool_result = messages[2]["content"][0]["toolResult"]
    assert tool_use["name"] == "mcp__chorus__chorus_checkin"
    assert tool_use["toolUseId"] == tool_result["toolUseId"] == "tooluse_1"
    # No placeholder: the call maps onto the declared tool.
    assert tool_names(chat_request) == ["shell", "mcp__chorus__chorus_checkin"]


def test_replayed_call_to_an_undeclared_tool_gets_a_placeholder(monkeypatch):
    chat_request = build(
        input=[
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c1", "namespace": "mcp__gone", "name": "x", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
        tools=[function("shell")],
    )

    args = bedrock_args(chat_request, monkeypatch)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == ["shell", "mcp__gone__x"]
    assert args["messages"][1]["content"][0]["toolUse"]["name"] == "mcp__gone__x"


# --- Effective tools echo ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_echoes_only_effective_tools(caplog):
    chat_model = FakeChatModel(
        response={
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"outputTokens": 1, "totalTokens": 3},
            "stopReason": "end_turn",
        }
    )
    tools = [
        {"type": "web_search"},
        function("shell"),
        namespace(
            "mcp__chorus",
            function("chorus_checkin", defer_loading=True),
            {"type": "web_search"},
        ),
        namespace("mcp__empty", {"type": "file_search"}),
    ]
    request = ResponsesRequest(model=MODEL, input="Hi", tools=tools)
    model = BedrockResponsesModel(chat_model=chat_model)

    with caplog.at_level("WARNING"):
        response = await model.respond(request)

    # The deferred member is registered up front.
    assert tool_names(chat_model.chat_request) == ["shell", "mcp__chorus__chorus_checkin"]
    assert response.tools == [
        function("shell"),
        {**namespace("mcp__chorus"), "tools": [function("chorus_checkin", defer_loading=True)]},
    ]
    assert "web_search" in caplog.text


# --- tool_search, additional_tools and custom tools ------------------------------------------

# The shapes Codex rust-v0.160.0 sends: tool_search per
# core/src/tools/handlers/tool_search_spec.rs, custom exec per a captured responses_lite request.
TOOL_SEARCH = {
    "type": "tool_search",
    "execution": "client",
    "description": "# Tool discovery",
    "parameters": {
        "type": "object",
        "properties": {"limit": {"type": "number"}, "query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
LARK = "start: SOURCE\nSOURCE: /[\\s\\S]+/\n"
EXEC = {
    "type": "custom",
    "name": "exec",
    "description": "Run JavaScript code",
    "format": {"type": "grammar", "syntax": "lark", "definition": LARK},
}
DEFERRED_CHORUS = namespace("mcp__chorus", function("chorus_checkin", defer_loading=True))


def converse_response(*blocks, stop_reason="tool_use"):
    return {
        "output": {"message": {"content": list(blocks)}},
        "usage": {"outputTokens": 4, "totalTokens": 12},
        "stopReason": stop_reason,
    }


def tool_use_chunks(tool_use_id, name, *input_parts):
    return [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
        *[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": part}}}}
            for part in input_parts
        ],
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"outputTokens": 4, "totalTokens": 12}}},
    ]


def test_client_tool_search_becomes_a_bedrock_function(monkeypatch):
    chat_request = build(input="find chorus tools", tools=[TOOL_SEARCH])

    tools = bedrock_args(chat_request, monkeypatch)["toolConfig"]["tools"]

    assert [tool["toolSpec"]["name"] for tool in tools] == ["tool_search"]
    assert tools[0]["toolSpec"]["inputSchema"]["json"] == TOOL_SEARCH["parameters"]


def test_hosted_tool_search_is_dropped(caplog):
    with caplog.at_level("WARNING"):
        chat_request = build(input="hi", tools=[{**TOOL_SEARCH, "execution": "server"}, function("shell")])

    assert tool_names(chat_request) == ["shell"]
    assert "tool_search" in caplog.text


def test_tool_search_does_not_collide_with_a_function_of_the_same_name():
    chat_request = build(input="hi", tools=[TOOL_SEARCH, function("tool_search")])

    names = tool_names(chat_request)
    assert names[0] == "tool_search"
    assert_valid_and_unique(names)


@pytest.mark.asyncio
async def test_non_streaming_tool_search_call():
    chat_model = FakeChatModel(
        response=converse_response(
            {"toolUse": {"toolUseId": "tooluse_s", "name": "tool_search", "input": {"query": "chorus", "limit": 5}}}
        )
    )
    request = ResponsesRequest(model=MODEL, input="find", tools=[TOOL_SEARCH])

    response = await BedrockResponsesModel(chat_model=chat_model).respond(request)

    item = json.loads(response.model_dump_json())["output"][0]
    assert item["id"].startswith("tsc_")
    assert {k: v for k, v in item.items() if k != "id"} == {
        "type": "tool_search_call",
        "call_id": "tooluse_s",
        "execution": "client",
        "status": "completed",
        "arguments": {"query": "chorus", "limit": 5},
    }


@pytest.mark.asyncio
async def test_streaming_tool_search_call_is_buffered():
    chunks = tool_use_chunks("tooluse_s", "tool_search", '{"query": "cho', 'rus"}')
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))
    request = ResponsesRequest(model=MODEL, input="find", tools=[TOOL_SEARCH], stream=True)

    events = await collect(model.respond_stream(request))

    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    added, done = events[2]["item"], events[3]["item"]
    assert added["type"] == done["type"] == "tool_search_call"
    assert added["status"] == "in_progress"
    assert added["id"] == done["id"]
    assert (done["call_id"], done["execution"], done["status"]) == ("tooluse_s", "client", "completed")
    assert done["arguments"] == {"query": "chorus"}
    assert events[-1]["response"]["output"] == [done]


def tool_search_history():
    return [
        {"role": "user", "content": "check in to chorus"},
        {
            "type": "tool_search_call",
            "call_id": "tooluse_s",
            "execution": "client",
            "status": "completed",
            "arguments": {"query": "chorus checkin"},
        },
        {
            "type": "tool_search_output",
            "call_id": "tooluse_s",
            "execution": "client",
            "status": "completed",
            "tools": [DEFERRED_CHORUS],
        },
    ]


def test_tool_search_replay_registers_the_loaded_tools(monkeypatch):
    chat_request = build(input=tool_search_history(), tools=[TOOL_SEARCH])

    args = bedrock_args(chat_request, monkeypatch)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == [
        "tool_search",
        "mcp__chorus__chorus_checkin",
    ]
    messages = args["messages"]
    tool_use = messages[1]["content"][0]["toolUse"]
    tool_result = messages[2]["content"][0]["toolResult"]
    assert tool_use == {"toolUseId": "tooluse_s", "name": "tool_search", "input": {"query": "chorus checkin"}}
    assert tool_result["toolUseId"] == "tooluse_s"
    assert "mcp__chorus__chorus_checkin: chorus_checkin tool" in tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_tool_loaded_by_tool_search_is_called_with_its_namespace():
    chat_model = FakeChatModel(
        response=converse_response(
            {"toolUse": {"toolUseId": "tooluse_c", "name": "mcp__chorus__chorus_checkin", "input": {}}}
        )
    )
    request = ResponsesRequest(model=MODEL, input=tool_search_history(), tools=[TOOL_SEARCH])
    model = BedrockResponsesModel(chat_model=chat_model)

    response = await model.respond(request)

    item = response.model_dump()["output"][0]
    assert (item["type"], item["namespace"], item["name"], item["call_id"]) == (
        "function_call",
        "mcp__chorus",
        "chorus_checkin",
        "tooluse_c",
    )
    # Loaded tools reach Bedrock but the echo mirrors request.tools.
    assert response.tools == [TOOL_SEARCH]


def test_replayed_tool_search_without_a_declared_tool_search_gets_a_placeholder(monkeypatch):
    chat_request = build(input=tool_search_history(), tools=[function("shell")])

    args = bedrock_args(chat_request, monkeypatch)

    names = [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]]
    assert names == ["shell", "mcp__chorus__chorus_checkin", "tool_search"]
    assert args["messages"][1]["content"][0]["toolUse"]["name"] == "tool_search"


def test_additional_tools_item_registers_tools_without_a_message(monkeypatch):
    # The Codex responses_lite shape: no request.tools, a developer-role item with no content.
    chat_request = build(
        input=[
            {
                "type": "additional_tools",
                "id": "at_1",
                "role": "developer",
                "tools": [namespace("functions", EXEC, function("wait")), TOOL_SEARCH],
            },
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "be helpful"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say hi"}]},
        ]
    )

    args = bedrock_args(chat_request, monkeypatch)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == [
        "functions__exec",
        "functions__wait",
        "tool_search",
    ]
    assert [message.role for message in chat_request.messages] == ["developer", "user"]
    assert [message["role"] for message in args["messages"]] == ["user"]


def test_additional_tools_alone_is_not_a_conversation():
    with pytest.raises(HTTPException) as error:
        build(input=[{"type": "additional_tools", "role": "developer", "tools": [function("shell")]}])

    assert error.value.status_code == 400


def test_custom_tool_becomes_a_function_with_a_string_input(monkeypatch):
    chat_request = build(input="run it", tools=[EXEC])

    spec = bedrock_args(chat_request, monkeypatch)["toolConfig"]["tools"][0]["toolSpec"]

    assert spec["name"] == "exec"
    assert spec["inputSchema"]["json"] == {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "The raw input for the tool."}},
        "required": ["input"],
    }
    assert spec["description"].startswith("Run JavaScript code")
    assert LARK in spec["description"]


@pytest.mark.asyncio
async def test_non_streaming_custom_tool_call():
    chat_model = FakeChatModel(
        response=converse_response(
            {"toolUse": {"toolUseId": "tooluse_x", "name": "functions__exec", "input": {"input": "console.log(1)"}}},
            {"toolUse": {"toolUseId": "tooluse_y", "name": "exec", "input": {"input": "2"}}},
        )
    )
    request = ResponsesRequest(model=MODEL, input="run", tools=[namespace("functions", EXEC), EXEC])

    response = await BedrockResponsesModel(chat_model=chat_model).respond(request)

    items = json.loads(response.model_dump_json())["output"]
    assert items[0]["id"].startswith("ctc_")
    assert {k: v for k, v in items[0].items() if k != "id"} == {
        "type": "custom_tool_call",
        "call_id": "tooluse_x",
        "name": "exec",
        "namespace": "functions",
        "input": "console.log(1)",
        "status": "completed",
    }
    assert items[1]["type"] == "custom_tool_call"
    assert (items[1]["name"], items[1]["input"]) == ("exec", "2")
    assert "namespace" not in items[1]


@pytest.mark.asyncio
async def test_streaming_custom_tool_call_is_buffered():
    chunks = tool_use_chunks("tooluse_x", "exec", '{"input": "console', '.log(1)"}')
    model = BedrockResponsesModel(chat_model=FakeChatModel(chunks=chunks))
    request = ResponsesRequest(model=MODEL, input="run", tools=[EXEC], stream=True)

    events = await collect(model.respond_stream(request))

    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
    ]
    added, done = events[2]["item"], events[5]["item"]
    assert (added["type"], added["status"], added["input"]) == ("custom_tool_call", "in_progress", "")
    assert events[3]["delta"] == events[4]["input"] == "console.log(1)"
    assert events[3]["item_id"] == events[4]["item_id"] == done["id"]
    assert (done["call_id"], done["name"], done["input"], done["status"]) == (
        "tooluse_x",
        "exec",
        "console.log(1)",
        "completed",
    )
    assert events[-1]["response"]["output"] == [done]


def test_custom_tool_call_replay_pairs_tool_use_and_result(monkeypatch):
    chat_request = build(
        input=[
            {"role": "user", "content": "run"},
            {"type": "custom_tool_call", "call_id": "tooluse_x", "namespace": "functions", "name": "exec", "input": "1+1"},
            {"type": "custom_tool_call_output", "call_id": "tooluse_x", "output": "2"},
        ],
        tools=[namespace("functions", EXEC)],
    )

    args = bedrock_args(chat_request, monkeypatch)

    assert [tool["toolSpec"]["name"] for tool in args["toolConfig"]["tools"]] == ["functions__exec"]
    tool_use = args["messages"][1]["content"][0]["toolUse"]
    tool_result = args["messages"][2]["content"][0]["toolResult"]
    assert tool_use == {"toolUseId": "tooluse_x", "name": "functions__exec", "input": {"input": "1+1"}}
    assert tool_result == {"toolUseId": "tooluse_x", "content": [{"text": "2"}]}


def test_parallel_replayed_calls_share_one_assistant_turn(monkeypatch):
    chat_request = build(
        input=[
            {"role": "user", "content": "go"},
            {"type": "custom_tool_call", "call_id": "a", "name": "exec", "input": "1"},
            {"type": "function_call", "call_id": "b", "name": "shell", "arguments": "{}"},
            {"type": "custom_tool_call_output", "call_id": "a", "output": "one"},
            {"type": "function_call_output", "call_id": "b", "output": "two"},
        ],
        tools=[EXEC, function("shell")],
    )

    messages = bedrock_args(chat_request, monkeypatch)["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert [block["toolUse"]["toolUseId"] for block in messages[1]["content"]] == ["a", "b"]
    assert [block["toolResult"]["toolUseId"] for block in messages[2]["content"]] == ["a", "b"]


def test_custom_tool_choice_selects_the_custom_tool():
    chat_request = build(input="run", tools=[EXEC, function("shell")], tool_choice={"type": "custom", "name": "exec"})

    assert chat_request.tool_choice == {"function": {"name": "exec"}}

    with pytest.raises(HTTPException) as error:
        build(input="run", tools=[function("shell")], tool_choice={"type": "custom", "name": "shell"})
    assert error.value.status_code == 400
