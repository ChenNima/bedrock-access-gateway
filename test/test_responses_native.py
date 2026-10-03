"""Tests for the native Responses passthrough, with the upstream HTTP call mocked out."""

import json

import pytest
from botocore.credentials import Credentials
from fastapi.testclient import TestClient

from api import app as app_module
from api.models import responses_native as native_module
from api.models.responses_native import build_payload, is_native_responses_model
from api.schema import ResponsesRequest
from api.setting import API_ROUTE_PREFIX, AWS_REGION, parse_patterns

MODEL = "global.openai.gpt-6-astra"
AUTH = {"Authorization": "Bearer test-api-key"}
URL = f"{API_ROUTE_PREFIX}/responses"

NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "mcp__chorus",
    "description": "Chorus MCP server tools",
    "tools": [
        {
            "type": "function",
            "name": "chorus_checkin",
            "description": "Check in to Chorus",
            "strict": False,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    ],
}
FUNCTION_CALL = {
    "type": "function_call",
    "call_id": "call_1",
    "name": "chorus_checkin",
    "namespace": "mcp__chorus",
    "arguments": "{}",
}

CLIENT_TOOL_SEARCH = {
    "type": "tool_search",
    "execution": "client",
    "description": "Search deferred tools",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}
FUNCTION_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather",
    "strict": True,
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "defer_loading": True,
}
CUSTOM_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run code",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
}
HOSTED_TOOLS = [
    {"type": "web_search", "external_web_access": False},
    {"type": "web_search_preview"},
    {"type": "file_search", "vector_store_ids": ["vs_1"]},
    {"type": "mcp", "server_label": "x", "server_url": "https://example.com/mcp"},
    {"type": "code_interpreter", "container": {"type": "auto"}},
    {"type": "image_generation"},
    {"type": "local_shell"},
    {"type": "computer_use_preview", "display_width": 1, "display_height": 1, "environment": "linux"},
    {"type": "tool_search"},
    {"type": "tool_search", "execution": "server"},
    {"type": "some_future_hosted_tool"},
]


class FakeUpstream:
    def __init__(self, status_code=200, body=b"", chunks=None, content_type="application/json"):
        self.status_code = status_code
        self.content = body
        self.chunks = chunks or []
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def iter_content(self, chunk_size=None):
        assert chunk_size is None
        yield from self.chunks

    def close(self):
        self.closed = True


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def credentials(monkeypatch):
    creds = Credentials("AKIDEXAMPLE", "secret", "session-token")

    class FakeSession:
        def get_credentials(self):
            return creds

    monkeypatch.setattr(native_module.boto3, "Session", FakeSession)
    return creds


@pytest.fixture
def upstream(monkeypatch, credentials):
    """Capture the outgoing request; tests set calls["response"] to choose the reply."""
    calls = {"response": FakeUpstream(body=b'{"id": "resp_1", "object": "response"}'), "count": 0}

    def fake_post(url, data=None, headers=None, stream=False, timeout=None):
        calls.update(url=url, body=json.loads(data), headers=headers, stream=stream, timeout=timeout)
        calls["count"] += 1
        return calls["response"]

    monkeypatch.setattr(native_module.requests, "post", fake_post)
    return calls


@pytest.mark.parametrize(
    "model",
    ["global.openai.gpt-6-astra", "openai.gpt-5.2", "openai.gpt-5.4-codex", "us.openai.gpt-5.1", "OpenAI.GPT-6"],
)
def test_openai_gpt_models_route_native(model):
    assert is_native_responses_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "openai.gpt-oss-120b-1:0",
        "openai.gpt-oss-20b-1:0",
        "global.anthropic.claude-opus-5",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "us.amazon.nova-pro-v1:0",
    ],
)
def test_other_models_use_converse(model):
    assert not is_native_responses_model(model)


def test_custom_patterns():
    patterns = parse_patterns(" *openai.gpt-6* , *mistral* ")
    assert patterns == ("*openai.gpt-6*", "*mistral*")
    assert is_native_responses_model("mistral.large", patterns, ())
    assert is_native_responses_model("global.openai.gpt-6-astra", patterns, ())
    assert not is_native_responses_model("openai.gpt-5.2", patterns, ())
    assert not is_native_responses_model("global.openai.gpt-6-astra", patterns, parse_patterns("*astra*"))


def test_empty_patterns_disable_passthrough(monkeypatch):
    assert parse_patterns("") == ()
    assert not is_native_responses_model(MODEL, parse_patterns(""), ())
    monkeypatch.setattr(native_module, "RESPONSES_NATIVE_MODEL_PATTERNS", ())
    assert not is_native_responses_model(MODEL)


def test_payload_keeps_client_fields():
    request = ResponsesRequest.model_validate(
        {
            "model": MODEL,
            "input": [{"role": "user", "content": "Check in."}, FUNCTION_CALL],
            "tools": [NAMESPACE_TOOL, CLIENT_TOOL_SEARCH],
            "tool_choice": "required",
            "prompt_cache_key": "abc",
            "client_metadata": {"x": 1},
            "extra_body": {"gateway": "only"},
        }
    )
    payload = build_payload(request)

    assert payload["tools"] == [NAMESPACE_TOOL, CLIENT_TOOL_SEARCH]
    assert payload["tool_choice"] == "required"
    assert payload["input"][1] == FUNCTION_CALL
    assert payload["prompt_cache_key"] == "abc"
    assert payload["client_metadata"] == {"x": 1}
    assert "extra_body" not in payload
    assert payload["store"] is False
    # Fields the client left unset are not filled in with gateway defaults.
    assert "truncation" not in payload
    assert "parallel_tool_calls" not in payload


def test_payload_drops_hosted_tools(caplog):
    supported = [NAMESPACE_TOOL, FUNCTION_TOOL, CUSTOM_TOOL, CLIENT_TOOL_SEARCH]
    tools = [HOSTED_TOOLS[0], *supported[:2], *HOSTED_TOOLS[1:], *supported[2:]]
    request = ResponsesRequest.model_validate({"model": MODEL, "input": "Hi", "tools": tools})

    with caplog.at_level("WARNING", logger=native_module.__name__):
        payload = build_payload(request)

    # Supported tools are forwarded in order and byte for byte.
    assert json.dumps(payload["tools"]) == json.dumps(supported)
    messages = " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    for tool_type in ("web_search", "web_search_preview", "file_search", "mcp", "some_future_hosted_tool"):
        assert f"type {tool_type}," in messages
    assert "tool_search with execution server" in messages
    assert "tool_search with execution None" in messages


def test_payload_without_hosted_tools_is_unchanged():
    tools = [NAMESPACE_TOOL, CUSTOM_TOOL, CLIENT_TOOL_SEARCH]
    request = ResponsesRequest.model_validate(
        {"model": MODEL, "input": "Hi", "tools": tools, "tool_choice": {"type": "function", "name": "get_weather"}}
    )
    payload = build_payload(request)
    assert payload["tools"] == tools
    assert payload["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_payload_drops_tools_field_when_only_hosted_tools():
    request = ResponsesRequest.model_validate({"model": MODEL, "input": "Hi", "tools": [{"type": "web_search"}]})
    assert "tools" not in build_payload(request)


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "web_search"},
        {"type": "web_search_preview"},
        {"type": "file_search"},
        {"type": "mcp", "server_label": "x"},
        {"type": "tool_search"},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search"}]},
    ],
)
def test_tool_choice_naming_dropped_tool_becomes_auto(tool_choice, caplog):
    tools = [NAMESPACE_TOOL, {"type": "web_search"}, {"type": "tool_search", "execution": "server"}]
    request = ResponsesRequest.model_validate(
        {"model": MODEL, "input": "Hi", "tools": tools, "tool_choice": tool_choice}
    )
    with caplog.at_level("WARNING", logger=native_module.__name__):
        payload = build_payload(request)
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [NAMESPACE_TOOL]
    assert any("falling back to auto" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        "none",
        {"type": "function", "name": "chorus_checkin", "namespace": "mcp__chorus"},
        {"type": "custom", "name": "exec"},
        {"type": "tool_search"},
        {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "chorus_checkin"}]},
    ],
)
def test_tool_choice_for_kept_tools_is_unchanged(tool_choice):
    tools = [NAMESPACE_TOOL, CUSTOM_TOOL, CLIENT_TOOL_SEARCH, {"type": "web_search"}]
    request = ResponsesRequest.model_validate(
        {"model": MODEL, "input": "Hi", "tools": tools, "tool_choice": tool_choice}
    )
    assert build_payload(request)["tool_choice"] == tool_choice


def test_forwarded_request_has_no_hosted_tools(client, upstream):
    response = client.post(
        URL,
        headers=AUTH,
        json={
            "model": MODEL,
            "input": "Hi",
            "tools": [NAMESPACE_TOOL, {"type": "web_search", "external_web_access": True}],
        },
    )
    assert response.status_code == 200
    assert upstream["body"]["tools"] == [NAMESPACE_TOOL]


def test_payload_keeps_explicit_store():
    assert build_payload(ResponsesRequest(model=MODEL, input="Hi", store=True))["store"] is True
    assert build_payload(ResponsesRequest(model=MODEL, input="Hi", store=False))["store"] is False


def test_request_is_signed_with_gateway_credentials(client, upstream):
    response = client.post(URL, headers=AUTH, json={"model": MODEL, "input": "Hi", "tools": [NAMESPACE_TOOL]})

    assert response.status_code == 200
    assert upstream["url"] == f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1/responses"
    headers = {k.lower(): v for k, v in upstream["headers"].items()}
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert f"/{AWS_REGION}/bedrock/aws4_request" in headers["authorization"]
    assert "test-api-key" not in headers["authorization"]
    assert "x-amz-date" in headers
    assert headers["x-amz-security-token"] == "session-token"
    assert headers["content-type"] == "application/json"
    assert upstream["body"]["tools"] == [NAMESPACE_TOOL]
    assert upstream["body"]["model"] == MODEL
    assert upstream["stream"] is False


def test_url_override():
    assert native_module.NativeResponsesProxy(url="https://example.test/v1/responses").url == (
        "https://example.test/v1/responses"
    )


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_requires_an_api_key(client, upstream, headers):
    response = client.post(URL, headers=headers, json={"model": MODEL, "input": "Hi"})

    assert response.status_code in (401, 403)
    assert upstream["count"] == 0


def test_non_streaming_json_returned_as_is(client, upstream):
    body = {"id": "resp_1", "object": "response", "output": [dict(FUNCTION_CALL, id="fc_1", status="completed")]}
    upstream["response"] = FakeUpstream(body=json.dumps(body).encode())

    response = client.post(URL, headers=AUTH, json={"model": MODEL, "input": "Hi"})

    assert response.status_code == 200
    assert response.json() == body
    assert upstream["response"].closed


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_passes_through(client, upstream, stream):
    error = b'{"error": {"message": "not authorized", "type": "access_denied"}}'
    upstream["response"] = FakeUpstream(status_code=403, body=error)

    response = client.post(URL, headers=AUTH, json={"model": MODEL, "input": "Hi", "stream": stream})

    assert response.status_code == 403
    assert response.content == error
    assert response.headers["content-type"].startswith("application/json")
    assert upstream["response"].closed


def test_streaming_relays_bytes_unchanged(client, upstream):
    chunks = [
        b'event: response.created\ndata: {"type":"response.created"}\n\n',
        b'event: response.output_item.done\ndata: {"type":"response.output_item.done",',
        b'"item":{"type":"function_call","namespace":"mcp__chorus"}}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed"}\n\n',
    ]
    upstream["response"] = FakeUpstream(chunks=chunks, content_type="text/event-stream")

    response = client.post(URL, headers=AUTH, json={"model": MODEL, "input": "Hi", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"".join(chunks)
    assert upstream["stream"] is True
    assert {k.lower(): v for k, v in upstream["headers"].items()}["accept"] == "text/event-stream"
    assert upstream["response"].closed


def test_converse_models_are_not_forwarded(client, upstream, monkeypatch):
    seen = {}

    class FakeModel:
        def build_chat_request(self, request):
            seen["model"] = request.model
            raise native_module.HTTPException(status_code=400, detail="stop here")

    monkeypatch.setattr("api.routers.responses.BedrockResponsesModel", FakeModel)
    response = client.post(URL, headers=AUTH, json={"model": "openai.gpt-oss-120b-1:0", "input": "Hi"})

    assert response.status_code == 400
    assert seen["model"] == "openai.gpt-oss-120b-1:0"
    assert upstream["count"] == 0
