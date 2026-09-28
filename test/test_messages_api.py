"""End-to-end tests for the /messages routes, with Bedrock's Converse API stubbed out."""

import json

import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.models import bedrock as bedrock_module
from api.setting import API_ROUTE_PREFIX

MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
URL = f"{API_ROUTE_PREFIX}/messages"
BODY = {"model": "claude-sonnet-4-5", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def supported_model(monkeypatch):
    monkeypatch.setattr(bedrock_module, "bedrock_model_list", {MODEL: {"modalities": ["TEXT", "IMAGE"]}})


@pytest.fixture
def converse(monkeypatch):
    calls = {}

    def fake_converse(**kwargs):
        calls.update(kwargs)
        return {
            "output": {"message": {"content": [{"text": "Hello there"}]}},
            "usage": {"inputTokens": 8, "outputTokens": 2, "totalTokens": 10},
            "stopReason": "end_turn",
        }

    monkeypatch.setattr(bedrock_module.bedrock_runtime, "converse", fake_converse)
    return calls


@pytest.mark.parametrize(
    "headers",
    [{"x-api-key": "test-api-key"}, {"Authorization": "Bearer test-api-key"}],
)
def test_accepts_both_anthropic_auth_headers(client, converse, headers):
    assert client.post(URL, headers=headers, json=BODY).status_code == 200


def test_rejects_a_bad_key_in_anthropic_format(client):
    response = client.post(URL, headers={"x-api-key": "wrong"}, json=BODY)

    assert response.status_code == 401
    assert response.json() == {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}}


def test_non_streaming_response(client, converse):
    # Claude Code appends ?beta=true to the url.
    response = client.post(f"{URL}?beta=true", headers={"x-api-key": "test-api-key"}, json=BODY)

    body = response.json()
    assert body["type"] == "message"
    assert body["model"] == "claude-sonnet-4-5"
    assert body["content"] == [{"type": "text", "text": "Hello there"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 8
    assert converse["modelId"] == MODEL


def test_errors_use_the_anthropic_format(client):
    response = client.post(URL, headers={"x-api-key": "test-api-key"}, json={**BODY, "model": "no.such-model"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_invalid_body_uses_the_anthropic_format(client):
    response = client.post(URL, headers={"x-api-key": "test-api-key"}, json={"model": MODEL})

    assert response.status_code == 400
    assert response.json()["type"] == "error"


def test_context_overflow_is_reported_as_prompt_too_long(client, monkeypatch):
    def too_long(**kwargs):
        raise bedrock_module.bedrock_runtime.exceptions.ValidationException(
            {"Error": {"Code": "ValidationException", "Message": "Input is too long for requested model."}},
            "Converse",
        )

    monkeypatch.setattr(bedrock_module.bedrock_runtime, "converse", too_long)
    response = client.post(URL, headers={"x-api-key": "test-api-key"}, json=BODY)

    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("prompt is too long")


def test_other_routes_keep_their_error_format(client):
    response = client.post(f"{API_ROUTE_PREFIX}/chat/completions", json={})

    assert response.status_code in (401, 403)
    assert "detail" in response.json()


def test_streaming_response(client, monkeypatch):
    chunks = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 8, "outputTokens": 2, "totalTokens": 10}}},
    ]
    monkeypatch.setattr(bedrock_module.bedrock_runtime, "converse_stream", lambda **kwargs: {"stream": chunks})

    with client.stream("POST", URL, headers={"x-api-key": "test-api-key"}, json={**BODY, "stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())

    types = [json.loads(line[len("data: ") :])["type"] for line in payload.split("\n") if line.startswith("data: ")]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_throttled_stream_returns_429_before_streaming(client, monkeypatch):
    def throttled(**kwargs):
        raise bedrock_module.bedrock_runtime.exceptions.ThrottlingException(
            {"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}},
            "ConverseStream",
        )

    monkeypatch.setattr(bedrock_module.bedrock_runtime, "converse_stream", throttled)
    response = client.post(URL, headers={"x-api-key": "test-api-key"}, json={**BODY, "stream": True})

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_count_tokens(client, monkeypatch):
    calls = {}

    def fake_count_tokens(**kwargs):
        calls.update(kwargs)
        return {"inputTokens": 42}

    monkeypatch.setattr(bedrock_module.bedrock_runtime, "count_tokens", fake_count_tokens)
    response = client.post(
        f"{URL}/count_tokens",
        headers={"x-api-key": "test-api-key"},
        json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.json() == {"input_tokens": 42}
    assert calls["input"]["converse"]["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
