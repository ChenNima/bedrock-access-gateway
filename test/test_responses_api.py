"""End-to-end tests for the /responses route, with Bedrock's Converse API stubbed out."""

import json

import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.models import bedrock as bedrock_module
from api.setting import API_ROUTE_PREFIX

MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
AUTH = {"Authorization": "Bearer test-api-key"}
URL = f"{API_ROUTE_PREFIX}/responses"


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def supported_model(monkeypatch):
    """Make MODEL pass validate() without listing models from Bedrock."""
    monkeypatch.setattr(bedrock_module, "bedrock_model_list", {MODEL: {"modalities": ["TEXT", "IMAGE"]}})


@pytest.fixture
def converse(monkeypatch):
    """Capture the Converse arguments and reply with a canned response."""
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


def test_requires_an_api_key(client):
    assert client.post(URL, json={"model": MODEL, "input": "Hi"}).status_code == 401


def test_non_streaming_response(client, supported_model, converse):
    response = client.post(
        URL,
        headers=AUTH,
        json={"model": MODEL, "instructions": "Be brief.", "input": "Hi"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["created_at"] > 0
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["text"] == "Hello there"
    assert body["usage"]["input_tokens"] == 8
    # error and incomplete_details are spelled out as nulls, as OpenAI does.
    assert body["error"] is None
    assert body["incomplete_details"] is None

    assert converse["modelId"] == MODEL
    assert converse["system"] == [{"text": "Be brief."}]
    assert converse["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]


def test_gpt_model_names_fall_back_to_the_default_model(client, supported_model, converse, monkeypatch):
    monkeypatch.setattr("api.routers.responses.DEFAULT_MODEL", MODEL)

    response = client.post(URL, headers=AUTH, json={"model": "gpt-5-codex", "input": "Hi"})

    assert response.status_code == 200
    assert response.json()["model"] == MODEL
    assert converse["modelId"] == MODEL


def test_unsupported_model_is_rejected(client, supported_model):
    response = client.post(URL, headers=AUTH, json={"model": "no.such-model", "input": "Hi"})

    assert response.status_code == 400


def test_streaming_response(client, supported_model, monkeypatch):
    chunks = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"outputTokens": 2, "totalTokens": 10}}},
    ]
    monkeypatch.setattr(
        bedrock_module.bedrock_runtime,
        "converse_stream",
        lambda **kwargs: {"stream": chunks},
    )

    with client.stream(
        "POST", URL, headers=AUTH, json={"model": MODEL, "input": "Hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())

    types = [
        json.loads(line[len("data: ") :])["type"]
        for line in payload.split("\n")
        if line.startswith("data: ")
    ]
    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"
