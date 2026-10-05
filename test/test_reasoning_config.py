"""Tests for how a reasoning effort becomes a Claude thinking configuration on Converse."""

import pytest
from fastapi import HTTPException

import api.models.bedrock as bedrock
from api.models.bedrock import BedrockModel
from api.models.responses import BedrockResponsesModel
from api.schema import ChatRequest, ResponsesRequest, UserMessage

ADAPTIVE_MODELS = [
    "anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5-5",
    "global.anthropic.claude-sonnet-5-5",
    "anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",
]

LEGACY_MODELS = [
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
]


@pytest.fixture
def model():
    return BedrockModel()


def parse(model, model_id, **kwargs):
    request = ChatRequest(model=model_id, messages=[UserMessage(content="Hi")], **kwargs)
    return model._parse_request(request)


@pytest.mark.parametrize("model_id", ADAPTIVE_MODELS)
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_current_claude_models_get_adaptive_thinking(model, model_id, effort):
    args = parse(model, model_id, reasoning_effort=effort)

    assert args["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }
    # Adaptive thinking has no budget, so max_tokens is optional.
    assert "maxTokens" not in args["inferenceConfig"]


def test_adaptive_thinking_keeps_max_tokens_when_given(model):
    args = parse(model, "global.anthropic.claude-opus-5", reasoning_effort="high", max_tokens=4096)

    assert args["inferenceConfig"]["maxTokens"] == 4096
    assert "reasoning_config" not in args["additionalModelRequestFields"]


def test_profile_resolving_to_a_current_model_gets_adaptive_thinking(model, monkeypatch):
    arn = "arn:aws:bedrock:ap-northeast-1:123456789012:application-inference-profile/abc123"
    monkeypatch.setitem(bedrock.profile_metadata, arn, {"underlying_model_id": "anthropic.claude-sonnet-5-5"})

    args = parse(model, arn, reasoning_effort="medium")

    assert args["additionalModelRequestFields"]["thinking"]["type"] == "adaptive"
    assert args["additionalModelRequestFields"]["output_config"] == {"effort": "medium"}


def test_profile_resolving_to_a_legacy_model_gets_a_budget(model, monkeypatch):
    arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/def456"
    monkeypatch.setitem(
        bedrock.profile_metadata, arn, {"underlying_model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0"}
    )

    args = parse(model, arn, reasoning_effort="medium", max_tokens=10_000)

    assert args["additionalModelRequestFields"]["reasoning_config"]["type"] == "enabled"


@pytest.mark.parametrize("model_id", LEGACY_MODELS)
def test_legacy_claude_models_keep_the_token_budget(model, model_id):
    args = parse(model, model_id, reasoning_effort="medium", max_tokens=10_000)

    assert args["additionalModelRequestFields"] == {
        "reasoning_config": {"type": "enabled", "budget_tokens": 6_000},
    }


@pytest.mark.parametrize("model_id", LEGACY_MODELS)
def test_legacy_claude_models_still_require_max_tokens(model, model_id):
    with pytest.raises(HTTPException) as exc:
        parse(model, model_id, reasoning_effort="low")

    assert exc.value.status_code == 400


def test_budget_model_patterns_override_moves_models_between_modes(model, monkeypatch):
    # An empty list sends every Claude model down the adaptive path.
    monkeypatch.setattr(bedrock, "BUDGET_THINKING_MODEL_PATTERNS", ())
    args = parse(model, "anthropic.claude-sonnet-4-5-20250929-v1:0", reasoning_effort="low")
    assert args["additionalModelRequestFields"]["thinking"]["type"] == "adaptive"

    # Matching ignores case, and a listed model goes back to a budget.
    monkeypatch.setattr(bedrock, "BUDGET_THINKING_MODEL_PATTERNS", ("*ANTHROPIC.CLAUDE-OPUS-5*",))
    args = parse(model, "global.anthropic.claude-opus-5", reasoning_effort="low", max_tokens=10_000)
    assert args["additionalModelRequestFields"]["reasoning_config"]["type"] == "enabled"


@pytest.mark.parametrize(
    "model_id", ["global.anthropic.claude-opus-5", "anthropic.claude-sonnet-5-5", "anthropic.claude-opus-4-8"]
)
@pytest.mark.parametrize("effort", [None, "high"])
def test_sampling_params_are_dropped_for_models_that_reject_them(model, model_id, effort):
    args = parse(model, model_id, temperature=0.2, top_p=0.9, reasoning_effort=effort)

    assert "temperature" not in args["inferenceConfig"]
    assert "topP" not in args["inferenceConfig"]


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
    ],
)
def test_sampling_params_are_kept_without_reasoning(model, model_id):
    args = parse(model, model_id, temperature=0.2, top_p=0.9)

    assert args["inferenceConfig"]["temperature"] == 0.2
    # Sonnet 4.6 takes only one of the two, which the conflict rule handles.
    if "sonnet-4-6" in model_id:
        assert "topP" not in args["inferenceConfig"]
    else:
        assert args["inferenceConfig"]["topP"] == 0.9


@pytest.mark.parametrize("model_id", ["anthropic.claude-opus-4-6-v1", "anthropic.claude-sonnet-4-6"])
def test_four_six_models_drop_top_p_when_thinking(model, model_id):
    args = parse(model, model_id, top_p=0.9, temperature=1.0, reasoning_effort="low")

    assert "topP" not in args["inferenceConfig"]
    assert args["inferenceConfig"]["temperature"] == 1.0
    assert args["additionalModelRequestFields"]["thinking"]["type"] == "adaptive"


def test_temperature_top_p_conflict_rule_is_unchanged(model):
    args = parse(model, "anthropic.claude-sonnet-4-5-20250929-v1:0", temperature=0.2, top_p=0.9)

    assert args["inferenceConfig"] == {"temperature": 0.2}


def test_non_claude_models_keep_sampling_params(model):
    args = parse(model, "meta.llama3-70b-instruct-v1:0", temperature=0.2, top_p=0.9)

    assert args["inferenceConfig"] == {"temperature": 0.2, "topP": 0.9}


class FakeChatModel:
    def validate(self, chat_request):
        pass


def responses_args(model, effort):
    request = ResponsesRequest(model="global.anthropic.claude-opus-5", input="Hi", reasoning={"effort": effort})
    chat_request = BedrockResponsesModel(chat_model=FakeChatModel()).build_chat_request(request)
    return model._parse_request(chat_request)


def test_responses_minimal_effort_becomes_low(model):
    args = responses_args(model, "minimal")

    assert args["additionalModelRequestFields"]["output_config"] == {"effort": "low"}
    assert args["additionalModelRequestFields"]["thinking"]["type"] == "adaptive"


def test_responses_none_effort_sends_no_thinking(model):
    args = responses_args(model, "none")

    assert "additionalModelRequestFields" not in args


def test_reasoning_text_reaches_the_chat_response(model):
    response = model._create_response(
        model="global.anthropic.claude-opus-5",
        message_id="chatcmpl-1",
        content=[
            {"reasoningContent": {"reasoningText": {"text": "Let me think.", "signature": "sig"}}},
            {"text": "Hello"},
        ],
        finish_reason="end_turn",
        input_tokens=5,
        output_tokens=7,
        total_tokens=12,
    )

    content = response.choices[0].message.content
    assert content == "<think>Let me think.</think>Hello"
    assert response.usage.completion_tokens_details.reasoning_tokens > 0


def test_reasoning_text_reaches_the_chat_stream(model):
    chunks = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Let me "}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think."}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "Hello"}}},
    ]

    # chat_stream resets this before each stream.
    model.think_emitted = False
    parts = []
    for chunk in chunks:
        response = model._create_response_stream("global.anthropic.claude-opus-5", "chatcmpl-1", chunk)
        if response:
            parts.append(response.choices[0].delta.content)

    assert "".join(parts) == "<think>Let me think.</think>Hello"
