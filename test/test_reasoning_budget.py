"""Tests for the reasoning token budget derived from max_tokens."""

import pytest
from fastapi import HTTPException

from api.models.bedrock import MIN_BUDGET_TOKENS, BedrockModel


@pytest.fixture
def model():
    return BedrockModel()


def test_budget_follows_the_effort_ratio(model):
    assert model._calc_budget_tokens(10_000, "low") == 3_000
    assert model._calc_budget_tokens(10_000, "medium") == 6_000
    assert model._calc_budget_tokens(10_000, "high") == 9_999


def test_budget_is_raised_to_the_bedrock_minimum(model):
    # 30% of 2,000 is 600, which Bedrock rejects outright.
    assert model._calc_budget_tokens(2_000, "low") == MIN_BUDGET_TOKENS


def test_max_tokens_below_the_minimum_is_rejected(model):
    with pytest.raises(HTTPException) as exc:
        model._calc_budget_tokens(MIN_BUDGET_TOKENS, "low")

    assert exc.value.status_code == 400
    assert str(MIN_BUDGET_TOKENS) in exc.value.detail
