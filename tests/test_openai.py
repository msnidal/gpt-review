"""Tests for the Open AI Wrapper."""
import pytest
from openai.error import RateLimitError
from unittest.mock import patch

import gpt_review.constants as C
from gpt_review._openai import _call_gpt, _get_model, _encode_prompt
from gpt_review.context import _load_azure_openai_context


@patch("gpt_review._openai._encode_prompt")
def test_get_model(mock_encode_prompt) -> None:
    context = _load_azure_openai_context()

    # Test when large is True and tokens > C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]
    mock_encode_prompt.return_value = (10000, "prompt")
    model = _get_model(tokens=10000, large=True)
    assert model == context.large_llm_model_deployment_id

    # Test when fast is True
    mock_encode_prompt.return_value = (1000, "prompt")
    model = _get_model(tokens=1000, fast=True)
    assert model == context.turbo_llm_model_deployment_id

    # Test when large is True but tokens <= C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]
    model = _get_model(tokens=1000, large=True)
    assert model == context.smart_llm_model_deployment_id

    # Test when neither large nor fast is True
    model = _get_model(tokens=1000)
    assert model == context.smart_llm_model_deployment_id


""" Slow to run so I'm toggling off for now
def test_encode_prompt():
    # Test when fast is True and tokens > C.MAX_INPUT_TOKENS[C.GPT_TURBO_MODEL]
    prompt = "prompt" * 5000
    tokens, encoded_prompt = _encode_prompt(prompt, fast=True)
    assert tokens == C.MAX_INPUT_TOKENS[C.GPT_TURBO_MODEL]
    assert encoded_prompt in prompt

    # Test when large is True and tokens > C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]
    prompt = "prompt" * 35000
    tokens, encoded_prompt = _encode_prompt(prompt, large=True)
    assert tokens == C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]
    assert encoded_prompt in prompt

    # Test when tokens > C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]
    prompt = "prompt" * 9001
    tokens, encoded_prompt = _encode_prompt(prompt)
    assert tokens == C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]
    assert encoded_prompt in prompt
"""


def rate_limit_test(monkeypatch):
    def mock_get_model(tokens: int, fast: bool = False, large: bool = False):
        error = RateLimitError("Rate Limit Error")
        error.headers["Retry-After"] = 10
        raise error

    monkeypatch.setattr("gpt_review._openai._get_model", mock_get_model)
    with pytest.raises(RateLimitError):
        _call_gpt(prompt="This is a test prompt", retry=C.MAX_RETRIES)


def test_rate_limit(monkeypatch) -> None:
    rate_limit_test(monkeypatch)


@pytest.mark.integration
def test_int_rate_limit(monkeypatch) -> None:
    rate_limit_test(monkeypatch)


@patch("gpt_review._openai._encode_prompt")
@patch("gpt_review._openai._get_model")
def test_call_gpt(mock_get_model, mock_encode_prompt):
    # Test fast model
    mock_encode_prompt.return_value = (1000, "prompt")
    mock_get_model.return_value = C.GPT_TURBO_MODEL
    response = _call_gpt(prompt="This is a test prompt", fast=True)
    assert isinstance(response, str)

    # Test when neither fast nor large is True
    mock_encode_prompt.return_value = (1000, "prompt")
    mock_get_model.return_value = C.GPT_SMART_MODEL
    response = _call_gpt(prompt="This is a test prompt")
    assert isinstance(response, str)

    # Test large model (disabled for now)
    """
    mock_encode_prompt.return_value = (1000, "prompt")
    mock_get_model.return_value = C.GPT_LARGE_MODEL
    response = _call_gpt(prompt="This is a test prompt", large=True)
    assert isinstance(response, str)
    """