"""Tests for the Open AI Wrapper."""
import pytest
from openai.error import RateLimitError
from unittest.mock import patch

import gpt_review.constants as C
from gpt_review._openai import _call_gpt, _get_model, _encode_prompt
from gpt_review.context import _load_azure_openai_context

@patch('gpt_review._openai._encode_prompt')
def test_get_model(mock_encode_prompt) -> None:
    context = _load_azure_openai_context()

    # Test when large is True and tokens > C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]
    mock_encode_prompt.return_value = (10000, "prompt")
    model = _get_model(prompt="prompt", large=True)
    assert model == context.large_llm_model_deployment_id

    # Test when fast is True
    mock_encode_prompt.return_value = (1000, "prompt")
    model = _get_model(prompt="prompt", fast=True)
    assert model == context.turbo_llm_model_deployment_id

    # Test when large is True but tokens <= C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]
    model = _get_model(prompt="prompt", large=True)
    assert model == context.smart_llm_model_deployment_id

    # Test when neither large nor fast is True
    model = _get_model(prompt="prompt")
    assert model == context.smart_llm_model_deployment_id

def rate_limit_test(monkeypatch):
    def mock_get_model(prompt: str, fast: bool = False, large: bool = False):
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
