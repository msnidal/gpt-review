"""Tests for the Open AI Wrapper."""
import pytest
from openai.error import RateLimitError
from unittest.mock import patch

import gpt_review.constants as C
from gpt_review._openai import _call_gpt, _get_model
from gpt_review.context import _load_azure_openai_context

@patch('gpt_review._openai._count_tokens')
def test_get_model(mock_count_tokens) -> None:
    context = _load_azure_openai_context()

    # Mock count_tokens to return 3000
    mock_count_tokens.return_value = 3000

    # Test when large is True and tokens + max_tokens > 8000
    model = _get_model(prompt="prompt", max_tokens=6000, large=True)
    assert model == context.large_llm_model_deployment_id

    # Test when fast is True and tokens + max_tokens <= 4000
    model = _get_model(prompt="prompt", max_tokens=1000, fast=True)
    assert model == context.turbo_llm_model_deployment_id

    # Test when large is True but tokens + max_tokens <= 8000
    model = _get_model(prompt="prompt", max_tokens=1000, large=True)
    assert model == context.smart_llm_model_deployment_id

    # Test when fast is True but tokens + max_tokens > 4000
    model = _get_model(prompt="prompt", max_tokens=2000, fast=True)
    assert model == context.smart_llm_model_deployment_id

    # Test when neither large nor fast is True
    model = _get_model(prompt="prompt", max_tokens=1000)
    assert model == context.smart_llm_model_deployment_id

def rate_limit_test(monkeypatch):
    def mock_get_model(prompt: str, max_tokens: int, fast: bool = False, large: bool = False):
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
