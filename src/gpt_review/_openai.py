"""Open AI API Call Wrapper."""
import logging
import time

import openai
from openai.error import RateLimitError
import tiktoken

import gpt_review.constants as C
from gpt_review.context import _load_azure_openai_context

ENCODING = tiktoken.encoding_for_model("gpt-4")


def _encode_prompt(prompt, fast=False, large=False) -> "tuple[int, str]":
    """
    Encode the prompt to determine the number of tokens and
    truncate if necessary. This incurs a computational overhead,
    but it's better than randomly truncating and then having to retry.

    Args:
        prompt (str): The prompt to send to GPT-4.

    Returns:
        tuple[int, str]: The number of tokens and the encoded prompt respectively.
    """
    encoded = ENCODING.encode(prompt)

    if fast and len(encoded) > C.MAX_INPUT_TOKENS[C.GPT_TURBO_MODEL]:
        logging.warn(
            f"Fast model requested, but prompt is over {C.MAX_INPUT_TOKENS[C.GPT_TURBO_MODEL]} tokens. Truncating prompt."
        )
        encoded = encoded[: C.MAX_INPUT_TOKENS[C.GPT_TURBO_MODEL]]
    elif large and len(encoded) > C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]:
        logging.warn(
            f"Large model requested, but prompt is over {C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]} tokens. Truncating prompt."
        )
        encoded = encoded[: C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]]
    elif len(encoded) > C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]:
        logging.warn(f"Prompt is over {C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]} tokens. Truncating prompt.")
        encoded = encoded[: C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]]

    decoded = ENCODING.decode(encoded)
    return len(encoded), decoded


def _get_model(tokens: str, fast: bool = False, large: bool = False) -> str:
    """
    Get the OpenAI model based on the prompt length.
    - if large is available and prompt + max_tokens > 8000, use the 32k context model
    - otherwise use gpt-4
    - enable fast to use gpt-35-turbo for small prompts

    Args:
        tokens (int): The number of tokens in the prompt.
        fast (bool, optional): Whether to use the fast model. Defaults to False.
        large (bool, optional): Whether to use the large model. Defaults to False.

    Returns:
        str: The model to use.
    """
    context = _load_azure_openai_context()

    if large and tokens > C.MAX_INPUT_TOKENS[C.GPT_SMART_MODEL]:
        logging.info("Using GPT-4 32k context model")
        return context.large_llm_model_deployment_id
    elif fast:
        logging.info("Using GPT-3.5 turbo model")
        return context.turbo_llm_model_deployment_id
    else:
        if large:
            logging.warn(
                "GPT-4 32k context requested, but prompt is under {} tokens. Using GPT-4 8k context model instead".format(
                    C.MAX_INPUT_TOKENS[C.GPT_LARGE_MODEL]
                )
            )
        else:
            logging.info("Using GPT-4 8k context model")
        return context.smart_llm_model_deployment_id


def _call_gpt(
    prompt: str,
    temperature=0.10,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    retry=0,
    messages=None,
    fast: bool = False,
    large: bool = False,
) -> str:
    """
    Call GPT with the given prompt.

    Args:
        prompt (str): The prompt to send to GPT-4.
        temperature (float, optional): The temperature to use. Defaults to 0.10.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 500.
        top_p (float, optional): The top_p to use. Defaults to 1.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.5.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.0.
        retry (int, optional): The number of times to retry the request. Defaults to 0.
        messages (List[Dict[str, str]], optional): The messages to send to GPT-4. Defaults to None.
        fast (bool, optional): Whether to use the fast model. Defaults to False.
        large (bool, optional): Whether to use the large model. Defaults to False.

    Returns:
        str: The response from GPT.
    """
    tokens, prompt = _encode_prompt(prompt, fast=fast, large=large)
    model = _get_model(tokens, fast=fast, large=large)
    messages = messages or [{"role": "user", "content": prompt}]

    try:
        logging.info("Prompt sent to GPT: %s\n", prompt)
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return completion.choices[0].message.content  # type: ignore
    except RateLimitError as error:
        if retry < C.MAX_RETRIES:
            logging.warning("Call to GPT failed due to rate limit, retry attempt %s of %s", retry, C.MAX_RETRIES)

            wait_time = int(error.headers["Retry-After"]) if error.headers["Retry-After"] else retry * 10
            logging.warning("Waiting for %s seconds before retrying.", wait_time)

            time.sleep(wait_time)

            return _call_gpt(prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, retry + 1)
        raise RateLimitError("Retry limit exceeded") from error
