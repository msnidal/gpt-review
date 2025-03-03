"""Contains constants for minimum and maximum values of various parameters used in GPT Review."""
import os
import sys

MAX_TOKENS_DEFAULT = 100
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = sys.maxsize

TEMPERATURE_DEFAULT = 0.7
TEMPERATURE_MIN = 0
TEMPERATURE_MAX = 1

TOP_P_DEFAULT = 0.5
TOP_P_MIN = 0
TOP_P_MAX = 1

FREQUENCY_PENALTY_DEFAULT = 0.5
FREQUENCY_PENALTY_MIN = 0
FREQUENCY_PENALTY_MAX = 2

PRESENCE_PENALTY_DEFAULT = 0
PRESENCE_PENALTY_MIN = 0
PRESENCE_PENALTY_MAX = 2

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 15))

AZURE_API_TYPE = "azure"
AZURE_API_VERSION = "2023-03-15-preview"
AZURE_CONFIG_FILE = "azure.yaml"
AZURE_KEY_VAULT = "https://dciborow-openai.vault.azure.net/"

GPT_TURBO_MODEL = "gpt-3.5-turbo"
GPT_SMART_MODEL = "gpt-4"
GPT_LARGE_MODEL = "gpt-4-32k"
GPT_EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_INPUT_TOKENS = {
    GPT_TURBO_MODEL: 4000,
    GPT_SMART_MODEL: 8100,
    GPT_LARGE_MODEL: 32000,
}

BUG_PROMPT_YAML = "prompt_bug.yaml"
COVERAGE_PROMPT_YAML = "prompt_coverage.yaml"
SUMMARY_PROMPT_YAML = "prompt_summary.yaml"
