"""scoring/llm/constants.py — Constants for LLM scoring."""

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_PATH = "/api/chat"
DEFAULT_MODEL = "qwen2.5:14b"
FALLBACK_MODEL = "mistral-nemo:12b"

SCORE_MIN = 0
SCORE_MAX = 10
PASS_THRESHOLD = 5  # LLM score >= 5 → passed
REQUEST_TIMEOUT = 30.0  # seconds per attempt
