"""LLM backends: OpenAI-compatible HTTP, Codex CLI and Claude Code CLI."""

from __future__ import annotations

from ..config import Settings
from .base import LLM, Backend, JsonCache, Reply, TransientLLMError, strict_schema
from .calllog import CallLog


def make_backend(settings: Settings) -> Backend:
    if settings.backend == "openai":
        from .openai_compat import OpenAIBackend
        return OpenAIBackend(settings.openai_base_url, settings.openai_api_key, settings.model, settings.llm_timeout)
    if settings.backend == "claude-code":
        from .cli import ClaudeCodeBackend
        return ClaudeCodeBackend(settings.model, settings.llm_timeout, effort=settings.llm_effort,
                                 user_config=settings.cli_user_config)
    if settings.backend == "codex":
        from .cli import CodexBackend
        return CodexBackend(settings.model, settings.llm_timeout, effort=settings.llm_effort,
                            user_config=settings.cli_user_config)
    raise ValueError(f"unknown backend {settings.backend!r}")


def make_llm(settings: Settings, backend: Backend | None = None) -> LLM:
    cache = JsonCache(settings.db_path + ".llmcache") if settings.cache_llm else None
    call_log = CallLog(settings.llm_log_file, settings.llm_log_max_mb * 2**20) if settings.llm_log_enabled else None
    return LLM(backend or make_backend(settings), concurrency=settings.llm_concurrency,
               retries=settings.llm_retries, cache=cache, call_log=call_log)


__all__ = ["CallLog", "LLM", "Backend", "JsonCache", "Reply", "TransientLLMError", "make_backend", "make_llm", "strict_schema"]
