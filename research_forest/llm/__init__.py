"""LLM backends: OpenAI-compatible HTTP, Codex CLI and Claude Code CLI."""

from __future__ import annotations

from ..config import Settings
from .base import LLM, Backend, JsonCache, TransientLLMError, strict_schema


def make_backend(settings: Settings) -> Backend:
    if settings.backend == "openai":
        from .openai_compat import OpenAIBackend
        return OpenAIBackend(settings.openai_base_url, settings.openai_api_key, settings.model, settings.llm_timeout)
    if settings.backend == "claude-code":
        from .cli import ClaudeCodeBackend
        return ClaudeCodeBackend(settings.model, settings.llm_timeout)
    if settings.backend == "codex":
        from .cli import CodexBackend
        return CodexBackend(settings.model, settings.llm_timeout)
    raise ValueError(f"unknown backend {settings.backend!r}")


def make_llm(settings: Settings, backend: Backend | None = None) -> LLM:
    cache = JsonCache(settings.db_path + ".llmcache") if settings.cache_llm else None
    return LLM(backend or make_backend(settings), concurrency=settings.llm_concurrency,
               retries=settings.llm_retries, cache=cache)


__all__ = ["LLM", "Backend", "JsonCache", "TransientLLMError", "make_backend", "make_llm", "strict_schema"]
