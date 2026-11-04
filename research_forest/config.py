"""Runtime configuration, read from environment variables (prefix ``RF_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Literal

BackendName = Literal["openai", "codex", "claude-code"]


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"RF_{name}", default)


@dataclass
class Settings:
    # --- inference ---
    backend: BackendName = "openai"
    model: str | None = None  # backend default when None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    llm_concurrency: int = 4
    llm_timeout: float = 300.0
    llm_retries: int = 2
    cache_llm: bool = True

    # --- Semantic Scholar ---
    s2_api_key: str | None = None
    s2_rps: float = 1.0  # requests per second (unauthenticated pool is much lower)

    # --- pipeline ---
    target_papers: int = 60
    max_candidates: int = 250
    snowball_seeds: int = 15
    max_refs_per_paper: int = 15

    # --- sandboxed full-text worker ---
    fulltext_enabled: bool = True
    fulltext_max_mb: int = 25
    fulltext_timeout: float = 90.0
    fulltext_mem_mb: int = 1024
    fulltext_allow_private: bool = False  # SSRF guard; only disable for local testing

    # --- web server ---
    api_token: str | None = None  # generated at startup when unset
    max_jobs: int = 2  # concurrent build/refine jobs across all projects

    db_path: str = "research.forest.db"
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values: dict = {}
        for f in fields(cls):
            if f.name == "extra":
                continue
            raw = _env(f.name.upper())
            if raw is None:
                continue
            if f.type in ("int", int):
                values[f.name] = int(raw)
            elif f.type in ("float", float):
                values[f.name] = float(raw)
            elif f.type in ("bool", bool):
                values[f.name] = raw.lower() in ("1", "true", "yes", "on")
            else:
                values[f.name] = raw
        values.setdefault("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        values.setdefault("s2_api_key", os.environ.get("S2_API_KEY"))
        if base := os.environ.get("OPENAI_BASE_URL"):
            values.setdefault("openai_base_url", base)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
