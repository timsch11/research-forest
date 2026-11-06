"""Runtime configuration.

* Settings come from a TOML config file (``research-forest.toml`` in the working directory, or
  ``--config PATH``). See ``research-forest.example.toml`` for every option.
* Secrets come only from the environment: a ``.env`` file, overridden by variables set in the
  terminal. They are rejected in the config file so it can be committed safely.

Precedence: defaults < config file < CLI flags; secrets: ``.env`` < terminal.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

BackendName = Literal["openai", "codex", "claude-code"]
BACKENDS = ("openai", "codex", "claude-code")
EFFORTS = ("", "low", "medium", "high", "xhigh", "max")  # "" = the CLI's own default
DEFAULT_CONFIG = "research-forest.toml"

# Secret field -> environment variables, in priority order.
SECRETS: dict[str, tuple[str, ...]] = {
    "openai_api_key": ("OPENAI_API_KEY", "RF_OPENAI_API_KEY"),
    "s2_api_key": ("S2_API_KEY", "RF_S2_API_KEY"),
    "api_token": ("RF_API_TOKEN",),
}

# (config section, key) -> (Settings field, accepted type)
CONFIG_KEYS: dict[tuple[str, str], tuple[str, type]] = {
    ("llm", "backend"): ("backend", str),
    ("llm", "model"): ("model", str),
    ("llm", "base_url"): ("openai_base_url", str),
    ("llm", "concurrency"): ("llm_concurrency", int),
    ("llm", "timeout"): ("llm_timeout", float),
    ("llm", "retries"): ("llm_retries", int),
    ("llm", "cache"): ("cache_llm", bool),
    ("llm", "effort"): ("llm_effort", str),
    ("llm", "cli_user_config"): ("cli_user_config", bool),
    ("semantic_scholar", "requests_per_second"): ("s2_rps", float),
    ("pipeline", "target_papers"): ("target_papers", int),
    ("pipeline", "max_candidates"): ("max_candidates", int),
    ("pipeline", "max_refs_per_paper"): ("max_refs_per_paper", int),
    ("pipeline", "baseline_papers"): ("baseline_papers", int),
    ("pipeline", "followup_papers"): ("followup_papers", int),
    ("pipeline", "followup_seeds"): ("followup_seeds", int),
    ("pipeline", "followup_years"): ("followup_years", int),
    ("pipeline", "update_papers"): ("update_papers", int),
    ("fulltext", "enabled"): ("fulltext_enabled", bool),
    ("fulltext", "max_mb"): ("fulltext_max_mb", int),
    ("fulltext", "timeout"): ("fulltext_timeout", float),
    ("fulltext", "memory_mb"): ("fulltext_mem_mb", int),
    ("fulltext", "allow_private"): ("fulltext_allow_private", bool),
    ("fulltext", "prefer_html"): ("fulltext_prefer_html", bool),
    ("fulltext", "single_pass"): ("fulltext_single_pass", bool),
    ("fulltext", "always_fill_datasets"): ("fulltext_always_fill_datasets", bool),
    ("fulltext", "limitation_sections"): ("fulltext_limitation_sections", bool),
    ("server", "max_jobs"): ("max_jobs", int),
    ("storage", "db_path"): ("db_path", str),
    ("logging", "llm_calls"): ("llm_log_enabled", bool),
    ("logging", "llm_log_file"): ("llm_log_file", str),
}


class ConfigError(ValueError):
    """Invalid configuration file or value."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a .env file; variables already set in the environment win."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key.strip(), value)


def read_config(path: str | Path) -> dict[str, Any]:
    """Parse and validate a TOML config file into Settings field values."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from None
    values: dict[str, Any] = {}
    for section, table in data.items():
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: '{section}' must be a [section]")
        for key, value in table.items():
            if key in SECRETS or any(key.lower() == env.lower() for envs in SECRETS.values() for env in envs) \
                    or "key" in key.lower() or "token" in key.lower():
                raise ConfigError(f"{path}: [{section}] {key}: secrets belong in .env or the environment, "
                                  "not in the config file")
            if (section, key) not in CONFIG_KEYS:
                known = ", ".join(f"[{s}] {k}" for s, k in CONFIG_KEYS)
                raise ConfigError(f"{path}: unknown setting [{section}] {key}. Known settings: {known}")
            field_name, kind = CONFIG_KEYS[section, key]
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
                raise ConfigError(f"{path}: [{section}] {key} must be {kind.__name__}, got {value!r}")
            values[field_name] = value
    if values.get("backend", "openai") not in BACKENDS:
        raise ConfigError(f"{path}: [llm] backend must be one of {', '.join(BACKENDS)}")
    if values.get("llm_effort", "low") not in EFFORTS:
        raise ConfigError(f"{path}: [llm] effort must be one of {', '.join(repr(e) for e in EFFORTS)}")
    return values


@dataclass
class Settings:
    # --- inference ---
    backend: BackendName = "openai"
    model: str | None = None  # backend default when None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None  # secret
    llm_concurrency: int = 4
    llm_timeout: float = 300.0
    llm_retries: int = 2
    cache_llm: bool = True
    llm_effort: str = "low"  # reasoning effort for the claude-code and codex CLIs ("" = the CLI's default)
    cli_user_config: bool = False  # let the CLIs load your own config (CLAUDE.md, hooks, plugins, config.toml)

    # --- Semantic Scholar ---
    s2_api_key: str | None = None  # secret
    s2_rps: float = 1.0  # requests per second (unauthenticated pool is much lower)

    # --- pipeline ---
    target_papers: int = 60
    max_candidates: int = 250
    max_refs_per_paper: int = 15
    baseline_papers: int = 8  # papers added because graph papers compare against them (0 disables)
    followup_papers: int = 6  # recent papers added because they cite key graph papers influentially (0 disables)
    followup_seeds: int = 8  # key graph papers whose recent influential citations are scanned (one request each)
    followup_years: int = 2  # how recent a follow-up must be, in publication years (2 = this year and last)
    update_papers: int = 10  # most papers one "check for new work" adds (half from search, the rest follow-ups)

    # --- sandboxed full-text worker ---
    fulltext_enabled: bool = True
    fulltext_max_mb: int = 25
    fulltext_timeout: float = 90.0
    fulltext_mem_mb: int = 1024
    fulltext_allow_private: bool = False  # SSRF guard; only disable for local testing
    fulltext_prefer_html: bool = True  # try arXiv's HTML rendering before PDFs (cleaner text and tables)
    fulltext_always_fill_datasets: bool = True  # gap-fill datasets for every paper, not only thin results
    fulltext_single_pass: bool = True  # with always_fill_datasets: read the full text first, extract in one call
    fulltext_limitation_sections: bool = True  # read Limitations/Conclusion/Discussion sections when gap filling

    # --- web server ---
    api_token: str | None = None  # secret; generated at startup when unset
    max_jobs: int = 2  # concurrent build/refine jobs across all projects

    db_path: str = "research.forest.db"

    # --- logging ---
    llm_log_enabled: bool = True  # record every LLM call (input, output, sizes) as JSON lines
    llm_log_file: str = "research-forest-llm.jsonl"

    @classmethod
    def load(cls, config: str | Path | None = None, env_file: str | Path = ".env", **overrides: Any) -> Settings:
        """Defaults < config file < ``overrides`` (CLI flags); secrets from ``.env`` < environment.

        ``config=None`` uses ``research-forest.toml`` if it exists; an explicit path must exist.
        """
        path = Path(config) if config else Path(DEFAULT_CONFIG)
        if config and not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        values = read_config(path) if path.is_file() else {}

        load_dotenv(env_file)
        for name, env_vars in SECRETS.items():
            if secret := next((os.environ[v] for v in env_vars if os.environ.get(v)), None):
                values[name] = secret

        values.update({k: v for k, v in overrides.items() if v is not None})
        return replace(cls(), **values)
