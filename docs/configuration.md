# Configuration

research-forest reads two files from the working directory:

| File | Holds | Committed? |
|---|---|---|
| `research-forest.toml` | All settings: backend, model, endpoint, limits, paths | Yes, it's safe to share |
| `.env` | Secrets only: API keys and the web token | No, it's git-ignored |

Start from the templates: `research-forest.example.toml` (every option with its default) and
`.env.example`.

**Precedence**
- Settings: built-in defaults < config file < CLI flags (`--backend`, `--model`, `--base-url`,
  `--db`, `--no-fulltext`). Use `--config PATH` for a config file elsewhere.
- Secrets: `.env` < variables exported in the terminal.

The config file is validated when it's loaded. Unknown keys (typos), wrong types and invalid
backends are reported with the offending `[section] key`. Secrets placed in the config file are
rejected, so it can never leak a key into git.

## Config file (`research-forest.toml`)

All settings are optional.

### `[llm]`

| Key | Default | Description |
|---|---|---|
| `backend` | `"openai"` | `"openai"`, `"claude-code"` or `"codex"` |
| `model` | `gpt-6-luna` for `openai` and `codex`, `haiku` for `claude-code` | Model passed to the backend. The CLI backends deliberately don't use your CLI's default model, which is often the largest one |
| `base_url` | `"https://api.openai.com/v1"` | OpenAI-compatible endpoint |
| `concurrency` | `4` | Parallel LLM calls |
| `timeout` | `300` | Seconds per call |
| `retries` | `2` | Retries after invalid output or transient errors |
| `cache` | `true` | Cache responses in `<db_path>.llmcache` |
| `effort` | `"low"` | Reasoning effort for `claude-code` and `codex`: `low`, `medium`, `high`, `xhigh` or `max`. `""` uses the CLI's default. Extraction doesn't need deep reasoning, and thinking tokens are usually the largest part of a call's output |
| `cli_user_config` | `false` | Let `claude` and `codex` load your own setup. By default, Claude Code runs with `--setting-sources ""` (no CLAUDE.md, memory, hooks or plugins) and Codex with `--ignore-user-config` (no `config.toml`, so none of its MCP servers, model or effort). Otherwise all of that is sent along with every call. Turn this on only if your login depends on it, e.g. an `apiKeyHelper` or proxy set in `~/.claude/settings.json` |

### `[semantic_scholar]`

| Key | Default | Description |
|---|---|---|
| `requests_per_second` | `1.0` | Client-side request rate limit |

### `[pipeline]`

| Key | Default | Description |
|---|---|---|
| `target_papers` | `60` | Default graph size (`build -n` overrides) |
| `max_candidates` | `250` | Upper bound on papers screened per screening step |
| `max_refs_per_paper` | `15` | In-graph references judged for builds-on per citing paper |
| `baseline_papers` | `8` | Add up to this many papers that graph papers compare against in their results (0 turns it off) |
| `followup_papers` | `6` | Add up to this many recent papers that cite key graph papers influentially (0 turns it off) |
| `followup_seeds` | `8` | Number of key graph papers whose recent citations are scanned, at one Semantic Scholar request each |
| `followup_years` | `2` | How recent a follow-up must be, in publication years (2 means this year and last) |
| `update_papers` | `10` | Most papers one *check for new work* adds: up to half from search, the rest follow-ups |

### `[fulltext]`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Sandboxed download + BM25 gap filling (`--no-fulltext` disables it) |
| `max_mb` | `25` | Maximum (decompressed) download size |
| `timeout` | `90` | Seconds per document |
| `memory_mb` | `1024` | Worker memory limit (POSIX) |
| `allow_private` | `false` | Disables the SSRF guard. **Local testing only.** |
| `prefer_html` | `true` | Try arXiv's HTML rendering before PDFs (cleaner text, readable tables) |
| `limitation_sections` | `true` | When the full text is read, also take limitations from its Limitations, Conclusion, Discussion and Future-work sections |
| `always_fill_datasets` | `true` | Search the full text for datasets for every paper. `false` does this only when fewer than 3 datasets were found |
| `single_pass` | `true` | With `always_fill_datasets`: read the full text *before* extracting, so each paper takes one extraction call instead of an extraction plus a gap-fill call. A gap-fill call is still made for a field that came back empty without full-text evidence |

### `[server]` and `[storage]`

| Key | Default | Description |
|---|---|---|
| `server.max_jobs` | `2` | Concurrent build and refine jobs |
| `storage.db_path` | `"research.forest.db"` | SQLite file holding all landscapes (`--db` overrides) |

### `[logging]`

| Key | Default | Description |
|---|---|---|
| `llm_calls` | `true` | Record every LLM call as one JSON line: backend, model, call type, attempt, cache hit, duration, the full system and user prompt, the output or error, word and character counts, and the tokens the backend reported |
| `llm_log_file` | `"research-forest-llm.jsonl"` | Where to write it (git-ignored). `research-forest llm-log` summarises it |

Tokens are logged as the backend reports them: `input` (including prompt-cache reads and
everything the CLI adds around our prompt), `cached`, `output` (including reasoning) and
`reasoning`. Claude Code also reports an estimated `cost_usd` and the number of `turns`. A server
that reports no usage only gets word and character counts (tokens ≈ 1.3 × words).

## Secrets (`.env` or environment)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` (or `RF_OPENAI_API_KEY`) | API key for the OpenAI-compatible endpoint |
| `S2_API_KEY` (or `RF_S2_API_KEY`) | Semantic Scholar API key (recommended) |
| `RF_API_TOKEN` | Fixed access token for the web API (random per start when unset) |

The `.env` format is one `KEY=value` per line, and `#` comments, quotes and `export` prefixes
are all fine. Only these secret variables are read from the environment. All other settings
come from the config file.
