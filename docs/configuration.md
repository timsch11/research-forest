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
| `model` | `gpt-6-luna` for `openai`; the CLI's default otherwise | Model passed to the backend |
| `base_url` | `"https://api.openai.com/v1"` | OpenAI-compatible endpoint |
| `concurrency` | `4` | Parallel LLM calls |
| `timeout` | `300` | Seconds per call |
| `retries` | `2` | Retries after invalid output or transient errors |
| `cache` | `true` | Cache responses in `<db_path>.llmcache` |

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

### `[fulltext]`

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Sandboxed download + BM25 gap filling (`--no-fulltext` disables it) |
| `max_mb` | `25` | Maximum (decompressed) download size |
| `timeout` | `90` | Seconds per document |
| `memory_mb` | `1024` | Worker memory limit (POSIX) |
| `allow_private` | `false` | Disables the SSRF guard. **Local testing only.** |
| `prefer_html` | `true` | Try arXiv's HTML rendering before PDFs (cleaner text, readable tables) |

### `[server]` and `[storage]`

| Key | Default | Description |
|---|---|---|
| `server.max_jobs` | `2` | Concurrent build and refine jobs |
| `storage.db_path` | `"research.forest.db"` | SQLite file holding all landscapes (`--db` overrides) |

## Secrets (`.env` or environment)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` (or `RF_OPENAI_API_KEY`) | API key for the OpenAI-compatible endpoint |
| `S2_API_KEY` (or `RF_S2_API_KEY`) | Semantic Scholar API key (recommended) |
| `RF_API_TOKEN` | Fixed access token for the web API (random per start when unset) |

The `.env` format is one `KEY=value` per line, and `#` comments, quotes and `export` prefixes
are all fine. Only these secret variables are read from the environment. All other settings
come from the config file.
