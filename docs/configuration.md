# Configuration

Settings come from `RF_*` environment variables. A few CLI flags override them (`--backend`,
`--model`, `--base-url`, `--db`, `--no-fulltext`). Boolean variables accept
`1/true/yes/on`.

## Inference

| Variable | Default | Description |
|---|---|---|
| `RF_BACKEND` | `openai` | `openai`, `claude-code` or `codex` |
| `RF_MODEL` | `gpt-6-luna` for `openai`; the CLI's default otherwise | Model passed to the backend |
| `RF_OPENAI_BASE_URL` / `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `RF_OPENAI_API_KEY` / `OPENAI_API_KEY` | – | API key for that endpoint |
| `RF_LLM_CONCURRENCY` | `4` | Parallel LLM calls |
| `RF_LLM_TIMEOUT` | `300` | Seconds per call |
| `RF_LLM_RETRIES` | `2` | Retries after invalid output or transient errors |
| `RF_CACHE_LLM` | `true` | Cache responses in `<db>.llmcache` |

## Semantic Scholar

| Variable | Default | Description |
|---|---|---|
| `RF_S2_API_KEY` / `S2_API_KEY` | – | API key (recommended) |
| `RF_S2_RPS` | `1.0` | Client-side request rate limit |

## Pipeline

| Variable | Default | Description |
|---|---|---|
| `RF_TARGET_PAPERS` | `60` | Default graph size (`-n` on `build`) |
| `RF_MAX_CANDIDATES` | `250` | Upper bound on papers screened per screening step |
| `RF_MAX_REFS_PER_PAPER` | `15` | In-graph references judged for builds-on per citing paper |

## Full-text gap filling

| Variable | Default | Description |
|---|---|---|
| `RF_FULLTEXT_ENABLED` | `true` | Enable the sandboxed download + BM25 fallback |
| `RF_FULLTEXT_MAX_MB` | `25` | Maximum (decompressed) download size |
| `RF_FULLTEXT_TIMEOUT` | `90` | Seconds per document |
| `RF_FULLTEXT_MEM_MB` | `1024` | Worker memory limit (POSIX) |
| `RF_FULLTEXT_ALLOW_PRIVATE` | `false` | Disables the SSRF guard. **Local testing only.** |

## Web server and storage

| Variable | Default | Description |
|---|---|---|
| `RF_DB_PATH` | `research.forest.db` | SQLite file holding all landscapes (`--db`) |
| `RF_API_TOKEN` | random per start | Fixed access token for the web API |
| `RF_MAX_JOBS` | `2` | Concurrent build and refine jobs |
