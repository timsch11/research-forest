# Development

## Setup

```bash
pip install -e ".[dev]"
pytest                      # ~230 tests, offline, a few seconds
python scripts/demo.py      # browse a hand-made landscape without API keys
```

The tests use a deterministic fake LLM backend and a mocked Semantic Scholar API (see
[`tests/conftest.py`](../tests/conftest.py)), so they need neither network nor credentials. The
browser XSS test ([`tests/test_ui_xss.py`](../tests/test_ui_xss.py)) runs when Node.js and
Playwright with Chromium are available, and is skipped otherwise.

## Code layout

```text
research_forest/
├── config.py          Settings (RF_* environment variables)
├── models.py          domain objects + strict, bounded LLM output schemas
├── prompts.py         all prompt templates (untrusted text fenced in data blocks)
├── safety.py          sanitisation, data fencing, injection heuristics, URL checks
├── engine.py          Forest: build · process (incremental) · refine · ask · undo
├── store.py           SQLite: papers, entities, typed edges, FTS5 index, snapshots
├── llm/
│   ├── base.py        LLM wrapper: caching, retries, validation, strict-schema conversion
│   ├── openai_compat.py
│   └── cli.py         claude-code and codex backends
├── sources/
│   ├── semantic_scholar.py   rate-limited async client
│   ├── fulltext.py           passage finder + sandbox launcher
│   └── fulltext_worker.py    isolated download → text → BM25 worker (stdlib + httpx + pypdf)
├── pipeline/
│   ├── discover.py    scope · search · screen · select · snowball · expand · prune · filter
│   ├── analyze.py     extraction + gap filling · lineage relations · entity normalisation
│   └── synthesize.py  limitation themes · open questions
├── web/
│   ├── server.py      FastAPI app, auth and security middleware, job runner
│   └── static/        single-page UI (vanilla JS + vendored Cytoscape.js)
└── cli.py
```

## Test map

| File | Covers |
|---|---|
| `test_pipeline.py` | Build, lineage, normalisation, gap filling, every refinement op, Q&A, end-to-end injection containment |
| `test_store.py` | Persistence, merges, orphan pruning, FTS escaping, snapshots and restore |
| `test_llm.py` | Retry, cache, all three backends (CLIs stubbed), settings parsing |
| `test_semantic_scholar.py` | Sanitisation, retries, throttling, batching, citation direction |
| `test_fulltext.py` | BM25, the real sandboxed worker, SSRF guard, redirects, gzip bombs, output validation |
| `test_prompts.py`, `test_safety.py`, `test_models.py` | Prompt fencing, sanitisation, output bounds, strict schemas |
| `test_web.py`, `test_cli.py` | Auth, CSRF, headers, input validation, job limits, CLI commands |
| `test_ui_xss.py` | Hostile data rendered in a real browser |

## Extending

**Add an inference backend.** Subclass `llm.base.Backend` and implement
`async generate(system, user, schema, schema_name) -> str | dict`, returning JSON that matches
`schema`. Then register it in `llm/__init__.py:make_backend`. Validation, retries and caching
come for free from `LLM`.

**Add a paper source.** Return `models.Paper` and `models.CitationLink` objects, as
`SemanticScholar` does. Sanitise every text field with `safety.sanitize` and every URL with
`safety.safe_url`, because source data is untrusted.

**Add a prompt.** Put third-party or LLM-derived text inside `safety.data_block(...)` after
`sanitize`/`clip`, and user text inside `safety.trusted_block(...)`. Add the prompt to the
`PROMPTS` table in `tests/test_prompts.py` so the containment test covers it.

## Regenerating screenshots

The README screenshots come from `scripts/demo.py`, captured with Playwright at 1440×860 and a
device scale factor of 2 in both colour schemes, then saved to `docs/assets/` as WebP.
