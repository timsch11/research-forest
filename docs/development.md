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
├── config.py          Settings: TOML config file + secrets from .env / environment
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

## Evaluating extraction quality

`eval/gold.json` holds 17 well-known papers (Transformer, BERT, ResNet, Mamba, FlashAttention,
…). Each comes with the datasets it uses (with accepted aliases) and the method it proposes.
13 of them also list limitations the paper itself states (22 in total). Limitations are free
text, so an LLM judge, using your configured backend, decides which gold limitations each
extraction covers.

```bash
research-forest evaluate --out before.json          # uses your config: backend, model, …
# … change a prompt, a query or the model …
research-forest evaluate --compare before.json      # shows deltas
```

Each paper is fetched from Semantic Scholar with its references and a sample of citing papers,
then analysed exactly as in a build, in a throwaway database. The report shows:
- **dataset recall:** the share of gold datasets that were extracted;
- **dataset precision:** a lower bound, because gold lists cover only the main datasets;
- **proposed method found:** whether the paper's own method was extracted;
- **papers with limitations:** how many got at least one limitation;
- **gap filling:** how many papers needed full-text passages;
- **baseline recall:** gold baselines that were extracted, and **baselines resolved to papers**
  (the per-paper report lists which paper each baseline resolved to, for review);
- **limitation recall:** judged by the LLM, plus the share of extracted limitations the authors
  state themselves, and limitations per paper;
- per paper, which gold datasets were missed.

A run makes about 50 Semantic Scholar requests and 20–40 LLM calls. Add papers to the gold
set to cover your own field.

For reference, with Claude Haiku through the `claude-code` backend and no Semantic Scholar key,
the evidence and ranking changes raised dataset recall from 56% to 73% and precision from 85%
to 90%.

### Comparing full-text sources

```bash
research-forest evaluate --sources
```

This runs the dataset passage search on each gold paper's arXiv HTML and on its PDF, then
counts the gold datasets named in the top six passages. No LLM is involved, so the result is
deterministic and costs nothing but downloads. On the shipped gold set, HTML scores 82% and
PDF 73%, and HTML is about twice as fast to process.

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
