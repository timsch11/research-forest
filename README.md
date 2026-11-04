# research-forest

Explore a research landscape as a **searchable, refinable graph**. Describe a research area in one
prompt. research-forest finds the relevant papers, traces **which papers build on which and what
each one changes**, and extracts the **methods**, **datasets**, **limitations** and
**open research questions** of the field. You can then refine the graph step by step with
natural-language commands, or ask it questions.

```
prompt ──► scope ──► search + LLM screening ──► citation snowball ──► per-paper extraction
                                                                       │  (abstract + citation contexts,
                                                                       │   gap filling from full text)
                     open questions ◄── limitation themes ◄── entity dedup ◄── lineage (builds-on + change type)
```

## Features

- **Paper discovery** via the [Semantic Scholar Graph API](https://api.semanticscholar.org/).
  The LLM writes the search queries and names landmark papers, screens the candidates for
  relevance, and the graph grows by snowballing through frequently co-cited references.
- **Lineage graph.** Each in-graph citation is classified as *builds on* or not. *Builds-on*
  edges record a typed change (`architecture`, `objective`, `training`, `data`, `efficiency`,
  `evaluation`, `theory`, `application`), the component that changed, and a one-sentence
  summary. Semantic Scholar citation contexts and intents are used as evidence.
- **Structured extraction.** Methods (proposed and used), datasets, limitations (stated by the
  authors, reported by citing papers, or inferred) and future work are extracted from the
  abstract plus the citation contexts.
- **Gap filling without reading whole papers.** When the abstract is not enough, research-forest
  first tries Semantic Scholar snippet search. If that finds nothing, it downloads the
  open-access PDF or HTML in a **sandboxed worker process** and runs BM25 passage search over
  it. Only the top passages are sent to the LLM.
- **Cross-paper synthesis.** Limitations are grouped into themes. Each theme is linked to the
  papers that have it and to the papers that address it. Open research questions are derived
  from the themes.
- **Search and Q&A.** SQLite FTS5 keyword search covers papers and all entities. You can also
  ask natural-language questions, which are answered from a retrieved subgraph, with clickable
  citations back to the nodes.
- **Refinement in natural language.** For example: *"expand around FlashAttention"*,
  *"drop vision papers"*, *"only work since 2021"*, *"focus on state-space models instead"*,
  *"look deeper into the limitations of Performer"*. Each command becomes a plan of graph
  operations (`search`, `expand`, `remove`, `filter`, `prune`, `deepen`, `rescope`,
  `resynthesize`). Only new work is computed, and every refinement takes a snapshot so you
  can **undo** it.
- **Three inference backends**, all behind one structured-output interface:
  - `openai`: any OpenAI-compatible `/chat/completions` endpoint (OpenAI, vLLM, Ollama,
    OpenRouter, LM Studio, …). It uses strict JSON-schema output and falls back automatically
    to a looser mode on servers that don't support it.
  - `claude-code`: the `claude` CLI (`claude -p --json-schema`, with tools disabled).
  - `codex`: the `codex` CLI (`codex exec --output-schema`, read-only sandbox).

  Every response is validated against a Pydantic model. Invalid output is retried with the
  validation error fed back to the model, and responses are cached on disk.

## Install

```bash
pip install -e .            # or: uv pip install -e .
pip install -e ".[dev]"     # with test dependencies
```

Python 3.11 or newer is required.

## Configure

All settings can be passed as CLI flags or as `RF_*` environment variables (see
`research_forest/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `RF_BACKEND` | `openai` | `openai`, `claude-code` or `codex` |
| `RF_MODEL` | `gpt-6-luna` (openai); CLI default otherwise | model name passed to the backend |
| `OPENAI_BASE_URL` / `RF_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` / `RF_OPENAI_API_KEY` | – | API key for the endpoint |
| `S2_API_KEY` / `RF_S2_API_KEY` | – | Semantic Scholar API key (recommended: higher rate limits, snippet search) |
| `RF_S2_RPS` | `1.0` | client-side request rate for Semantic Scholar |
| `RF_TARGET_PAPERS` | `60` | default graph size |
| `RF_LLM_CONCURRENCY` | `4` | parallel LLM calls |
| `RF_FULLTEXT_ENABLED` | `true` | sandboxed full-text gap filling |
| `RF_DB_PATH` | `research.forest.db` | SQLite file holding all landscapes |

Examples:

```bash
# OpenAI-compatible (local vLLM / Ollama works the same way)
export OPENAI_API_KEY=sk-...            # RF_MODEL defaults to gpt-6-luna
# Claude Code CLI (uses your Claude Code login)
export RF_BACKEND=claude-code
# Codex CLI
export RF_BACKEND=codex
```

## Use

### Web explorer

```bash
research-forest serve            # http://127.0.0.1:8000
```

1. Enter a research-area prompt and a target number of papers, then click **Build graph**. The
   progress log streams each stage, and the graph fills in as the build runs.
2. **Lineage** view: a timeline with papers placed by year and *builds-on* edges labelled with
   their change type. **Landscape** view: every node type, laid out by force direction.
3. Filter by node type, year range and change type. The search box covers papers, methods,
   datasets, limitations and questions.
4. Click any node to see its details: contribution, what it builds on and what changes,
   methods, datasets, limitations with their source, related open questions.
5. In the bottom bar, **Refine** takes natural-language commands that change the graph (and
   **Undo** reverts them). **Ask** takes questions; cited nodes are highlighted in the graph.

### CLI

```bash
research-forest build "Efficient attention mechanisms for long-context transformers" -n 60
research-forest list
research-forest refine 1 "expand around FlashAttention and drop papers before 2019"
research-forest ask 1 "Which efficient-attention methods were never evaluated on long-document benchmarks?"
research-forest search 1 "state space"
research-forest undo 1
```

## Sandbox for full-text processing

Downloads and PDF parsing are untrusted input, so they run in `sources/fulltext_worker.py`, a
separate process that is:

- started with `python -I` (isolated mode: no `PYTHONPATH`, no user site-packages, no cwd on
  `sys.path`) and imports nothing from this package;
- run in a throwaway temp directory with a minimal environment (only `PATH`, locale, proxy and
  CA variables);
- resource-limited with `RLIMIT_AS` (memory), `RLIMIT_CPU`, `RLIMIT_FSIZE` (≈ no disk writes)
  and no core dumps, plus a wall-clock timeout;
- restricted to `http(s)` URLs, to PDF, HTML or plain-text responses, and to a maximum download
  size (`RF_FULLTEXT_MAX_MB`), with documents held in memory only;
- wrapped in **bubblewrap** (`bwrap`) when it is installed: read-only system and Python paths,
  your home directory hidden, a private `/tmp`, every namespace unshared except the network.

## Platform support

The code is plain Python and runs on Linux, macOS and Windows. Only the sandboxing of the
full-text worker depends on the operating system:

| | Linux | macOS | Windows |
|---|---|---|---|
| Isolated subprocess (`python -I`, temp dir, minimal env, timeout, size/type caps) | ✓ | ✓ | ✓ |
| rlimits (memory, CPU, file size, core dumps) | ✓ | ✓ (memory limit is often ignored by the kernel) | – |
| bubblewrap namespace sandbox | ✓ if `bwrap` is installed | – | – |

So the full-text sandbox is strongest on Linux with `bwrap`. On Windows it falls back to process
isolation with a timeout, without kernel-level resource limits. If you want the strongest
isolation there, run the tool under WSL2. You can also disable full-text gap filling with
`--no-fulltext` (or `RF_FULLTEXT_ENABLED=false`).

Other notes: the `claude-code` / `codex` backends need those CLIs on `PATH` (on Windows,
`claude.cmd` / `codex.cmd` are found automatically). SQLite must include FTS5, which the
python.org, Homebrew and most Linux-distribution builds do.

## Architecture

```
research_forest/
  config.py            Settings (RF_* env vars)
  models.py            domain objects + strict LLM output schemas
  prompts.py           all prompt templates
  llm/                 LLM wrapper (cache, retries, validation) + openai / claude-code / codex backends
  sources/             Semantic Scholar client, passage finder, sandboxed full-text worker
  pipeline/            discover (scope, search, screen, select, snowball, expand, prune)
                       analyze (extract + gap filling, lineage relations, entity normalisation)
                       synthesize (limitation themes, open questions)
  store.py             SQLite: papers, entities, typed edges, FTS5 index, snapshots
  engine.py            Forest: build / process (incremental) / refine / ask / undo
  web/                 FastAPI app + Cytoscape.js single-page UI (vendored, works offline)
  cli.py               command-line interface
```

The graph is stored as typed edges between `p:<paper>` and `e:<entity>` nodes: `cites`,
`builds_on`, `proposes`, `uses`, `evaluates_on`, `has_limitation`, `addresses`, `motivates`
and `about`. Processing is incremental. Citation links are fetched once per paper, and
extraction runs once per paper (or again on `deepen`). Each citation pair is judged for
builds-on exactly once, so refinements only pay for what is new.

## Tests

```bash
pytest
```

The tests use a deterministic fake LLM backend and a mocked Semantic Scholar API, so they run
offline. The full-text tests start the real sandboxed worker against a local HTTP server.
