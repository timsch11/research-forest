# Getting started

## Install

```bash
git clone https://github.com/timsch11/research-forest.git
cd research-forest
pip install -e .            # or: uv pip install -e .
```

Python 3.11 or newer is required.

## Pick an inference backend

research-forest talks to LLMs through one interface. Pick the backend that suits you:

| Backend | Setup | Good for |
|---|---|---|
| `openai` (default) | `export OPENAI_API_KEY=sk-…` | OpenAI or any OpenAI-compatible server: vLLM, Ollama, OpenRouter, LM Studio, … |
| `claude-code` | `export RF_BACKEND=claude-code` and log in to the `claude` CLI | Using your Claude Code subscription |
| `codex` | `export RF_BACKEND=codex` and log in to the `codex` CLI | Using your Codex subscription |

The `openai` backend defaults to `gpt-6-luna`. Override the model with `RF_MODEL` (or
`--model`). For a local server, set the endpoint:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
export RF_MODEL=qwen3:32b
```

> [!TIP]
> Get a free [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)
> and `export S2_API_KEY=…`. The shared unauthenticated rate limit is low, so a 60-paper
> build is much faster with a key. Snippet search (used for gap filling) may also require one.

## Build your first landscape

**In the browser:**

```bash
research-forest serve
# research-forest explorer: http://127.0.0.1:8000/?token=…
```

1. Open the printed URL. It contains a one-time access token.
2. Describe a research area.
3. Click **Build graph**.

The progress log shows each stage as the graph fills in. A 60-paper build takes a few minutes,
depending on the backend and the Semantic Scholar rate limit.

**From the terminal:**

```bash
research-forest build "Efficient attention mechanisms for long-context transformers" -n 60
```

**Just looking?** `python scripts/demo.py` serves a hand-made example landscape, with no API
keys needed.

## Platform support

research-forest runs on Linux, macOS and Windows. Only the sandbox around full-text downloads
depends on the operating system:

| | Linux | macOS | Windows |
|---|:-:|:-:|:-:|
| Isolated worker process (`python -I`, temp dir, minimal env, timeout, size and type caps) | ✓ | ✓ | ✓ |
| Kernel resource limits (memory, CPU, file size, core dumps) | ✓ | ✓¹ | – |
| [bubblewrap](https://github.com/containers/bubblewrap) namespace sandbox | ✓² | – | – |

¹ macOS often ignores the memory limit. ² Used when `bwrap` is installed.

On Windows, use WSL2 for the strongest isolation, or turn off full-text gap filling with
`--no-fulltext`. The `claude-code` and `codex` backends need their CLIs on `PATH`; on Windows,
the `.cmd` shims are found automatically. Your Python's SQLite must include FTS5, which the
python.org, Homebrew and most Linux-distribution builds do.
