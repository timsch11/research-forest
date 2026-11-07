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
| `claude-code` (default) | Log in to the `claude` CLI | Using your Claude Code subscription |
| `codex` | `backend = "codex"` in the config, and log in to the `codex` CLI | Using your Codex subscription |
| `openai` | `backend = "openai"` in the config, `OPENAI_API_KEY` in `.env` | OpenAI or any OpenAI-compatible server: vLLM, Ollama, OpenRouter, LM Studio, … |

Settings live in two places (see [Configuration](configuration.md)):
- **`research-forest.toml`** holds settings like the backend, model, endpoint and graph size.
  It is safe to commit.
- **`.env`** holds secrets only. Variables you `export` in the terminal override it.

`research-forest.toml` is preset to the defaults; edit it in place. For secrets:

```bash
cp .env.example .env               # then fill in OPENAI_API_KEY and S2_API_KEY
```

The `openai` backend defaults to `gpt-6-luna`. For a local server, edit the config:

```toml
[llm]
backend = "openai"
base_url = "http://localhost:11434/v1"   # e.g. Ollama
model = "qwen3:32b"
```

> [!TIP]
> Get a free [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)
> and put it in `.env` as `S2_API_KEY`. The shared unauthenticated rate limit is low, so a
> 60-paper build is much faster with a key. Snippet search (used for gap filling) may also
> require one.

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
