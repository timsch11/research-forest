import json
from pathlib import Path

import httpx
import pytest

from research_forest.config import Settings
from research_forest.llm import LLM, Backend, JsonCache, TransientLLMError, make_backend
from research_forest.llm import base as base_mod
from research_forest.llm import cli as cli_mod
from research_forest.llm.base import parse_json
from research_forest.llm.cli import ClaudeCodeBackend, CodexBackend
from research_forest.llm.openai_compat import DEFAULT_MODEL, OpenAIBackend
from research_forest.models import Scope

SCOPE = {"title": "t", "summary": "s", "search_queries": ["q"], "seed_papers": [], "include_criteria": "i",
         "exclude_criteria": "e", "year_min": None}


class Scripted(Backend):
    name = "scripted"

    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    async def generate(self, system, user, schema, schema_name):
        self.prompts.append(user)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_s):
        return None
    monkeypatch.setattr(base_mod.asyncio, "sleep", instant)


@pytest.mark.parametrize("text", [
    json.dumps(SCOPE),
    "```json\n" + json.dumps(SCOPE) + "\n```",
    "Here you go: " + json.dumps(SCOPE) + " hope it helps",
])
def test_parse_json_variants(text):
    assert parse_json(text)["title"] == "t"


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json("no json here")


async def test_retry_on_invalid_output_then_cache(tmp_path):
    backend = Scripted(["not json", "```json\n" + json.dumps(SCOPE) + "\n```"])
    llm = LLM(backend, retries=2, cache=JsonCache(tmp_path / "c.db"))
    scope = await llm.json(Scope, "sys", "user")
    assert scope.title == "t"
    assert "previous answer was invalid" in backend.prompts[1]
    again = await llm.json(Scope, "sys", "user")  # served from cache
    assert again == scope and len(backend.prompts) == 2


async def test_cache_bypass_and_key_sensitivity(tmp_path):
    backend = Scripted([json.dumps(SCOPE)] * 3)
    llm = LLM(backend, cache=JsonCache(tmp_path / "c.db"))
    await llm.json(Scope, "sys", "user")
    await llm.json(Scope, "sys", "user", use_cache=False)
    await llm.json(Scope, "other system", "user")
    assert len(backend.prompts) == 3


async def test_transient_errors_are_retried():
    llm = LLM(Scripted([TransientLLMError("429"), json.dumps(SCOPE)]), retries=1)
    assert (await llm.json(Scope, "s", "u")).title == "t"


async def test_gives_up_after_retries():
    llm = LLM(Scripted(["{}", "{}"]), retries=1)
    with pytest.raises(RuntimeError, match="Scope"):
        await llm.json(Scope, "sys", "user")


async def test_oversized_output_is_rejected():
    llm = LLM(Scripted(["x" * (base_mod.MAX_OUTPUT_CHARS + 1)]), retries=0)
    with pytest.raises(RuntimeError, match="too large"):
        await llm.json(Scope, "s", "u")


async def test_dict_output_from_cli_backends_is_validated():
    llm = LLM(Scripted([{"title": "only title"}]), retries=0)
    with pytest.raises(RuntimeError):
        await llm.json(Scope, "s", "u")


# ------------------------------------------------------------------- openai


def openai_backend(handler, **kw):
    return OpenAIBackend("http://llm.local/v1", "key", kw.pop("model", "m"), transport=httpx.MockTransport(handler))


async def test_openai_backend_downgrades_response_format():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("response_format", {}).get("type"))
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "response_format not supported"})
        assert "JSON schema" in body["messages"][0]["content"]  # schema moved into the prompt
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(SCOPE)}}]})

    backend = openai_backend(handler)
    assert (await LLM(backend).json(Scope, "sys", "user")).title == "t"
    assert seen == ["json_schema", "json_schema", "json_object"]
    await backend.aclose()


async def test_openai_backend_request_shape():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(SCOPE)}}]})

    backend = openai_backend(handler)
    await LLM(backend).json(Scope, "sys", "user")
    body = captured["body"]
    assert captured["auth"] == "Bearer key"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    await backend.aclose()


async def test_openai_backend_transient_errors():
    backend = openai_backend(lambda r: httpx.Response(503, text="busy"))
    with pytest.raises(TransientLLMError):
        await backend.generate("s", "u", {}, "S")
    await backend.aclose()


def test_openai_default_model_and_plain_http_warning(caplog):
    assert OpenAIBackend("https://api.openai.com/v1", None, None).model == DEFAULT_MODEL == "gpt-6-luna"
    OpenAIBackend("http://remote.example/v1", "secret", None)
    assert "plain HTTP" in caplog.text
    caplog.clear()
    OpenAIBackend("http://localhost:8000/v1", "secret", None)
    assert "plain HTTP" not in caplog.text


# ---------------------------------------------------------------- CLI tools


async def test_claude_code_backend_is_locked_down(monkeypatch):
    captured = {}

    async def fake_run(args, stdin, cwd, timeout):
        captured.update(args=args, stdin=stdin, cwd=cwd)
        return json.dumps({"type": "result", "is_error": False, "result": "", "structured_output": SCOPE})

    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    monkeypatch.setattr(cli_mod, "_run", fake_run)
    scope = await LLM(ClaudeCodeBackend(model="opus")).json(Scope, "sys", "hello")
    assert scope.title == "t"
    args = captured["args"]
    assert args[args.index("--tools") + 1] == ""
    assert {"--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"} <= set(args)
    assert args[args.index("--model") + 1] == "opus"
    assert captured["stdin"] == "hello" and "rf-claude-" in captured["cwd"]
    assert not Path(captured["cwd"]).exists()  # throwaway working dir is removed


async def test_claude_code_error_envelope(monkeypatch):
    async def fake_run(args, stdin, cwd, timeout):
        return json.dumps({"is_error": True, "result": "rate limited"})

    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    monkeypatch.setattr(cli_mod, "_run", fake_run)
    with pytest.raises(TransientLLMError, match="rate limited"):
        await ClaudeCodeBackend().generate("s", "u", {}, "S")


async def test_codex_backend(monkeypatch):
    captured = {}

    async def fake_run(args, stdin, cwd, timeout):
        captured.update(args=args, stdin=stdin)
        schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
        assert schema["additionalProperties"] is False
        Path(args[args.index("--output-last-message") + 1]).write_text(json.dumps(SCOPE))
        return ""

    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    monkeypatch.setattr(cli_mod, "_run", fake_run)
    assert (await LLM(CodexBackend(model="gpt-6-luna")).json(Scope, "SYS", "USER")).title == "t"
    args = captured["args"]
    assert args[args.index("--sandbox") + 1] == "read-only" and args[-1] == "-"
    assert captured["stdin"].startswith("SYS") and captured["stdin"].endswith("USER")


async def test_codex_without_output_is_transient(monkeypatch):
    async def fake_run(args, stdin, cwd, timeout):
        return ""

    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    monkeypatch.setattr(cli_mod, "_run", fake_run)
    with pytest.raises(TransientLLMError):
        await CodexBackend().generate("s", "u", {"type": "object"}, "S")


async def test_run_helper_reports_failures_and_timeouts(tmp_path):
    import sys
    out = await cli_mod._run([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], "hi",
                             str(tmp_path), 10)
    assert out.strip() == "HI"
    with pytest.raises(TransientLLMError, match="exited 3"):
        await cli_mod._run([sys.executable, "-c", "raise SystemExit(3)"], "", str(tmp_path), 10)
    with pytest.raises(TransientLLMError, match="timed out"):
        await cli_mod._run([sys.executable, "-c", "import time; time.sleep(5)"], "", str(tmp_path), 0.3)


def test_missing_cli_binary():
    with pytest.raises(RuntimeError, match="not found"):
        ClaudeCodeBackend(binary="definitely-not-installed-xyz")


def test_make_backend_dispatch(monkeypatch):
    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    assert make_backend(Settings(backend="openai")).name == "openai"
    assert make_backend(Settings(backend="claude-code")).name == "claude-code"
    assert make_backend(Settings(backend="codex")).name == "codex"
    with pytest.raises(ValueError):
        make_backend(Settings(backend="nope"))  # type: ignore[arg-type]
