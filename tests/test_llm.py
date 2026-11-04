import json

import httpx
import pytest

from research_forest.llm import LLM, Backend, JsonCache, strict_schema
from research_forest.llm import cli as cli_mod
from research_forest.llm.cli import ClaudeCodeBackend
from research_forest.llm.openai_compat import OpenAIBackend
from research_forest.models import Extraction, RefinePlan, Scope


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


@pytest.mark.parametrize("model", [Scope, Extraction, RefinePlan])
def test_strict_schema(model):
    schema = strict_schema(model)
    assert "$defs" not in schema
    for node in walk(schema):
        assert "$ref" not in node and "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


class Scripted(Backend):
    name = "scripted"

    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    async def generate(self, system, user, schema, schema_name):
        self.prompts.append(user)
        return self.outputs.pop(0)


SCOPE = {"title": "t", "summary": "s", "search_queries": ["q"], "seed_papers": [], "include_criteria": "i",
         "exclude_criteria": "e", "year_min": None}


async def test_retry_on_invalid_output_then_cache(tmp_path):
    backend = Scripted(["not json", "```json\n" + json.dumps(SCOPE) + "\n```"])
    llm = LLM(backend, retries=2, cache=JsonCache(tmp_path / "c.db"))
    scope = await llm.json(Scope, "sys", "user")
    assert scope.title == "t"
    assert "previous answer was invalid" in backend.prompts[1]
    again = await llm.json(Scope, "sys", "user")  # served from cache
    assert again == scope and len(backend.prompts) == 2


async def test_gives_up_after_retries():
    llm = LLM(Scripted(["{}", "{}"]), retries=1)
    with pytest.raises(RuntimeError, match="Scope"):
        await llm.json(Scope, "sys", "user")


async def test_openai_backend_downgrades_response_format():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("response_format", {}).get("type"))
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "response_format not supported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(SCOPE)}}]})

    backend = OpenAIBackend("http://llm.local/v1", "key", "m", transport=httpx.MockTransport(handler))
    scope = await LLM(backend).json(Scope, "sys", "user")
    assert scope.title == "t"
    assert seen == ["json_schema", "json_schema", "json_object"]
    await backend.aclose()


async def test_claude_code_backend_reads_structured_output(monkeypatch):
    captured = {}

    async def fake_run(args, stdin, cwd, timeout):
        captured.update(args=args, stdin=stdin)
        return json.dumps({"type": "result", "is_error": False, "result": "", "structured_output": SCOPE})

    monkeypatch.setattr(cli_mod, "_require", lambda b: b)
    monkeypatch.setattr(cli_mod, "_run", fake_run)
    scope = await LLM(ClaudeCodeBackend(model="opus")).json(Scope, "sys", "hello")
    assert scope.title == "t"
    args = captured["args"]
    assert args[args.index("--tools") + 1] == "" and "--json-schema" in args and captured["stdin"] == "hello"
