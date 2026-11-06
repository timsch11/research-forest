import json

import pytest

from research_forest import cli
from research_forest.config import Settings
from research_forest.llm import LLM, Backend, CallLog, JsonCache, make_llm
from research_forest.llm.calllog import read_entries, summarize
from research_forest.models import Scope

SCOPE = {"title": "t", "summary": "two words", "search_queries": ["q"], "seed_papers": [], "include_criteria": "i",
         "exclude_criteria": "e", "year_min": None}


class Scripted(Backend):
    name, model = "scripted", "m1"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate(self, system, user, schema, schema_name):
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


async def test_every_call_is_logged_with_sizes(tmp_path):
    path = tmp_path / "calls.jsonl"
    llm = LLM(Scripted(["not json", json.dumps(SCOPE)]), retries=1, call_log=CallLog(path),
              cache=JsonCache(tmp_path / "c.db"))
    await llm.json(Scope, "system words here", "user prompt text")
    await llm.json(Scope, "system words here", "user prompt text")  # cache hit
    entries = list(read_entries(path))
    assert [(e["call"], e["attempt"], e["cached"], e["error"] is None) for e in entries] == [
        ("Scope", 1, False, False), ("Scope", 2, False, True), ("Scope", 1, True, True)]
    first, second = entries[0], entries[1]
    assert first["output"] == "not json" and "previous answer was invalid" in second["user"]
    assert second["words"]["system"] == 3 and second["words"]["output"] == len(second["output"].split())
    assert second["chars"]["user"] == len(second["user"]) and second["seconds"] >= 0
    assert first["backend"] == "scripted" and first["model"] == "m1" and first["ts"]


async def test_reported_tokens_are_logged_and_summarised(tmp_path):
    from research_forest.llm import Reply
    path = tmp_path / "calls.jsonl"
    tokens = {"input": 4000, "cached": 3000, "output": 900, "reasoning": 600, "cost_usd": 0.0011, "turns": 2}
    llm = LLM(Scripted([Reply("not json", tokens), Reply(SCOPE, tokens)]), retries=1, call_log=CallLog(path),
              cache=JsonCache(tmp_path / "c.db"))
    await llm.json(Scope, "s", "u")
    await llm.json(Scope, "s", "u")  # cache hit: no tokens
    entries = list(read_entries(path))
    assert [e["tokens"] for e in entries] == [tokens, tokens, None]  # failed attempts cost tokens too
    report = summarize(path)
    total = next(line for line in report.splitlines() if line.startswith("total"))
    assert total.split()[1:] == ["3", "1", "1", *total.split()[4:6], "8000", "6000", "1800", "1200",
                                 total.split()[-1]]
    assert "for 2 of 2 uncached calls" in report and "$0.0022" in report


def test_summary_without_token_counts(tmp_path):
    path = tmp_path / "calls.jsonl"
    CallLog(path).record(backend="b", model=None, call="Scope", attempt=1, system="s", user="u v", output="x")
    assert "reported no token counts" in summarize(path)


async def test_backend_errors_are_logged_then_raised(tmp_path):
    path = tmp_path / "calls.jsonl"
    llm = LLM(Scripted([KeyError("boom")]), call_log=CallLog(path))
    with pytest.raises(KeyError):
        await llm.json(Scope, "s", "u")
    (entry,) = read_entries(path)
    assert "boom" in entry["error"] and entry["output"] == ""


async def test_logging_can_be_disabled(tmp_path):
    on = make_llm(Settings(db_path=str(tmp_path / "db"), llm_log_file=str(tmp_path / "x.jsonl")))
    off = make_llm(Settings(db_path=str(tmp_path / "db"), llm_log_enabled=False))
    assert on.call_log is not None and off.call_log is None
    llm = LLM(Scripted([json.dumps(SCOPE)]), call_log=None)
    await llm.json(Scope, "s", "u")  # no log, no error


async def test_unwritable_log_does_not_break_calls(tmp_path):
    llm = LLM(Scripted([json.dumps(SCOPE)]), call_log=CallLog(tmp_path / "missing-dir" / "x.jsonl"))
    assert (await llm.json(Scope, "s", "u")).title == "t"


def test_summary_and_cli(tmp_path, capsys, monkeypatch):
    path = tmp_path / "calls.jsonl"
    log = CallLog(path)
    log.record(backend="b", model="m", call="Extraction", attempt=1, system="a b", user="c d e", output="f g")
    log.record(backend="b", model="m", call="Extraction", attempt=1, system="a b", user="c d e", output="f",
               cached=True)
    log.record(backend="b", model="m", call="Scope", attempt=1, system="a", user="b", error="bad")
    text = summarize(path)
    assert "Extraction" in text and "total" in text and "tokens ≈ 1.3 × words" in text
    row = next(l for l in text.splitlines() if l.startswith("Extraction")).split()
    assert row[1:6] == ["2", "1", "0", "10", "3"]  # calls, cached, errors, words in, words out
    monkeypatch.chdir(tmp_path)
    cli.main(["llm-log", str(path)])
    assert "Extraction" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no LLM call log"):
        cli.main(["llm-log", str(tmp_path / "nope.jsonl")])
