"""Shared fixtures: a deterministic fake LLM backend and a mocked Semantic Scholar API."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from research_forest.config import Settings
from research_forest.engine import Forest
from research_forest.llm import LLM, Backend
from research_forest.sources.semantic_scholar import SemanticScholar

# Synthetic corpus: id -> (title, year, citations, references)
CORPUS = {
    "T": ("Attention Is All You Need", 2017, 90000, []),
    "S": ("Sparse Transformer: Generating Long Sequences", 2019, 1500, ["T"]),
    "L": ("Linformer: Self-Attention with Linear Complexity", 2020, 1800, ["T", "S"]),
    "P": ("Performer: Rethinking Attention with Kernels", 2020, 1700, ["T", "L"]),
    "F": ("FlashAttention: Fast and Memory-Efficient Exact Attention", 2022, 2500, ["T", "L", "P"]),
    "F2": ("FlashAttention-2: Faster Attention with Better Parallelism", 2023, 900, ["F", "T"]),
    "M": ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023, 2000, ["T", "F", "P"]),
    "C": ("A Survey of Cooking Recipes", 2021, 5, []),
}


def s2_paper(pid: str, full: bool = True) -> dict:
    title, year, cites, _ = CORPUS[pid]
    d = {"paperId": pid, "title": title, "abstract": f"We study {title.lower()}.", "year": year,
         "venue": "NeurIPS", "authors": [{"name": "A. Author"}], "citationCount": cites,
         "url": f"https://s2/{pid}", "externalIds": {"ArXiv": f"2000.{pid}"}}
    if full:
        d |= {"tldr": {"text": f"TLDR of {title}"}, "openAccessPdf": None}
    return d


def s2_handler(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, request.url.params
    if path.endswith("/paper/search"):
        q = params["query"].lower()
        hits = [s2_paper(p) for p, (t, *_rest) in CORPUS.items() if any(w in t.lower() for w in q.split())]
        return httpx.Response(200, json={"data": hits})
    if path.endswith("/paper/search/match"):
        q = params["query"].lower()
        hits = [s2_paper(p) for p, (t, *_rest) in CORPUS.items() if t.lower() == q]
        return httpx.Response(200 if hits else 404, json={"data": hits[:1]})
    if path.endswith("/paper/batch"):
        ids = json.loads(request.content)["ids"]
        return httpx.Response(200, json=[s2_paper(i) if i in CORPUS else None for i in ids])
    if m := re.search(r"/paper/([^/]+)/(references|citations)$", path):
        pid, direction = m.groups()
        if direction == "references":
            rows = [{"citedPaper": s2_paper(r, False), "contexts": [f"{CORPUS[pid][0]} extends {CORPUS[r][0]}"],
                     "intents": ["methodology"], "isInfluential": True} for r in CORPUS[pid][3]]
        else:
            rows = [{"citingPaper": s2_paper(c, False), "contexts": [], "intents": [], "isInfluential": False}
                    for c, v in CORPUS.items() if pid in v[3]]
        return httpx.Response(200, json={"data": rows})
    if path.endswith("/snippet/search"):
        return httpx.Response(200, json={"data": [
            {"snippet": {"text": "A key limitation is the reliance on custom CUDA kernels."}}]})
    return httpx.Response(404)


class FakeBackend(Backend):
    """Answers each schema with deterministic, prompt-derived content."""

    name, model = "fake", "fake-1"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> str:
        self.calls.append((schema_name, user))
        refs = [int(i) for i in re.findall(r"^\[(\d+)\]", user, re.M)]
        handler = getattr(self, f"_{schema_name.lower()}")
        return json.dumps(handler(user, refs))

    def _scope(self, user, refs):
        return {"title": "Efficient attention", "summary": "Making attention cheaper.",
                "search_queries": ["attention", "linear", "survey"], "seed_papers": ["Attention Is All You Need"],
                "include_criteria": "efficient sequence models", "exclude_criteria": "cooking", "year_min": None}

    def _screening(self, user, refs):
        lines = dict(re.findall(r"^\[(\d+)\] (.*)$", user, re.M))
        if "REMOVE" in user:  # prune prompt: match papers before 2019
            return {"items": [{"ref": r, "relevance": 3 if "(2017)" in lines[str(r)] else 0} for r in refs]}
        return {"items": [{"ref": r, "relevance": 0 if "Cooking" in lines[str(r)] else 3} for r in refs]}

    def _extraction(self, user, refs):
        title = re.search(r"^Title: (.*)$", user, re.M).group(1)
        name = title.split(":")[0]
        gap_filled = "Full-text passages" in user
        return {"contribution": f"{name} makes attention efficient.",
                "methods_proposed": [{"name": name, "description": f"The {name} method"}],
                "methods_used": ["Transformer"] if "Attention Is All" not in title else [],
                "datasets": [{"name": "WikiText-103", "usage": "evaluation"},
                             {"name": "WikiText 103 (WT103)" if "Mamba" in title else "LRA", "usage": "evaluation"}],
                "limitations": [{"text": "Relies on custom CUDA kernels", "source": "stated"}] if gap_filled else
                               [{"text": "Quadratic memory in sequence length", "source": "inferred"}],
                "future_work": ["Scale to longer contexts"],
                "missing": ["limitations"] if name == "FlashAttention" and not gap_filled else [],
                "search_terms": ["kernel"]}

    def _relations(self, user, refs):
        return {"relations": [{"ref": r, "builds_on": True, "change_type": "efficiency",
                               "component": "attention", "summary": "Makes it faster"} for r in refs]}

    def _grouping(self, user, refs):
        lines = dict(re.findall(r"^\[(\d+)\] ([^—\n]*)", user, re.M))
        wiki = [int(i) for i, n in lines.items() if "wikitext" in n.lower().replace("-", "")]
        return {"groups": [{"canonical": "WikiText-103", "members": wiki}] if len(wiki) > 1 else []}

    def _synthesis(self, user, refs):
        return {"themes": [{"name": "Memory cost", "description": "Quadratic memory", "papers": refs[:3],
                            "addressed_by": refs[3:5]}],
                "questions": [{"question": "Can exact attention be linear-time?", "rationale": "Open",
                               "themes": [0], "papers": refs[:2]}]}

    def _refineplan(self, user, refs):
        command = re.search(r'refine the graph: "(.*)"', user).group(1)
        op = {"action": "filter", "query": None, "papers": [], "direction": None, "limit": None,
              "year_min": None, "year_max": None, "min_citations": None}
        if "2020" in command:
            op["year_min"] = 2020
        elif "remove old" in command:
            op |= {"action": "prune", "query": "papers from 2017"}
        return {"explanation": f"Handling: {command}", "ops": [op]}

    def _answer(self, user, refs):
        return {"answer": "FlashAttention is key [n0].", "nodes": ["n0"]}


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "test.forest.db"), target_papers=7, s2_rps=0,
                    fulltext_enabled=True, cache_llm=False)


@pytest.fixture
async def forest(settings, fake_backend):
    s2 = SemanticScholar(rps=0, transport=httpx.MockTransport(s2_handler))
    f = Forest(settings, llm=LLM(fake_backend, concurrency=4), s2=s2)
    yield f
    await f.aclose()
