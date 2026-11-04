"""Per-paper analysis: structured extraction (with gap filling), lineage relations, entity normalisation."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from ..models import Extraction, Grouping, MissingField, Relations
from ..prompts import SYSTEM, extract_prompt, normalize_prompt, relate_prompt
from ..store import PaperRecord, pid
from .context import Ctx

ALL_FIELDS: list[str] = list(MissingField.__args__)  # type: ignore[attr-defined]
ENTITY_EDGES = ("proposes", "uses", "evaluates_on")
MAX_NORMALIZE = 400


def incoming_contexts(ctx: Ctx) -> dict[str, list[str]]:
    """Citation contexts in which included papers describe each cited paper."""
    included = {r.node for r in ctx.store.papers(ctx.project, "included")}
    out: dict[str, list[str]] = defaultdict(list)
    for e in ctx.store.edges(ctx.project, ["cites"]):
        if e["src"] in included:
            out[e["dst"]] += e["data"].get("contexts", [])
    return out


async def extract(ctx: Ctx, recs: list[PaperRecord], deep: bool = False) -> int:
    """Extract methods/datasets/limitations; fill gaps from full-text passages. Returns #papers done."""
    contexts = incoming_contexts(ctx)

    async def one(r: PaperRecord) -> None:
        cites = contexts.get(r.node, [])[:8]
        ex = await ctx.llm.json(Extraction, SYSTEM, extract_prompt(r.paper, cites))
        fields = ALL_FIELDS if deep else list(dict.fromkeys(ex.missing))
        if fields and ctx.finder:
            passages = await ctx.finder.find(r.paper, fields, ex.search_terms)
            if passages:
                ex = await ctx.llm.json(Extraction, SYSTEM, extract_prompt(r.paper, cites, passages))
        apply_extraction(ctx, r.paper.id, ex)

    results = await asyncio.gather(*(one(r) for r in recs), return_exceptions=True)
    failed = [(r, e) for r, e in zip(recs, results) if isinstance(e, BaseException)]
    for r, e in failed:
        ctx.log(f"extraction failed for {r.paper.title!r}: {e}")
    ctx.log(f"analysed {len(recs) - len(failed)} papers")
    return len(recs) - len(failed)


def apply_extraction(ctx: Ctx, paper_id: str, ex: Extraction) -> None:
    store, project, node = ctx.store, ctx.project, pid(paper_id)
    store.set_extraction(project, paper_id, ex)
    store.delete_edges(project, ENTITY_EDGES, src=[node])
    edges = []
    proposed = set()
    for m in ex.methods_proposed:
        n = store.upsert_entity(project, "method", m.name, m.description)
        proposed.add(n)
        edges.append((node, n, "proposes", None))
    for name in ex.methods_used:
        n = store.upsert_entity(project, "method", name)
        if n not in proposed:
            edges.append((node, n, "uses", None))
    for d in ex.datasets:
        edges.append((node, store.upsert_entity(project, "dataset", d.name), "evaluates_on", {"usage": d.usage}))
    store.add_edges(project, edges)


async def relate(ctx: Ctx) -> int:
    """Classify unjudged in-graph citations as builds-on (with typed change) or plain citation."""
    included = {r.paper.id: r for r in ctx.store.papers(ctx.project, "included")}
    pending: dict[str, list[dict]] = defaultdict(list)
    for e in ctx.store.edges(ctx.project, ["cites"]):
        s, d = e["src"][2:], e["dst"][2:]
        if s != d and s in included and d in included and not e["data"].get("judged"):
            pending[s].append(e)

    async def one(citing_id: str, edges: list[dict]) -> int:
        edges.sort(key=lambda e: (e["data"].get("influential", False), len(e["data"].get("contexts", []))),
                   reverse=True)
        batch = edges[:ctx.settings.max_refs_per_paper]
        citing = included[citing_id]
        cited = [(included[e["dst"][2:]].paper, _contribution(included[e["dst"][2:]]), e["data"]) for e in batch]
        res = await ctx.llm.json(Relations, SYSTEM, relate_prompt(citing.paper, _contribution(citing), cited))
        new = []
        for rel in res.relations:
            if rel.builds_on and 0 <= rel.ref < len(batch):
                new.append((batch[rel.ref]["src"], batch[rel.ref]["dst"], "builds_on",
                            {"change_type": rel.change_type or "other", "component": rel.component,
                             "summary": rel.summary}))
        ctx.store.add_edges(ctx.project, new + [(e["src"], e["dst"], "cites", {**e["data"], "judged": True})
                                                for e in edges])
        return len(new)

    results = await asyncio.gather(*(one(s, es) for s, es in pending.items()), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            ctx.log(f"relation analysis failed: {r}")
    total = sum(r for r in results if isinstance(r, int))
    if pending:
        ctx.log(f"found {total} builds-on relations across {len(pending)} citing papers")
    return total


def _contribution(r: PaperRecord) -> str | None:
    return r.extraction.contribution if r.extraction else None


async def normalize(ctx: Ctx) -> None:
    """Merge duplicate method/dataset entities (aliases, abbreviations) via one LLM call per kind."""
    degree = defaultdict(int)
    for e in ctx.store.edges(ctx.project, ENTITY_EDGES):
        degree[e["dst"]] += 1

    async def one(kind: str) -> int:
        ents = sorted(ctx.store.entities(ctx.project, kind), key=lambda e: -degree[e["node"]])[:MAX_NORMALIZE]
        if len(ents) < 2:
            return 0
        grouping = await ctx.llm.json(Grouping, SYSTEM,
                                      normalize_prompt(kind, [(e["name"], e["description"]) for e in ents]))
        merged, consumed = 0, set()
        for g in grouping.groups:
            members = [ents[i] for i in dict.fromkeys(g.members) if 0 <= i < len(ents) and i not in consumed]
            consumed.update(g.members)
            if len(members) < 2:
                continue
            members.sort(key=lambda e: -degree[e["node"]])
            ctx.store.merge_entities(ctx.project, members[0]["node"], [m["node"] for m in members[1:]], g.canonical)
            merged += len(members) - 1
        return merged

    results = await asyncio.gather(one("method"), one("dataset"), return_exceptions=True)
    for kind, r in zip(("method", "dataset"), results):
        if isinstance(r, BaseException):
            ctx.log(f"{kind} normalisation failed: {r}")
        elif r:
            ctx.log(f"merged {r} duplicate {kind} entities")
