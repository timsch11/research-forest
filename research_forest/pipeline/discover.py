"""Finding papers: scoping, search, LLM screening, ranking/selection and citation snowballing."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from typing import Iterable, Literal

from ..models import Scope, Screening
from ..prompts import SYSTEM, prune_prompt, rescope_prompt, scope_prompt, screen_prompt
from .context import Ctx, chunks

SCREEN_BATCH = 25


async def make_scope(ctx: Ctx, prompt: str) -> Scope:
    return await ctx.llm.json(Scope, SYSTEM, scope_prompt(prompt))


async def rescope(ctx: Ctx, instruction: str) -> Scope:
    old = ctx.store.scope(ctx.project)
    new = await ctx.llm.json(Scope, SYSTEM, rescope_prompt(old, instruction))
    merged = new.model_copy(update={"search_queries": list(dict.fromkeys(old.search_queries + new.search_queries))})
    ctx.store.set_scope(ctx.project, merged)
    return new


async def search(ctx: Ctx, queries: list[str], seeds: Iterable[str] = (), per_query: int = 40,
                 year_min: int | None = None) -> list[str]:
    """Run keyword queries and title matches; store hits as candidates and return all hit ids."""
    seeds = list(seeds)
    results = await asyncio.gather(*(ctx.s2.search(q, per_query, year_min) for q in queries),
                                   *(ctx.s2.match_title(t) for t in seeds), return_exceptions=True)
    papers, seed_ids = [], []
    for i, res in enumerate(results):
        if isinstance(res, BaseException):
            ctx.log(f"search failed for {(queries + seeds)[i]!r}: {res}")
        elif i < len(queries):
            papers += res
        elif res:
            papers.append(res)
            seed_ids.append(res.id)
    new = ctx.store.add_papers(ctx.project, papers)
    ctx.store.set_relevance(ctx.project, {i: 3 for i in seed_ids if i in new})  # LLM-named landmark papers
    ctx.log(f"search: {len(papers)} hits, {len(new)} new candidates")
    return list(dict.fromkeys(p.id for p in papers))


async def screen(ctx: Ctx, ids: Iterable[str] | None = None, force: bool = False) -> None:
    """LLM relevance screening of unscreened candidates (or of ``ids`` regardless of state when ``force``)."""
    ids = None if ids is None else set(ids)
    recs = [r for r in ctx.store.papers(ctx.project, ids=ids)
            if r.status != "excluded" and (force or (r.status == "candidate" and r.relevance is None))]
    recs = recs[:ctx.settings.max_candidates]
    if not recs:
        return
    scope = ctx.store.scope(ctx.project)

    async def run(batch):
        res = await ctx.llm.json(Screening, SYSTEM, screen_prompt(scope, [r.paper for r in batch]))
        scores = {r.paper.id: 0 for r in batch}
        for item in res.items:
            if 0 <= item.ref < len(batch):
                scores[batch[item.ref].paper.id] = max(0, min(3, item.relevance))
        return scores

    results = await asyncio.gather(*(run(b) for b in chunks(recs, SCREEN_BATCH)), return_exceptions=True)
    scores: dict[str, int] = {}
    for res in results:
        if isinstance(res, BaseException):
            ctx.log(f"screening batch failed: {res}")
        else:
            scores.update(res)
    ctx.store.set_relevance(ctx.project, scores)
    ctx.log(f"screened {len(scores)} papers, {sum(s >= 2 for s in scores.values())} relevant")


def select(ctx: Ctx, count: int, pool: Iterable[str] | None = None) -> list[str]:
    """Include up to ``count`` relevant candidates, ranked by relevance, citations and in-graph citations."""
    if count <= 0:
        return []
    pool = None if pool is None else set(pool)
    included = {r.node for r in ctx.store.papers(ctx.project, "included")}
    incoming = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["cites"]) if e["src"] in included)
    cands = [r for r in ctx.store.papers(ctx.project, "candidate")
             if (r.relevance or 0) >= 2 and (pool is None or r.paper.id in pool)]
    cands.sort(key=lambda r: r.relevance * 10 + 1.5 * math.log1p(r.paper.citation_count) + 3 * incoming[r.node],
               reverse=True)
    chosen = [r.paper.id for r in cands[:count]]
    ctx.store.update_papers(ctx.project, chosen, status="included")
    if chosen:
        ctx.log(f"added {len(chosen)} papers to the graph")
    return chosen


async def fetch_links(ctx: Ctx, ids: list[str],
                      direction: Literal["references", "citations"] = "references") -> list[str]:
    """Fetch citation links (with contexts) for papers; returns ids of the papers on the other end."""
    results = await asyncio.gather(*(ctx.s2.links(i, direction) for i in ids), return_exceptions=True)
    others: list[str] = []
    for paper_id, res in zip(ids, results):
        if isinstance(res, BaseException):
            ctx.log(f"fetching {direction} of {paper_id} failed: {res}")
            continue
        ctx.store.add_papers(ctx.project, (p for p, _ in res))
        ctx.store.add_citations(ctx.project, (link for _, link in res))
        others += [p.id for p, _ in res]
        if direction == "references":
            ctx.store.update_papers(ctx.project, [paper_id], refs_fetched=1)
    return list(dict.fromkeys(others))


async def refresh_metadata(ctx: Ctx, ids: list[str]) -> None:
    """Batch-fetch full metadata (TL;DR, open-access PDF) for papers first seen via citation lists."""
    if ids:
        ctx.store.add_papers(ctx.project, await ctx.s2.batch(ids))


async def snowball(ctx: Ctx, limit: int) -> list[str]:
    """Screen and add frequently co-cited references of the current graph."""
    included = {r.node for r in ctx.store.papers(ctx.project, "included")}
    counts = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["cites"])
                     if e["src"] in included and e["dst"] not in included)
    frequent = [n[2:] for n, c in counts.most_common(3 * limit) if c >= 2]
    await screen(ctx, frequent)
    return select(ctx, limit, frequent)


async def expand(ctx: Ctx, ids: list[str], direction: str, limit: int) -> list[str]:
    directions = ["references", "citations"] if direction == "both" else [direction or "references"]
    pool: list[str] = []
    for d in directions:
        pool += await fetch_links(ctx, ids, d)  # type: ignore[arg-type]
    await screen(ctx, pool)
    return select(ctx, limit * max(1, len(ids)), pool)


async def prune(ctx: Ctx, criterion: str) -> list[str]:
    batches = chunks(ctx.store.papers(ctx.project, "included"), SCREEN_BATCH)
    results = await asyncio.gather(*(ctx.llm.json(Screening, SYSTEM, prune_prompt(criterion, [r.paper for r in b]))
                                     for b in batches))
    drop = [b[i.ref].paper.id for b, res in zip(batches, results)
            for i in res.items if 0 <= i.ref < len(b) and i.relevance >= 2]
    ctx.store.exclude_papers(ctx.project, drop)
    ctx.log(f"pruned {len(drop)} papers matching {criterion!r}")
    return drop


def apply_filter(ctx: Ctx, year_min: int | None, year_max: int | None, min_citations: int | None) -> list[str]:
    drop = [r.paper.id for r in ctx.store.papers(ctx.project, "included")
            if (year_min and (r.paper.year or 0) < year_min) or (year_max and (r.paper.year or 9999) > year_max)
            or (min_citations is not None and r.paper.citation_count < min_citations)]
    ctx.store.exclude_papers(ctx.project, drop)
    ctx.log(f"filter removed {len(drop)} papers")
    return drop
