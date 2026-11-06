"""Finding papers: scoping, search, LLM screening, ranking/selection and citation snowballing."""

from __future__ import annotations

import asyncio
import datetime
import math
from collections import Counter
from typing import Iterable, Literal

from ..models import MAX_REFINE_LIMIT, Paper, Scope, Screening
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
    ctx.work(len(queries) + len(seeds), f"Searching Semantic Scholar: {len(queries)} queries"
             + (f" and {len(seeds)} landmark titles" if seeds else ""))

    async def ticked(coro):
        try:
            return await coro
        finally:
            ctx.tick()

    results = await asyncio.gather(*(ticked(ctx.s2.search(q, per_query, year_min)) for q in queries),
                                   *(ticked(ctx.s2.match_title(t)) for t in seeds), return_exceptions=True)
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
    unique = len({p.id for p in papers})
    ctx.log(f"Search found {unique} papers ({len(new)} new); matched {len(seed_ids)}/{len(seeds)} landmark titles"
            if seeds else f"Search found {unique} papers ({len(new)} new)")
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
    batches = chunks(recs, SCREEN_BATCH)
    ctx.work(len(recs), f"Rating relevance of {len(recs)} candidates ({len(batches)} LLM batches)")

    async def run(batch):
        try:
            res = await ctx.llm.json(Screening, SYSTEM, screen_prompt(scope, [r.paper for r in batch]))
        finally:
            ctx.tick(len(batch))
        scores = {r.paper.id: 0 for r in batch}
        for item in res.items:
            if 0 <= item.ref < len(batch):
                scores[batch[item.ref].paper.id] = max(0, min(3, item.relevance))
        return scores

    results = await asyncio.gather(*(run(b) for b in batches), return_exceptions=True)
    scores: dict[str, int] = {}
    for res in results:
        if isinstance(res, BaseException):
            ctx.log(f"A screening batch failed and was skipped: {res}")
        else:
            scores.update(res)
    ctx.store.set_relevance(ctx.project, scores)
    hist = Counter(scores.values())
    ctx.log(f"Rated {len(scores)} papers: {hist[3]} core, {hist[2]} relevant, {hist[1]} tangential, "
            f"{hist[0]} off-topic")


UNKNOWN_AGE = 5  # years assumed when a paper has no publication year


def _age(paper: Paper, current_year: int) -> int:
    return max(1, current_year - paper.year + 1 if paper.year else UNKNOWN_AGE)


def citations_per_year(paper: Paper, current_year: int) -> float:
    """Citation velocity: total citations divided by years since publication (at least one).

    Ranking by raw citation counts favours old papers; a 2024 paper with 300 citations is
    usually more notable than a 2014 paper with 1,000.
    """
    return paper.citation_count / _age(paper, current_year)


def influential_per_year(paper: Paper, current_year: int) -> float:
    """Highly influential citations (Semantic Scholar's classifier) per year since publication."""
    return paper.influential_count / _age(paper, current_year)


def rank_score(paper: Paper, relevance: int, current_year: int) -> float:
    """Relevance first; then citation velocity, where influential citations (papers that use the
    work heavily) count more than mentions in passing."""
    return (relevance * 10 + 1.5 * math.log1p(citations_per_year(paper, current_year))
            + 2.0 * math.log1p(influential_per_year(paper, current_year)))


def select(ctx: Ctx, count: int, pool: Iterable[str] | None = None, boost: dict[str, float] | None = None,
           velocity: bool = True) -> list[str]:
    """Include up to ``count`` relevant candidates, ranked by relevance, citation velocity and in-graph citations.

    ``boost`` adds to a candidate's score (by paper id); ``velocity=False`` ignores citation
    counts, for recent work that hasn't been cited yet."""
    if count <= 0:
        return []
    pool = None if pool is None else set(pool)
    included = {r.node for r in ctx.store.papers(ctx.project, "included")}
    incoming = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["cites"]) if e["src"] in included)
    # being compared against in a graph paper's results is a stronger signal than a citation
    compared = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["compares_with"]) if e["src"] in included)
    cands = [r for r in ctx.store.papers(ctx.project, "candidate")
             if (r.relevance or 0) >= 2 and (pool is None or r.paper.id in pool)]
    year, boost = datetime.date.today().year, boost or {}
    cands.sort(key=lambda r: (rank_score(r.paper, r.relevance, year) if velocity else r.relevance * 10)
               + 3 * incoming[r.node] + 4 * compared[r.node] + boost.get(r.paper.id, 0), reverse=True)
    chosen = [r.paper.id for r in cands[:count]]
    ctx.store.update_papers(ctx.project, chosen, status="included")
    if chosen:
        total = len(included) + len(chosen)
        ctx.log(f"Added {len(chosen)} papers to the graph ({total} in total)")
    elif count:
        ctx.log("No further relevant candidates to add")
    return chosen


async def fetch_links(ctx: Ctx, ids: list[str],
                      direction: Literal["references", "citations"] = "references") -> list[str]:
    """Fetch citation links (with contexts) for papers; returns ids of the papers on the other end."""
    ctx.work(len(ids), f"Fetching {direction} with citation contexts for {len(ids)} papers")

    async def ticked(paper_id):
        try:
            return await ctx.s2.links(paper_id, direction)
        finally:
            ctx.tick()

    results = await asyncio.gather(*(ticked(i) for i in ids), return_exceptions=True)
    others: list[str] = []
    for paper_id, res in zip(ids, results):
        if isinstance(res, BaseException):
            ctx.log(f"Could not fetch {direction} of paper {paper_id}: {res}")
            continue
        ctx.store.add_papers(ctx.project, (p for p, _ in res))
        ctx.store.add_citations(ctx.project, (link for _, link in res))
        others += [p.id for p, _ in res]
        if direction == "references":
            ctx.store.update_papers(ctx.project, [paper_id], refs_fetched=1)
    others = list(dict.fromkeys(others))
    ctx.log(f"Fetched {direction}: {len(others)} linked papers")
    return others


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
    ctx.log(f"Found {len(frequent)} references cited by at least two graph papers")
    await screen(ctx, frequent)
    return select(ctx, limit, frequent)


MAX_EXPAND_SEEDS = 10
MAX_PAPERS_PER_OP = MAX_REFINE_LIMIT


async def expand(ctx: Ctx, ids: list[str], direction: str, limit: int) -> list[str]:
    """Grow the graph around seed papers. Fan-out is capped so one (possibly injected) op stays cheap."""
    ids = ids[:MAX_EXPAND_SEEDS]
    directions = ["references", "citations"] if direction == "both" else [direction or "references"]
    pool: list[str] = []
    for d in directions:
        pool += await fetch_links(ctx, ids, d)  # type: ignore[arg-type]
    await screen(ctx, pool)
    return select(ctx, min(MAX_PAPERS_PER_OP, limit * max(1, len(ids))), pool)


async def prune(ctx: Ctx, criterion: str) -> list[str]:
    batches = chunks(ctx.store.papers(ctx.project, "included"), SCREEN_BATCH)
    ctx.work(len(batches), f"Checking {sum(map(len, batches))} papers against: {criterion}")

    async def run(b):
        try:
            return await ctx.llm.json(Screening, SYSTEM, prune_prompt(criterion, [r.paper for r in b]))
        finally:
            ctx.tick()

    results = await asyncio.gather(*(run(b) for b in batches))
    drop = [b[i.ref].paper.id for b, res in zip(batches, results)
            for i in res.items if 0 <= i.ref < len(b) and i.relevance >= 2]
    ctx.store.exclude_papers(ctx.project, drop)
    ctx.log(f"Removed {len(drop)} papers matching {criterion!r}")
    return drop


def apply_filter(ctx: Ctx, year_min: int | None, year_max: int | None, min_citations: int | None) -> list[str]:
    drop = [r.paper.id for r in ctx.store.papers(ctx.project, "included")
            if (year_min and (r.paper.year or 0) < year_min) or (year_max and (r.paper.year or 9999) > year_max)
            or (min_citations is not None and r.paper.citation_count < min_citations)]
    ctx.store.exclude_papers(ctx.project, drop)
    bounds = ", ".join(x for x in (f"from {year_min}" if year_min else "", f"until {year_max}" if year_max else "",
                                   f"≥{min_citations} citations" if min_citations is not None else "") if x)
    ctx.log(f"Filter ({bounds or 'no bounds'}) removed {len(drop)} papers")
    return drop
