"""Recent follow-up work: papers that cite graph papers *influentially* but aren't visible yet.

Search and snowballing favour papers that are already cited. Semantic Scholar marks a
citation as highly influential when the citing paper uses the cited work heavily (it mentions
it often, and in its method or experiments). Citations come newest first, so one request per
seed paper yields its recent influential citers, typically with no citations of their own yet.

Candidates that influentially cite several graph papers rank first. They are screened against
the scope like any other candidate (influential citers include other fields that merely reuse
a method), and ranked without citation counts, which recent work doesn't have.
"""

from __future__ import annotations

import asyncio
import datetime
import math
from collections import Counter, defaultdict

from ..models import CitationLink, Paper
from ..store import FOLLOWUPS
from . import discover
from .context import Ctx

CITATIONS_PER_SEED = 1000  # newest citations scanned per seed paper (one request)


def seeds(ctx: Ctx, count: int) -> list[str]:
    """The graph's most important papers: built upon and cited within the graph, then influential overall."""
    included = {r.node: r.paper for r in ctx.store.papers(ctx.project, "included")}
    built = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["builds_on"]) if e["src"] in included)
    cited = Counter(e["dst"] for e in ctx.store.edges(ctx.project, ["cites"]) if e["src"] in included)
    ranked = sorted(included, key=lambda n: 3 * built[n] + cited[n] + math.log1p(included[n].influential_count),
                    reverse=True)
    return [n[2:] for n in ranked[:count]]


async def find(ctx: Ctx, limit: int, seed_ids: list[str] | None = None, year_min: int | str | None = None) -> list[str]:
    """Add up to ``limit`` recent papers that influentially cite graph papers (``seed_ids``, by default
    the key papers). ``year_min`` (a year or a YYYY-MM-DD date) defaults to the last
    ``followup_years``. Returns the added ids."""
    if limit <= 0:
        return []
    seed_ids = seed_ids or seeds(ctx, ctx.settings.followup_seeds)
    if not seed_ids:
        return []
    year_min = year_min or datetime.date.today().year - max(1, ctx.settings.followup_years) + 1
    ctx.work(len(seed_ids), f"Looking for recent work (since {year_min}) that builds on {len(seed_ids)} key papers")

    async def scan(paper_id: str):
        try:
            return await ctx.s2.links(paper_id, "citations", CITATIONS_PER_SEED, year_min=year_min,
                                      influential_only=True)
        finally:
            ctx.tick()

    results = await asyncio.gather(*(scan(i) for i in seed_ids), return_exceptions=True)
    known = {r.paper.id for r in ctx.store.papers(ctx.project) if r.status != "candidate"}
    papers: dict[str, Paper] = {}
    links: dict[str, list[CitationLink]] = defaultdict(list)
    for seed, res in zip(seed_ids, results):
        if isinstance(res, BaseException):
            ctx.log(f"Could not fetch citations of paper {seed}: {res}")
            continue
        for paper, link in res:
            if paper.id not in known:
                papers[paper.id] = paper
                links[paper.id].append(link)
    if not papers:
        ctx.log("No recent influential citations outside the graph")
        return []

    # Following up several graph papers is the strongest signal; then how much the paper discusses
    # them (citation contexts), then recency.
    def score(pid: str) -> tuple:
        return (len(links[pid]), sum(len(l.contexts) for l in links[pid]), papers[pid].year or 0)

    pool = sorted(papers, key=score, reverse=True)[:4 * limit]
    ctx.store.add_papers(ctx.project, (papers[i] for i in pool))
    ctx.store.add_citations(ctx.project, (l for i in pool for l in links[i]))
    multi = sum(len(links[i]) > 1 for i in papers)
    ctx.log(f"Found {len(papers)} recent papers that cite graph papers influentially ({multi} cite several); "
            f"screening the {len(pool)} most connected")
    await discover.screen(ctx, pool)
    chosen = discover.select(ctx, limit, pool, boost={i: 3.0 * (len(links[i]) - 1) for i in pool}, velocity=False)
    if chosen:
        record = ctx.store.meta(ctx.project, FOLLOWUPS, {})
        record.update({i: sorted({l.cited for l in links[i]}) for i in chosen})
        ctx.store.set_meta(ctx.project, FOLLOWUPS, record)
    return chosen
