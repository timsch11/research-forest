"""Per-paper analysis: structured extraction (with gap filling), lineage relations, entity normalisation.

Token budget notes: each paper's abstract and citation contexts are sent exactly once
(gap filling sends only the passages); each citation pair is judged once, grouped so that the
fewest prompts are needed; normalisation only looks at entities that are new since the last
run, plus the existing names that could plausibly be their aliases.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict

from ..models import Extraction, GapFill, Grouping, MissingField, Relations
from ..prompts import SYSTEM, extract_prompt, gapfill_prompt, normalize_prompt, relate_prompt
from ..safety import looks_like_injection
from ..store import PaperRecord, entity_key, pid
from .context import Ctx

ALL_FIELDS: list[str] = list(MissingField.__args__)  # type: ignore[attr-defined]
ENTITY_EDGES = ("proposes", "uses", "evaluates_on")
MAX_NORMALIZE = 400
NORMALIZED_UPTO = "normalized_upto"  # meta key: highest entity id already normalised


def incoming_contexts(ctx: Ctx) -> dict[str, list[str]]:
    """Citation contexts in which included papers describe each cited paper."""
    included = {r.node for r in ctx.store.papers(ctx.project, "included")}
    out: dict[str, list[str]] = defaultdict(list)
    for e in ctx.store.edges(ctx.project, ["cites"]):
        if e["src"] in included:
            out[e["dst"]] += e["data"].get("contexts", [])
    return out


# ------------------------------------------------------------- extraction


def _union(items: list, extra: list, key) -> list:
    seen, out = set(), []
    for x in [*items, *extra]:
        if (k := key(x)) not in seen:
            seen.add(k)
            out.append(x)
    return out


def merge_gapfill(ex: Extraction, fill: GapFill, filled: list[str]) -> Extraction:
    """Add gap-fill items to an extraction, skipping duplicates (caps are re-applied on validation)."""
    dump = lambda xs: [x.model_dump() if hasattr(x, "model_dump") else x for x in xs]  # noqa: E731
    return Extraction.model_validate({
        **ex.model_dump(),
        "methods_used": _union(ex.methods_used, fill.methods_used, entity_key),
        "datasets": dump(_union(ex.datasets, fill.datasets, lambda d: entity_key(d.name))),
        "limitations": dump(_union(ex.limitations, fill.limitations, lambda l: l.text.lower())),
        "future_work": _union(ex.future_work, fill.future_work, str.lower),
        "missing": [m for m in ex.missing if m not in filled],
    })


async def extract(ctx: Ctx, recs: list[PaperRecord], deep: bool = False) -> int:
    """Extract methods/datasets/limitations; fill gaps from full-text passages. Returns #papers done.

    ``deep`` (the *deepen* refinement) reuses an existing extraction and only adds full-text
    findings for every field.
    """
    contexts = incoming_contexts(ctx)
    ctx.work(len(recs), (f"Re-analysing {len(recs)} papers with full-text passages" if deep else
                         f"Extracting methods, datasets and limitations from {len(recs)} papers"))
    gap_filled = 0

    async def one(r: PaperRecord) -> None:
        try:
            await analyse(r)
        finally:
            ctx.tick()

    async def analyse(r: PaperRecord) -> None:
        nonlocal gap_filled
        cites = contexts.get(r.node, [])
        if deep and r.extraction is not None:
            ex = r.extraction
        else:
            if any(looks_like_injection(t) for t in (r.paper.title, r.paper.abstract, r.paper.tldr, *cites[:8])):
                ctx.log(f"warning: instruction-like text in metadata of {r.paper.title[:80]!r} (treated as data)")
            ex = await ctx.llm.json(Extraction, SYSTEM, extract_prompt(r.paper, cites))
        fields = ALL_FIELDS if deep else list(dict.fromkeys(ex.missing))
        if fields and ctx.finder:
            passages = await ctx.finder.find(r.paper, fields, ex.search_terms)
            if any(looks_like_injection(t) for texts in passages.values() for t in texts):
                ctx.log(f"warning: instruction-like text in full text of {r.paper.title[:80]!r} (treated as data)")
            if passages:
                fill = await ctx.llm.json(GapFill, SYSTEM, gapfill_prompt(r.paper, ex.contribution, passages))
                ex = merge_gapfill(ex, fill, list(passages))
                gap_filled += 1
        apply_extraction(ctx, r.paper.id, ex)

    results = await asyncio.gather(*(one(r) for r in recs), return_exceptions=True)
    failed = [(r, e) for r, e in zip(recs, results) if isinstance(e, BaseException)]
    for r, e in failed:
        ctx.log(f"Could not analyse {r.paper.title[:80]!r}: {e}")
    ctx.log(f"Analysed {len(recs) - len(failed)}/{len(recs)} papers"
            + (f"; {gap_filled} were completed from full-text passages" if gap_filled else ""))
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


# ---------------------------------------------------------------- lineage


def _group(pairs: list[dict], key) -> dict[tuple[str, bool], list[dict]]:
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for e in pairs:
        groups[key(e)].append(e)
    return groups


def _greedy_hubs(pairs: list[dict]) -> dict[tuple[str, bool], list[dict]]:
    """Repeatedly anchor on the endpoint with the most unassigned pairs (greedy set cover)."""
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    left = list(pairs)
    while left:
        load = defaultdict(int)
        for e in left:
            load[(e["src"], True)] += 1
            load[(e["dst"], False)] += 1
        (node, citing), _ = max(load.items(), key=lambda kv: (kv[1], kv[0][1]))
        side = "src" if citing else "dst"
        groups[(node[2:], citing)] += [e for e in left if e[side] == node]
        left = [e for e in left if e[side] != node]
    return groups


def plan_relation_prompts(pairs: list[dict], cap: int) -> list[tuple[str, bool, list[dict]]]:
    """Group pending citation pairs into as few prompts (of at most ``cap`` pairs) as possible.

    Every prompt sends its anchor paper's context once for all its pairs, so fewer prompts means
    less repeated context. Three groupings are compared: by citing paper (best for a fresh
    build), by cited paper, and greedy hubs (best after adding a few papers that both cite and
    are cited by older ones). Returns (anchor paper id, anchor_is_citing, pairs) tuples.
    """
    candidates = [_group(pairs, lambda e: (e["src"][2:], True)),
                  _group(pairs, lambda e: (e["dst"][2:], False)),
                  _greedy_hubs(pairs)]
    cost = lambda groups: sum(-(-len(g) // cap) for g in groups.values())  # noqa: E731 - ceil division
    best = min(candidates, key=cost)  # first wins ties: by citing paper
    return [(anchor, citing, group[i:i + cap]) for (anchor, citing), group in best.items()
            for i in range(0, len(group), cap)]


async def relate(ctx: Ctx) -> int:
    """Classify unjudged in-graph citations as builds-on (with typed change) or plain citation."""
    included = {r.paper.id: r for r in ctx.store.papers(ctx.project, "included")}
    cap = ctx.settings.max_refs_per_paper
    per_citing: dict[str, list[dict]] = defaultdict(list)
    for e in ctx.store.edges(ctx.project, ["cites"]):
        s, d = e["src"][2:], e["dst"][2:]
        if s != d and s in included and d in included and not e["data"].get("judged"):
            per_citing[s].append(e)

    # Judge at most `cap` references per citing paper (the strongest evidence first); the rest
    # are marked judged without a prompt.
    pairs, skipped = [], []
    for edges in per_citing.values():
        edges.sort(key=lambda e: (e["data"].get("influential", False), len(e["data"].get("contexts", []))),
                   reverse=True)
        pairs += edges[:cap]
        skipped += edges[cap:]

    async def one(anchor_id: str, anchor_is_citing: bool, batch: list[dict]) -> int:
        try:
            return await judge(anchor_id, anchor_is_citing, batch)
        finally:
            ctx.tick()

    async def judge(anchor_id: str, anchor_is_citing: bool, batch: list[dict]) -> int:
        other = (lambda e: included[e["dst"][2:]]) if anchor_is_citing else (lambda e: included[e["src"][2:]])
        others = [(other(e).paper, _contribution(other(e)), e["data"]) for e in batch]
        anchor = included[anchor_id]
        res = await ctx.llm.json(Relations, SYSTEM, relate_prompt(anchor.paper, _contribution(anchor), others,
                                                                   anchor_is_citing))
        new = []
        for rel in res.relations:
            if rel.builds_on and 0 <= rel.ref < len(batch):
                new.append((batch[rel.ref]["src"], batch[rel.ref]["dst"], "builds_on",
                            {"change_type": rel.change_type or "other", "component": rel.component,
                             "summary": rel.summary}))
        ctx.store.add_edges(ctx.project, new + [(e["src"], e["dst"], "cites", {**e["data"], "judged": True})
                                                for e in batch])
        return len(new)

    prompts = plan_relation_prompts(pairs, cap)
    ctx.work(len(prompts), f"Judging {len(pairs)} citations between graph papers ({len(prompts)} LLM calls)")
    results = await asyncio.gather(*(one(*p) for p in prompts), return_exceptions=True)
    ctx.store.add_edges(ctx.project, [(e["src"], e["dst"], "cites", {**e["data"], "judged": True}) for e in skipped])
    for r in results:
        if isinstance(r, BaseException):
            ctx.log(f"A lineage batch failed and was skipped: {r}")
    total = sum(r for r in results if isinstance(r, int))
    if pairs:
        ctx.log(f"Found {total} builds-on relations among {len(pairs)} citations")
    else:
        ctx.log("No new citations between graph papers to judge")
    return total


def _contribution(r: PaperRecord) -> str | None:
    return r.extraction.contribution if r.extraction else None


# ---------------------------------------------------------- normalisation


def _alias_tokens(name: str) -> set[str]:
    """Cheap blocking keys: words, number runs, initials and a 4-char key prefix."""
    low = name.lower()
    words = re.findall(r"[a-z]+", low)
    tokens = {w for w in words if len(w) >= 3} | set(re.findall(r"\d{2,}", low))
    if len(words) >= 2:
        tokens.add("".join(w[0] for w in words))
    if len(key := entity_key(name)) >= 4:
        tokens.add(key[:4])
    return tokens


def normalisation_batch(ents: list[dict], upto: int, degree: dict[str, int]) -> list[dict]:
    """Entities to send: all on the first run; afterwards only new ones and plausible aliases."""
    new = [e for e in ents if e["id"] > upto]
    if not new:
        return []
    if upto == 0:
        batch = ents
    else:
        keys = set().union(*(_alias_tokens(e["name"]) for e in new))
        batch = new + [e for e in ents if e["id"] <= upto and _alias_tokens(e["name"]) & keys]
    return sorted(batch, key=lambda e: -degree.get(e["node"], 0))[:MAX_NORMALIZE] if len(batch) >= 2 else []


async def normalize(ctx: Ctx) -> None:
    """Merge duplicate method/dataset entities (aliases, abbreviations) via one LLM call per kind."""
    degree = defaultdict(int)
    for e in ctx.store.edges(ctx.project, ENTITY_EDGES):
        degree[e["dst"]] += 1
    upto = ctx.store.meta(ctx.project, NORMALIZED_UPTO, 0)
    high = max((e["id"] for e in ctx.store.entities(ctx.project)), default=0)

    async def one(kind: str) -> int:
        ents = normalisation_batch(ctx.store.entities(ctx.project, kind), upto, degree)
        if not ents:
            return 0
        grouping = await ctx.llm.json(Grouping, SYSTEM, normalize_prompt(kind, [e["name"] for e in ents]))
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

    ctx.work(2, "Merging duplicate method and dataset names")

    async def ticked(kind: str) -> int:
        try:
            return await one(kind)
        finally:
            ctx.tick()

    results = await asyncio.gather(ticked("method"), ticked("dataset"), return_exceptions=True)
    failed = False
    for kind, r in zip(("method", "dataset"), results):
        if isinstance(r, BaseException):
            failed = True
            ctx.log(f"Merging duplicate {kind}s failed: {r}")
        else:
            ctx.log(f"Merged {r} duplicate {kind} names" if r else f"No duplicate {kind} names found")
    if not failed:
        ctx.store.set_meta(ctx.project, NORMALIZED_UPTO, high)
