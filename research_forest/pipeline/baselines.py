"""Baseline relations: which methods a paper compares against, and which papers propose them.

Extraction and gap filling return ``baselines`` (method names from the abstract and, mainly,
the main results tables). Here each name is resolved to a paper, without an LLM, in order of
reliability:

1. **table citation**: the table cell cites a bibliography entry ("Linformer [84]"), whose
   title or arXiv id matches one of the paper's references;
2. **proposed in graph**: a graph paper proposes a method of that name;
3. **reference title**: one of the paper's references is titled "<Name>: ..." or "<Name>".

Resolved baselines become ``compares_with`` edges paper -> paper (also to papers not yet in the
graph: those are discovery candidates); unresolved ones point to a method entity. Linking runs on
every ``process()``, so baselines resolve to papers that join the graph later.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..models import Paper
from ..store import entity_key, pid
from . import discover
from .context import Ctx

COMPARES = "compares_with"
MIN_KEY = 3  # shorter names ("T5" excepted by equality) are too ambiguous for containment matching


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_head(title: str) -> str:
    """'Linformer: Self-Attention with Linear Complexity' -> 'linformer'."""
    return entity_key(re.split(r"[:—–]| - ", title, maxsplit=1)[0])


def match_reference(title: str, arxiv: str | None, refs: list[Paper]) -> Paper | None:
    """The reference a bibliography entry denotes: same arXiv id, else (nearly) the same title."""
    if arxiv:
        for p in refs:
            if p.arxiv_id and p.arxiv_id.split("v")[0] == arxiv:
                return p
    want = _tokens(title)
    best, best_score = None, 0.0
    for p in refs:
        have = _tokens(p.title)
        if len(have) < 2:
            continue
        score = len(have & want) / len(have | want)
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 0.8 else None


def _label_matches(name: str, label: str) -> bool:
    key, lkey = entity_key(name), entity_key(label)
    return key == lkey or (len(key) > MIN_KEY and key in lkey)


def resolve(name: str, refs: list[Paper], table_refs: list[dict], proposers: dict[str, str],
            self_node: str) -> tuple[str | None, str]:
    """(paper node, how) for a baseline name, or (None, "unresolved")."""
    key = entity_key(name)
    for tr in table_refs:
        if _label_matches(name, tr.get("label", "")):
            if (p := match_reference(tr.get("title", ""), tr.get("arxiv"), refs)) and pid(p.id) != self_node:
                return pid(p.id), "table citation"
    if (node := proposers.get(key)) and node != self_node:
        return node, "proposed in graph"
    if len(key) >= MIN_KEY:
        for p in refs:
            if title_head(p.title) == key and pid(p.id) != self_node:
                return pid(p.id), "reference title"
    return None, "unresolved"


def link(ctx: Ctx) -> dict[str, int]:
    """Recompute ``compares_with`` edges for all included papers. Returns counts by resolution."""
    store, project = ctx.store, ctx.project
    included = [r for r in store.papers(project, "included") if r.extraction and r.extraction.baselines]
    nodes = {r.node for r in store.papers(project, "included")}
    references: dict[str, list[str]] = defaultdict(list)
    for e in store.edges(project, ["cites"]):
        if e["src"] in nodes:
            references[e["src"]].append(e["dst"][2:])
    known = {p.paper.id: p.paper for p in store.papers(project, ids={i for ids in references.values() for i in ids})}
    names = {e["node"]: e["name"] for e in store.entities(project, "method")}
    proposers: dict[str, str] = {}
    for e in store.edges(project, ["proposes"]):
        if e["src"] in nodes and e["dst"] in names:
            proposers.setdefault(entity_key(names[e["dst"]]), e["src"])

    store.delete_edges(project, [COMPARES], src=list(nodes))
    edges, counts = [], Counter()
    for r in included:
        refs = [known[i] for i in references.get(r.node, []) if i in known]
        table_refs = store.meta(project, f"table_refs:{r.paper.id}", [])
        for name in dict.fromkeys(r.extraction.baselines):
            target, how = resolve(name, refs, table_refs, proposers, r.node)
            if target is None:
                target = store.upsert_entity(project, "method", name)
            counts[how] += 1
            counts["in graph"] += target in nodes
            edges.append((r.node, target, COMPARES, {"name": name, "via": how}))
    store.add_edges(project, edges)
    if edges:
        papers = len(edges) - counts["unresolved"]
        ctx.log(f"Linked {len(edges)} baselines: {papers} to papers ({counts['in graph']} in the graph, "
                f"{counts['table citation']} via table citations), {counts['unresolved']} to method names only")
    return dict(counts)


def candidates(ctx: Ctx) -> Counter:
    """Papers outside the graph that graph papers compare against, by number of such papers."""
    store, project = ctx.store, ctx.project
    included = {r.node for r in store.papers(project, "included")}
    excluded = {r.node for r in store.papers(project, "excluded")}
    return Counter(e["dst"][2:] for e in store.edges(project, [COMPARES])
                   if e["src"] in included and e["dst"].startswith("p:")
                   and e["dst"] not in included and e["dst"] not in excluded)


async def expand(ctx: Ctx, limit: int) -> list[str]:
    """Screen and add the papers most often used as baselines by graph papers."""
    if limit <= 0:
        return []
    pool = [i for i, _ in candidates(ctx).most_common(3 * limit)]
    if not pool:
        ctx.log("No baseline papers outside the graph")
        return []
    ctx.log(f"Found {len(pool)} papers that graph papers compare against but the graph lacks")
    await discover.screen(ctx, pool)
    return discover.select(ctx, limit, pool)
