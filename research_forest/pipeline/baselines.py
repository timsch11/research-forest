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

from ..models import CitationLink, Paper
from ..store import entity_key, pid
from . import discover
from .context import Ctx

COMPARES = "compares_with"
MIN_KEY = 3  # shorter names ("T5" excepted by equality) are too ambiguous for containment matching


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _stems(text: str) -> set[str]:
    """Tokens with a plural "s" removed, so "Performer" matches "Performers"."""
    return {t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t for t in _tokens(text)}


_QUALIFIERS = re.compile(r"\b(implementation|impl\.?|model|baseline|original|standard|vanilla|our|re-?implemented|"
                         r"reproduced|reported)\b", re.I)


def name_variants(name: str) -> list[str]:
    """Names to try for a baseline as written in a results table:
    'XLNet (K = 6)' -> XLNet; 'BiT (Big Transfer)' -> BiT, Big Transfer;
    'Megatron-LM implementation' -> Megatron-LM; 'Transformer (Vaswani et al., 2017)' -> Transformer."""
    variants = [name]
    outer = _clean(re.sub(r"\([^)]*\)|\[[^]]*\]", " ", name))
    variants.append(outer)
    variants.append(_clean(_QUALIFIERS.sub(" ", outer)))
    for inner in re.findall(r"\(([^)]*)\)", name):
        if re.search(r"[A-Za-z]{2}", inner) and not re.search(r"\d{4}|et al|=", inner):
            variants.append(_clean(inner))
    return [v for v in dict.fromkeys(variants) if len(entity_key(v)) >= 2]


def _clean(text: str) -> str:
    return " ".join(text.replace(" ,", ",").split()).strip(" ,;-")


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


def by_title(name: str, titles: list[str]) -> int | None:
    """Index of the one title that names ``name``: 'Name: ...' first, else the only title whose
    words contain all of the name's words ('Performer' -> 'Rethinking Attention with Performers')."""
    key = entity_key(name)
    if len(key) < MIN_KEY:
        return None
    heads = [i for i, t in enumerate(titles) if title_head(t) == key]
    if heads:
        return heads[0]
    words = _stems(name)
    if not words or (len(words) == 1 and len(key) < 5):  # single short words are too ambiguous
        return None
    hits = [i for i, t in enumerate(titles) if words <= _stems(t)]
    return hits[0] if len(hits) == 1 else None


def resolve(name: str, refs: list[Paper], table_refs: list[dict], proposers: dict[str, str],
            self_node: str) -> tuple[str | None, str]:
    """(paper node, how) for a baseline name, or (None, "unresolved")."""
    variants = name_variants(name)
    for variant in variants:
        for tr in table_refs:
            if _label_matches(variant, tr.get("label", "")):
                if (p := match_reference(tr.get("title", ""), tr.get("arxiv"), refs)) and pid(p.id) != self_node:
                    return pid(p.id), "table citation"
    for variant in variants:
        if (node := proposers.get(entity_key(variant))) and node != self_node:
            return node, "proposed in graph"
    candidates = [p for p in refs if pid(p.id) != self_node]
    for variant in variants:
        if (i := by_title(variant, [p.title for p in candidates])) is not None:
            return pid(candidates[i].id), "reference title"
    return None, "unresolved"


def bibliography_wanted(name: str, table_refs: list[dict], bibliography: list[dict]) -> dict | None:
    """The paper's own bibliography entry for an unresolved baseline (to look up), if any."""
    variants = name_variants(name)
    for variant in variants:
        for tr in table_refs:
            if _label_matches(variant, tr.get("label", "")):
                return {"title": tr.get("title", ""), "arxiv": tr.get("arxiv")}
    titles = [b["title"] for b in bibliography]
    for variant in variants:
        if (i := by_title(variant, titles)) is not None:
            return bibliography[i]
    return None


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


MAX_LOOKUPS = 25  # bibliography lookups on Semantic Scholar per run
LOOKUPS = "bibliography_lookups"  # meta key: title -> paper id (or None), so nothing is looked up twice


async def lookup_bibliography(ctx: Ctx, limit: int = MAX_LOOKUPS) -> int:
    """Find papers for baselines that no known reference explains, via the paper's own bibliography.

    Semantic Scholar sometimes has no (or partial) references for a paper. Bibliography entries
    that name an unresolved baseline are looked up by arXiv id (one batch call) or title, added
    as candidates and as references of the citing paper, so ``link`` resolves them normally.
    """
    store, project = ctx.store, ctx.project
    included = [r for r in store.papers(project, "included") if r.extraction and r.extraction.baselines]
    if not included:
        return 0
    nodes = {r.node for r in store.papers(project, "included")}
    references: dict[str, list[str]] = defaultdict(list)
    for e in store.edges(project, ["cites"]):
        if e["src"] in nodes:
            references[e["src"]].append(e["dst"][2:])
    known = {p.paper.id: p.paper for p in store.papers(project, ids={i for ids in references.values() for i in ids})}
    done: dict[str, str | None] = store.meta(project, LOOKUPS, {})
    wanted: dict[str, tuple[dict, set[str]]] = {}  # title key -> (entry, citing paper ids)
    for r in included:
        refs = [known[i] for i in references.get(r.node, []) if i in known]
        table_refs = store.meta(project, f"table_refs:{r.paper.id}", [])
        bibliography = store.meta(project, f"bibliography:{r.paper.id}", [])
        for name in r.extraction.baselines:
            if resolve(name, refs, table_refs, {}, r.node)[0]:
                continue
            if (entry := bibliography_wanted(name, table_refs, bibliography)) and entry.get("title"):
                key = entity_key(entry["title"])
                if done.get(key) is None and key not in done:
                    wanted.setdefault(key, (entry, set()))[1].add(r.paper.id)
                elif done.get(key):  # looked up before: just make sure it is a reference
                    store.add_citations(project, [CitationLink(citing=r.paper.id, cited=done[key])])
    if not wanted:
        return 0
    items = list(wanted.items())[:limit]
    by_arxiv = {e["arxiv"]: k for k, (e, _) in items if e.get("arxiv")}
    found: dict[str, Paper] = {}
    try:
        for p in await ctx.s2.batch([f"ARXIV:{a}" for a in by_arxiv]) if by_arxiv else []:
            if p.arxiv_id and (k := by_arxiv.get(p.arxiv_id.split("v")[0])):
                found[k] = p
        for k, (entry, _) in items:
            if k not in found and (p := await ctx.s2.match_title(entry["title"].rstrip("."))):
                if match_reference(entry["title"], None, [p]):  # S2's best match must really be it
                    found[k] = p
    except Exception as e:  # noqa: BLE001 - lookups are best effort (rate limits)
        ctx.log(f"Bibliography lookups stopped early: {e}")
    store.add_papers(project, found.values())
    store.add_citations(project, [CitationLink(citing=c, cited=found[k].id)
                                  for k, (_, citing) in items if k in found for c in citing])
    done.update({k: (found[k].id if k in found else None) for k, _ in items})
    store.set_meta(project, LOOKUPS, done)
    ctx.log(f"Looked up {len(items)} baseline papers from bibliographies: found {len(found)}")
    return len(found)


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
