"""Extraction quality evaluation against a small gold set (``eval/gold.json``).

Each gold paper is fetched from Semantic Scholar together with its references (whose citation
sentences are the paper's own text) and a sample of citing papers (whose sentences describe it),
then analysed exactly like in a build. Reported per paper and overall:

* dataset recall: gold datasets that were extracted (each gold dataset is a list of aliases);
* dataset precision (lower bound): extracted datasets matching a gold dataset. Gold lists hold
  the main datasets, so unmatched predictions are not necessarily wrong;
* proposed-method hit rate, and how many papers got at least one limitation;
* how many papers needed full-text gap filling.

Run with ``research-forest evaluate``; it uses a throwaway database and never touches yours.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .engine import Forest
from .models import Scope
from .pipeline import analyze, baselines, discover
from .prompts import SYSTEM, clip
from .safety import data_block
from .store import entity_key, pid

DEFAULT_GOLD = Path("eval/gold.json")
CITATION_SAMPLE = 30  # citing papers fetched per gold paper, for incoming citation contexts


def _tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+", name.lower()))


def matches(predicted: str, aliases: list[str]) -> bool:
    """True if ``predicted`` names the same thing as one of ``aliases``.

    Equal normalised keys, or one name's tokens contained in the other's ("Wikipedia" vs
    "English Wikipedia", "ImageNet" vs "ImageNet-1k"), while "CIFAR-10" never matches "CIFAR-100".
    """
    key, toks = entity_key(predicted), _tokens(predicted)
    for alias in aliases:
        a_toks = _tokens(alias)
        if key == entity_key(alias) or (toks and a_toks and (toks <= a_toks or a_toks <= toks)):
            return True
    return False


class LimitationJudgement(BaseModel):
    covered: list[int] = Field(description="Refs of the gold limitations that the extracted list covers")


def judge_prompt(gold: list[str], extracted: list[str]) -> str:
    return ("Below are the limitations a paper itself states (gold, by ref) and the limitations an automatic "
            "extractor produced. Return the refs of the gold limitations that the extracted list covers: the same "
            "weakness, possibly worded differently or more generally. A vague extracted item (e.g. 'has some "
            "limitations') covers nothing.\n\n"
            + data_block("gold limitations", (f"[{i}] {clip(g, 300)}" for i, g in enumerate(gold))) + "\n\n"
            + data_block("extracted limitations", (f"- {clip(e, 300)}" for e in extracted)))


async def judge_limitations(llm, gold: list[str], extracted: list[str]) -> list[int]:
    """Indices of gold limitations covered by the extracted ones (LLM judge; [] when nothing to judge)."""
    if not gold or not extracted:
        return []
    res = await llm.json(LimitationJudgement, SYSTEM, judge_prompt(gold, extracted))
    return sorted({i for i in res.covered if 0 <= i < len(gold)})


def score_paper(gold: dict, extraction, covered: list[int] | None = None,
                resolved: dict[str, str] | None = None) -> dict:
    predicted = [d.name for d in extraction.datasets] if extraction else []
    gold_sets: list[list[str]] = gold["datasets"]
    found = [aliases[0] for aliases in gold_sets if any(matches(p, aliases) for p in predicted)]
    correct = [p for p in predicted if any(matches(p, aliases) for aliases in gold_sets)]
    proposed = [m.name for m in extraction.methods_proposed] if extraction else []
    return {
        "id": gold["id"], "title": gold["title"],
        "dataset_recall": len(found) / len(gold_sets) if gold_sets else 1.0,
        "dataset_precision": len(correct) / len(predicted) if predicted else 0.0,
        "method_hit": any(matches(m, aliases) for m in proposed for aliases in gold.get("methods_proposed", [])),
        "has_limitations": bool(extraction and extraction.limitations),
        "datasets_found": found,
        "datasets_missed": [a[0] for a in gold_sets if a[0] not in found],
        "datasets_extra": [p for p in predicted if p not in correct],
        "methods_proposed": proposed,
        "limitations": [f"{l.text} ({l.source})" for l in extraction.limitations] if extraction else [],
        **_limitation_scores(gold.get("limitations", []), extraction, covered or []),
        **_baseline_scores(gold.get("baselines", []), extraction, resolved or {}),
    }


def _baseline_scores(gold: list[list[str]], extraction, resolved: dict[str, str]) -> dict:
    """Baseline recall against the gold list, and how many extracted baselines resolved to a paper.
    ``resolved`` maps extracted baseline names to the title of the paper they resolved to."""
    predicted = list(extraction.baselines) if extraction else []
    found = [aliases[0] for aliases in gold if any(matches(p, aliases) for p in predicted)]
    return {
        "baseline_recall": len(found) / len(gold) if gold else None,
        "baselines_missed": [a[0] for a in gold if a[0] not in found],
        "baselines": predicted,
        "baselines_resolved": resolved,
        "baseline_resolution": len(resolved) / len(predicted) if predicted else None,
    }


def _limitation_scores(gold: list[str], extraction, covered: list[int]) -> dict:
    sources = [l.source for l in extraction.limitations] if extraction else []
    return {
        "limitation_recall": len(covered) / len(gold) if gold else None,  # None: no curated gold
        "limitations_found": [gold[i] for i in covered],
        "limitations_missed": [g for i, g in enumerate(gold) if i not in covered],
        "limitation_sources": {s: sources.count(s) for s in ("stated", "citing", "inferred")},
    }


def summarise(papers: list[dict], gap_filled: int) -> dict:
    n = len(papers) or 1
    judged = [p["limitation_recall"] for p in papers if p.get("limitation_recall") is not None]
    mean = lambda key: (round(sum(v) / len(v), 3) if (v := [p[key] for p in papers if p.get(key) is not None])  # noqa: E731
                        else None)
    sources = {s: sum(p.get("limitation_sources", {}).get(s, 0) for p in papers) for s in ("stated", "citing", "inferred")}
    return {
        "limitation_recall": round(sum(judged) / len(judged), 3) if judged else None,
        "baseline_recall": mean("baseline_recall"),
        "baseline_resolution": mean("baseline_resolution"),
        "limitations_per_paper": round(sum(sources.values()) / n, 1),
        "stated_limitation_share": round(sources["stated"] / (sum(sources.values()) or 1), 3),
        "papers": len(papers),
        "dataset_recall": round(sum(p["dataset_recall"] for p in papers) / n, 3),
        "dataset_precision_lower_bound": round(sum(p["dataset_precision"] for p in papers) / n, 3),
        "method_hit_rate": round(sum(p["method_hit"] for p in papers) / n, 3),
        "with_limitations": round(sum(p["has_limitations"] for p in papers) / n, 3),
        "gap_filled_papers": gap_filled,
    }


async def evaluate(settings: Settings, gold_path: Path = DEFAULT_GOLD, limit: int | None = None,
                   forest: Forest | None = None, on_log: Callable[[str], None] = print) -> dict:
    """Run the extraction pipeline on the gold papers and score it."""
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))["papers"][:limit]
    owned = forest is None
    if owned:
        tmp = tempfile.mkdtemp(prefix="rf-eval-")
        forest = Forest(replace(settings, db_path=str(Path(tmp) / "eval.forest.db")))
    forest.on_log = lambda _p, msg: on_log(f"  · {msg}")
    store = forest.store
    try:
        project = store.create_project("extraction evaluation")
        store.set_scope(project, Scope(title="Extraction evaluation", summary="Gold-set evaluation.",
                                       search_queries=[], seed_papers=[], include_criteria="all",
                                       exclude_criteria="none", year_min=None))
        ctx = forest.ctx(project)
        resolved: list[tuple[dict, str]] = []
        for g in gold:
            try:
                papers = await forest.s2.batch([g["id"]])
            except (RuntimeError, httpx.HTTPError) as e:  # e.g. exhausted retries on the shared rate limit
                on_log(f"  ! could not fetch {g['id']} ({g['title']}): {e}")
                continue
            if not papers:
                on_log(f"  ! not found on Semantic Scholar: {g['id']} ({g['title']})")
                continue
            store.add_papers(project, papers)
            store.update_papers(project, [papers[0].id], status="included")
            resolved.append((g, papers[0].id))
        ids = [i for _, i in resolved]
        await discover.fetch_links(ctx, ids, "references")  # the papers' own citation sentences
        for paper_id in ids:  # how other papers describe them (citing papers count as in-graph)
            try:
                links = await forest.s2.links(paper_id, "citations", limit=CITATION_SAMPLE)
            except (RuntimeError, httpx.HTTPError) as e:
                on_log(f"  ! citing papers unavailable for {paper_id}: {e} (scored without them)")
                continue
            store.add_papers(project, (p for p, _ in links))
            store.add_citations(project, (link for _, link in links))
            store.update_papers(project, [p.id for p, _ in links], status="included")
        stats = await analyze.extract(ctx, store.papers(project, ids=ids))

        records = {r.paper.id: r for r in store.papers(project, ids=ids)}

        async def judged(g: dict, paper_id: str) -> list[int]:
            ex = records[paper_id].extraction
            try:
                return await judge_limitations(forest.llm, g.get("limitations", []),
                                               [l.text for l in ex.limitations] if ex else [])
            except RuntimeError as e:
                on_log(f"  ! limitation judge failed for {g['title']}: {e}")
                return []

        covered = await asyncio.gather(*(judged(g, i) for g, i in resolved))
        await baselines.lookup_bibliography(ctx)
        baselines.link(ctx)  # resolve extracted baselines via table citations, the graph and references
        titles = {r.node: r.paper.title for r in store.papers(project)}
        linked: dict[str, dict[str, str]] = {}
        for e in store.edges(project, ["compares_with"]):
            if e["dst"].startswith("p:"):
                linked.setdefault(e["src"][2:], {})[e["data"]["name"]] = titles.get(e["dst"], e["dst"])
        papers = [score_paper(g, records[i].extraction, c, linked.get(i, {}))
                  for (g, i), c in zip(resolved, covered)]
        return {"summary": summarise(papers, stats.gap_filled), "papers": papers,
                "settings": {"backend": settings.backend, "model": settings.model,
                             "fulltext": settings.fulltext_enabled}}
    finally:
        if owned:
            await forest.aclose()


# ------------------------------------------------- full-text source comparison


def mentions(text: str, alias: str) -> bool:
    """Whether ``alias`` occurs in ``text`` as whole words, ignoring case and punctuation/spacing."""
    tokens = re.findall(r"[a-z]+|\d+", alias.lower())
    if not tokens:
        return False
    pattern = r"(?<![a-z0-9])" + r"[\W_]*".join(map(re.escape, tokens)) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


async def evaluate_sources(settings: Settings, gold_path: Path = DEFAULT_GOLD, limit: int | None = None,
                           top_k: int = 6, on_log: Callable[[str], None] = print) -> dict:
    """Compare full-text sources without an LLM: for each gold paper, run the dataset passage search
    on its arXiv HTML and on its PDF, and count the gold datasets named in the top passages."""
    from .sources.fulltext import FIELD_QUERIES, run_worker

    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))["papers"][:limit]
    sources = {"html": "https://arxiv.org/html/{}", "pdf": "https://arxiv.org/pdf/{}"}
    papers = []
    for g in gold:
        arxiv = g["id"].removeprefix("ARXIV:")
        row: dict[str, Any] = {"id": g["id"], "title": g["title"]}
        for name, template in sources.items():
            started = time.monotonic()
            res = await run_worker([template.format(arxiv)], {"datasets": FIELD_QUERIES["datasets"]}, settings, top_k)
            text = " ".join(p["text"] for p in (res.get("passages") or {}).get("datasets", []))
            found = [aliases[0] for aliases in g["datasets"] if any(mentions(text, a) for a in aliases)]
            row[name] = {"ok": not res.get("error"), "error": res.get("error"),
                         "recall": len(found) / len(g["datasets"]), "found": found,
                         "seconds": round(time.monotonic() - started, 1)}
        on_log(f"  · {g['title'][:40]}: html {row['html']['recall']:.0%}, pdf {row['pdf']['recall']:.0%}")
        papers.append(row)
    n = len(papers) or 1
    summary = {name: {"available": sum(p[name]["ok"] for p in papers),
                      "passage_recall": round(sum(p[name]["recall"] for p in papers) / n, 3),
                      "seconds": round(sum(p[name]["seconds"] for p in papers) / n, 1)} for name in sources}
    return {"summary": summary, "papers": papers, "top_k": top_k}


def format_sources_report(report: dict) -> str:
    s = report["summary"]
    lines = [f"Gold datasets named in the top {report['top_k']} dataset passages (no LLM involved)", "",
             f"  {'source':<8}{'available':>11}{'passage recall':>17}{'avg seconds':>14}"]
    for name, v in s.items():
        lines.append(f"  {name:<8}{v['available']:>8}/{len(report['papers']):<2}{v['passage_recall']:>16.0%}"
                     f"{v['seconds']:>14}")
    lines.append("")
    for p in report["papers"]:
        h, d = p["html"], p["pdf"]
        lines.append(f"  {p['title'][:40]:<40}  html {h['recall']:>4.0%}{'' if h['ok'] else ' (n/a)':<7}"
                     f"pdf {d['recall']:>4.0%}{'' if d['ok'] else ' (n/a)'}")
    return "\n".join(lines)


def format_report(report: dict, baseline: dict | None = None) -> str:
    s, b = report["summary"], (baseline or {}).get("summary", {})
    rows = [("Dataset recall", "dataset_recall"), ("Dataset precision (≥)", "dataset_precision_lower_bound"),
            ("Proposed method found", "method_hit_rate"), ("Papers with limitations", "with_limitations"),
            ("Limitation recall (judge)", "limitation_recall"), ("Stated limitations share", "stated_limitation_share"),
            ("Baseline recall", "baseline_recall"), ("Baselines resolved to papers", "baseline_resolution")]
    lines = [f"Evaluated {s['papers']} papers ({s['gap_filled_papers']} needed full-text gap filling)", ""]
    for label, key in rows:
        if s.get(key) is None:
            continue
        delta = f"  ({s[key] - b[key]:+.0%} vs baseline)" if b.get(key) is not None else ""
        lines.append(f"  {label:<26}{s[key]:>6.0%}{delta}")
    lines.append(f"  {'Limitations per paper':<26}{s.get('limitations_per_paper', 0):>6}")
    lines.append("")
    lines.append("  datasets  limitations")
    for p in sorted(report["papers"], key=lambda p: p["dataset_recall"]):
        missed = f" — missed: {', '.join(p['datasets_missed'])}" if p["datasets_missed"] else ""
        lim = "   —   " if p.get("limitation_recall") is None else f"{p['limitation_recall']:>6.0%} "
        lines.append(f"  {p['dataset_recall']:>6.0%}  {lim}  {p['title'][:40]:<40}{missed}")
    return "\n".join(lines)


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
