"""The Forest engine: builds, incrementally updates, refines and queries research graphs."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Callable

from .config import Settings
from .llm import LLM, make_llm
from .models import Answer, Paper, RefinePlan
from .pipeline import analyze, baselines, discover
from .pipeline.context import Ctx
from .pipeline.synthesize import synthesize as run_synthesis
from .prompts import SYSTEM, answer_prompt, clip, refine_prompt
from .sources.fulltext import PassageFinder
from .sources.semantic_scholar import SemanticScholar
from .store import Store

CORE_FRACTION = 0.6  # share of the paper budget filled from search before snowballing

# Ordered stages of each job, shown as a stepper in the UI. process() covers the shared tail.
PROCESS_STAGES = [("links", "Citations"), ("extract", "Analysis"), ("lineage", "Lineage"),
                  ("normalize", "Duplicates"), ("baselines", "Baselines"), ("synthesize", "Open questions")]
JOB_STAGES = {
    "build": [("scope", "Scope"), ("search", "Search"), ("screen", "Screening"), ("snowball", "Snowball"),
              *PROCESS_STAGES],
    "refine": [("plan", "Plan"), ("apply", "Apply"), *PROCESS_STAGES],
}


class Forest:
    def __init__(self, settings: Settings, *, store: Store | None = None, llm: LLM | None = None,
                 s2: SemanticScholar | None = None, on_log: Callable[[int, str], None] | None = None,
                 on_progress: Callable[[int, dict], None] | None = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.llm = llm or make_llm(settings)
        self.s2 = s2 or SemanticScholar(settings.s2_api_key, settings.s2_rps)
        self.finder = PassageFinder(self.s2, settings) if settings.fulltext_enabled else None
        self.on_log = on_log
        self.on_progress = on_progress

    def ctx(self, project: int) -> Ctx:
        return Ctx(self.store, self.llm, self.s2, self.finder, self.settings, project, self.on_log, self.on_progress)

    def _begin(self, project: int, job: str) -> None:
        if self.on_progress:
            self.on_progress(project, {"job": job, "plan": [{"key": k, "label": l} for k, l in JOB_STAGES[job]]})

    def _summary(self, project: int, started: float) -> str:
        g = self.store.graph(project)
        kinds = Counter(n["kind"] for n in g["nodes"])
        builds = sum(e["kind"] == "builds_on" for e in g["edges"])
        mins, secs = divmod(int(time.monotonic() - started), 60)
        return (f"Done in {mins}m{secs:02d}s: {kinds['paper']} papers, {builds} builds-on edges, "
                f"{kinds['method']} methods, {kinds['dataset']} datasets, {kinds['limitation']} limitation themes, "
                f"{kinds['question']} open questions")

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.s2.aclose()

    # ------------------------------------------------------------- building

    async def build(self, prompt: str, target: int | None = None, project: int | None = None) -> int:
        project = project or self.store.create_project(prompt)
        ctx, target = self.ctx(project), target or self.settings.target_papers
        started = time.monotonic()
        self.store.set_status(project, "building")
        self._begin(project, "build")
        try:
            ctx.stage("scope", "Defining the scope of the research area")
            ctx.work(1)
            scope = await discover.make_scope(ctx, prompt)
            ctx.tick()
            self.store.set_scope(project, scope)
            ctx.log(f"Scope: {scope.title}. Queries: {'; '.join(scope.search_queries)}"
                    + (f". Landmark papers: {'; '.join(scope.seed_papers)}" if scope.seed_papers else ""))

            ctx.stage("search", "Searching Semantic Scholar")
            hits = await discover.search(ctx, scope.search_queries, scope.seed_papers, year_min=scope.year_min)
            ctx.stage("screen", f"Screening search results for relevance (target: {target} papers)")
            await discover.screen(ctx, hits)
            discover.select(ctx, math.ceil(target * CORE_FRACTION))
            await self._fetch_new_links(ctx)
            ctx.stage("snowball", "Snowballing: adding frequently co-cited references")
            await discover.snowball(ctx, target - len(self.store.papers(project, "included")))
            discover.select(ctx, target - len(self.store.papers(project, "included")))  # top up if snowball was thin
            await self.process(project, synthesize=True, baseline_papers=self.settings.baseline_papers)
            ctx.log(self._summary(project, started))
            self.store.set_status(project, "ready")
        except BaseException:
            self.store.set_status(project, "error")
            raise
        return project

    async def _fetch_new_links(self, ctx: Ctx) -> None:
        todo = [r.paper.id for r in self.store.papers(ctx.project, "included") if not r.refs_fetched]
        if todo:
            await discover.refresh_metadata(ctx, todo)
            await discover.fetch_links(ctx, todo, "references")

    async def process(self, project: int, synthesize: bool = False, baseline_papers: int = 0) -> None:
        """Bring derived data up to date for the current paper set (only new work is done).

        ``baseline_papers``: afterwards, add up to this many papers that graph papers use as
        baselines but the graph lacks, and analyse them too.
        """
        ctx = self.ctx(project)
        analysed = await self._analyse(ctx, staged=True)
        if baseline_papers > 0:
            ctx.stage("baselines", "Adding papers that graph papers compare against")
            if await baselines.expand(ctx, baseline_papers):
                analysed += await self._analyse(ctx, staged=False)
        self.store.prune_orphans(project)
        if synthesize or analysed:
            ctx.stage("synthesize", "Synthesising limitation themes and open questions")
            await run_synthesis(ctx)
        self.store.reindex(project)

    async def _analyse(self, ctx: Ctx, staged: bool) -> int:
        """Citations, extraction, baseline links, lineage and normalisation for new papers.
        Returns the number of newly analysed papers. ``staged=False`` runs inside another stage."""
        stage = ctx.stage if staged else (lambda key, label: ctx.log(label))
        stage("links", "Fetching citations and citation contexts")
        await self._fetch_new_links(ctx)
        todo = [r for r in self.store.papers(ctx.project, "included") if r.extraction is None]
        if todo:
            stage("extract", f"Analysing {len(todo)} papers")
            await analyze.extract(ctx, todo)
        stage("lineage", "Tracing lineage: baseline comparisons and which papers build on which")
        await baselines.lookup_bibliography(ctx)
        baselines.link(ctx)
        await analyze.relate(ctx)
        if todo:
            stage("normalize", "Merging duplicate methods and datasets")
            await analyze.normalize(ctx)
        return len(todo)

    # ----------------------------------------------------------- refinement

    def _stats(self, project: int, papers: list[Paper]) -> str:
        kinds = Counter(n["kind"] for n in self.store.graph(project)["nodes"])
        years = [p.year for p in papers if p.year]
        degree = Counter(e["dst"] for e in self.store.edges(project, ["proposes", "uses", "evaluates_on"]))
        names = {e["node"]: (e["kind"], e["name"]) for e in self.store.entities(project)}
        top = defaultdict(list)
        for node, _ in degree.most_common():
            if node in names and len(top[names[node][0]]) < 8:
                top[names[node][0]].append(names[node][1])
        span = f"{min(years)}-{max(years)}" if years else "n/a"
        return (f"{dict(kinds)}; years {span}; top methods: {', '.join(top['method'])}; "
                f"top datasets: {', '.join(top['dataset'])}")

    async def plan(self, project: int, command: str) -> tuple[RefinePlan, list[Paper]]:
        papers = sorted((r.paper for r in self.store.papers(project, "included")),
                        key=lambda p: (p.year or 0, p.title))
        prompt = refine_prompt(self.store.scope(project), self._stats(project, papers), papers, command)
        return await self.llm.json(RefinePlan, SYSTEM, prompt, use_cache=False), papers

    async def refine(self, project: int, command: str) -> RefinePlan:
        """Apply a natural-language refinement command. A snapshot is taken first (see :meth:`undo`)."""
        ctx = self.ctx(project)
        started = time.monotonic()
        self._begin(project, "refine")
        ctx.stage("plan", f"Planning: {clip(command, 120)}")
        ctx.work(1)
        plan, papers = await self.plan(project, command)
        ctx.tick()
        ops = ", ".join(op.action for op in plan.ops) or "none"
        ctx.log(f"Plan ({ops}): {plan.explanation}")
        self.store.snapshot(project, command)
        self.store.set_status(project, "refining")
        try:
            ctx.stage("apply", f"Applying {len(plan.ops)} operation(s)")
            before = {p.id for p in papers}
            forced = False
            for op in plan.ops:
                ids = list(dict.fromkeys(papers[i].id for i in op.papers if 0 <= i < len(papers)))
                ctx.log(f"Operation {op.action}" + (f": {op.query}" if op.query else "")
                        + (f" on {len(ids)} paper(s)" if ids else ""))
                forced |= await self._apply(ctx, op, ids)
            after = {r.paper.id for r in self.store.papers(project, "included")}
            removed = len(before - after)
            if before and removed / len(before) > 0.5:
                ctx.log(f"warning: this refinement removed {removed} of {len(before)} papers — use undo if unintended")
            # Synthesis is one of the largest prompts: skip it when the refinement changed nothing.
            await self.process(project, synthesize=forced or after != before)
            ctx.log(self._summary(project, started))
            self.store.set_status(project, "ready")
        except BaseException:
            self.store.set_status(project, "error")
            raise
        return plan

    async def _apply(self, ctx: Ctx, op, ids: list[str]) -> bool:
        """Run one refinement op. Returns True when it requires re-synthesis regardless of paper changes."""
        match op.action:
            case "search" if op.query:
                hits = await discover.search(ctx, [op.query], year_min=op.year_min)
                await discover.screen(ctx, hits)
                discover.select(ctx, op.limit or 10, hits)
            case "expand" if ids:
                await discover.expand(ctx, ids, op.direction or "both", op.limit or 5)
            case "remove":
                self.store.exclude_papers(ctx.project, ids)
            case "filter":
                discover.apply_filter(ctx, op.year_min, op.year_max, op.min_citations)
            case "prune" if op.query:
                await discover.prune(ctx, op.query)
            case "deepen" if ids:
                await analyze.extract(ctx, self.store.papers(ctx.project, ids=ids), deep=True)
                return True
            case "rescope" if op.query:
                new = await discover.rescope(ctx, op.query)
                included = [r.paper.id for r in self.store.papers(ctx.project, "included")]
                await discover.screen(ctx, included, force=True)
                off = [r.paper.id for r in self.store.papers(ctx.project, "included") if (r.relevance or 0) < 2]
                self.store.exclude_papers(ctx.project, off)
                ctx.log(f"rescope removed {len(off)} off-focus papers")
                hits = await discover.search(ctx, new.search_queries, new.seed_papers, year_min=new.year_min)
                await discover.screen(ctx, hits)
                discover.select(ctx, op.limit or min(discover.MAX_PAPERS_PER_OP, max(10, len(off))), hits)
            case "resynthesize":
                return True
            case _:
                ctx.log(f"skipped incomplete op {op.action}")
        return False

    def undo(self, project: int) -> str | None:
        return self.store.restore(project)

    # --------------------------------------------------------------- Q & A

    async def ask(self, project: int, question: str, previous: dict | None = None, max_nodes: int = 50) -> dict:
        """Answer a natural-language question from a retrieved subgraph.

        Context is kept lean: method/dataset links are inlined into paper lines instead of being
        listed as edges, plain citations are omitted, and open questions are only attached when
        the question asks about gaps. ``previous`` ({"question", "nodes"}) makes follow-ups like
        "which of those…" work by re-using the nodes the last answer cited — its text is not re-sent.
        """
        graph = self.store.graph(project)
        nodes = {n["id"]: n for n in graph["nodes"]}
        adj = defaultdict(list)
        for e in graph["edges"]:
            if e["kind"] != "cites":
                adj[e["source"]].append(e)
                adj[e["target"]].append(e)

        follow_up = bool(previous and previous.get("question") and FOLLOW_UP.search(question))
        carried = [n for n in (previous or {}).get("nodes", []) if n in nodes][:20] if follow_up else []
        seeds = list(dict.fromkeys(carried + [h["node"] for h in self.store.search(project, question, limit=20,
                                                                                    mode="OR") if h["node"] in nodes]))
        chosen = list(seeds)
        if GAP_INTENT.search(question) or len(seeds) < 3:
            chosen += [n for n, v in nodes.items() if v["kind"] == "question"][:8]
        for s in seeds:
            if len(chosen) >= max_nodes:
                break
            chosen += [e["target"] if e["source"] == s else e["source"] for e in adj[s]][:6]
        chosen = list(dict.fromkeys(chosen))[:max_nodes]

        handle = {n: f"n{i}" for i, n in enumerate(chosen)}
        records = {r.node: r for r in self.store.papers(project, "included")}
        lines = []
        for n in chosen:
            if (r := records.get(n)) is not None:
                lines.append(f"{handle[n]} [paper {r.paper.year or '?'}] {clip(r.paper.title, 150)} — "
                             + _paper_summary(r))
            else:
                lines.append(f"{handle[n]} [{nodes[n]['kind']}] {clip(nodes[n]['label'], 200)}")
        seen = set()
        for n in chosen:
            for e in adj[n]:
                key = (e["source"], e["target"], e["kind"])
                if (key in seen or e["kind"] not in CONTEXT_EDGES
                        or e["source"] not in handle or e["target"] not in handle):
                    continue
                seen.add(key)
                label = e["kind"] + (f"[{clip(e.get('change_type'), 20)}: {clip(e.get('summary'), 120)}]"
                                     if e["kind"] == "builds_on" else "")
                lines.append(f"{handle[e['source']]} -{label}-> {handle[e['target']]}")

        prompt = answer_prompt(self.store.scope(project), lines, question,
                               previous["question"] if follow_up else None)
        res = await self.llm.json(Answer, SYSTEM, prompt)
        back = {h: n for n, h in handle.items()}
        answer = res.answer
        for h, n in back.items():
            answer = answer.replace(f"[{h}]", f"[{h}]({n})")
        return {"answer": answer, "nodes": [back[h] for h in res.nodes if h in back]}


# Edges worth spelling out in Q&A context; method/dataset links are inlined in paper lines.
CONTEXT_EDGES = {"builds_on", "compares_with", "has_limitation", "addresses", "motivates", "about"}
GAP_INTENT = re.compile(r"\b(open|gaps?|future|unsolved|unresolved|questions?|limitations?|challenges?|"
                        r"problems?|directions?|missing|lack)\b", re.IGNORECASE)
FOLLOW_UP = re.compile(r"^\s*(and|also|but|so|then|what about|how about|which of|of (those|these|them))\b"
                       r"|\b(those|these|them|they|their|it|its|that one|the same|above|previous)\b",
                       re.IGNORECASE)


def _paper_summary(r) -> str:
    ex = r.extraction
    if not ex:
        return clip(r.paper.abstract, 200)
    parts = [clip(ex.contribution, 200)]
    if ex.methods_proposed:
        parts.append("proposes: " + ", ".join(clip(m.name, 40) for m in ex.methods_proposed[:3]))
    if ex.datasets:
        parts.append("datasets: " + ", ".join(clip(d.name, 40) for d in ex.datasets[:5]))
    if ex.limitations:
        parts.append("limitations: " + "; ".join(clip(l.text, 90) for l in ex.limitations[:3]))
    return " | ".join(parts)
