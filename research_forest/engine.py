"""The Forest engine: builds, incrementally updates, refines and queries research graphs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable

from .config import Settings
from .llm import LLM, make_llm
from .models import Answer, Paper, RefinePlan
from .pipeline import analyze, discover
from .pipeline.context import Ctx
from .pipeline.synthesize import synthesize as run_synthesis
from .prompts import SYSTEM, answer_prompt, clip, refine_prompt
from .sources.fulltext import PassageFinder
from .sources.semantic_scholar import SemanticScholar
from .store import Store

CORE_FRACTION = 0.6  # share of the paper budget filled from search before snowballing


class Forest:
    def __init__(self, settings: Settings, *, store: Store | None = None, llm: LLM | None = None,
                 s2: SemanticScholar | None = None, on_log: Callable[[int, str], None] | None = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.llm = llm or make_llm(settings)
        self.s2 = s2 or SemanticScholar(settings.s2_api_key, settings.s2_rps)
        self.finder = PassageFinder(self.s2, settings) if settings.fulltext_enabled else None
        self.on_log = on_log

    def ctx(self, project: int) -> Ctx:
        return Ctx(self.store, self.llm, self.s2, self.finder, self.settings, project, self.on_log)

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.s2.aclose()

    # ------------------------------------------------------------- building

    async def build(self, prompt: str, target: int | None = None, project: int | None = None) -> int:
        project = project or self.store.create_project(prompt)
        ctx, target = self.ctx(project), target or self.settings.target_papers
        self.store.set_status(project, "building")
        try:
            ctx.log("defining scope")
            scope = await discover.make_scope(ctx, prompt)
            self.store.set_scope(project, scope)
            ctx.log(f"scope: {scope.title} — queries: {', '.join(scope.search_queries)}")

            hits = await discover.search(ctx, scope.search_queries, scope.seed_papers, year_min=scope.year_min)
            await discover.screen(ctx, hits)
            discover.select(ctx, math.ceil(target * CORE_FRACTION))
            await self._fetch_new_links(ctx)
            ctx.log("snowballing through references")
            await discover.snowball(ctx, target - len(self.store.papers(project, "included")))
            discover.select(ctx, target - len(self.store.papers(project, "included")))  # top up if snowball was thin
            await self.process(project, synthesize=True)
            self.store.set_status(project, "ready")
        except BaseException:
            self.store.set_status(project, "error")
            raise
        return project

    async def _fetch_new_links(self, ctx: Ctx) -> None:
        todo = [r.paper.id for r in self.store.papers(ctx.project, "included") if not r.refs_fetched]
        if todo:
            ctx.log(f"fetching citation links for {len(todo)} papers")
            await discover.refresh_metadata(ctx, todo)
            await discover.fetch_links(ctx, todo, "references")

    async def process(self, project: int, synthesize: bool = False) -> None:
        """Bring derived data up to date for the current paper set (only new work is done)."""
        ctx = self.ctx(project)
        await self._fetch_new_links(ctx)
        todo = [r for r in self.store.papers(project, "included") if r.extraction is None]
        if todo:
            ctx.log(f"analysing {len(todo)} papers")
            await analyze.extract(ctx, todo)
        ctx.log("tracing lineage (builds-on relations)")
        await analyze.relate(ctx)
        if todo:
            await analyze.normalize(ctx)
        self.store.prune_orphans(project)
        if synthesize or todo:
            ctx.log("synthesising limitations and open questions")
            await run_synthesis(ctx)
        self.store.reindex(project)

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
        plan, papers = await self.plan(project, command)
        ctx.log(f"plan: {plan.explanation}")
        self.store.snapshot(project, command)
        self.store.set_status(project, "refining")
        try:
            before = len(papers)
            for op in plan.ops:
                ids = list(dict.fromkeys(papers[i].id for i in op.papers if 0 <= i < len(papers)))
                ctx.log(f"op {op.action}" + (f": {op.query}" if op.query else ""))
                await self._apply(ctx, op, ids)
            removed = before - len({r.paper.id for r in self.store.papers(project, "included")} & {p.id for p in papers})
            if before and removed / before > 0.5:
                ctx.log(f"warning: this refinement removed {removed} of {before} papers — use undo if unintended")
            await self.process(project, synthesize=True)
            self.store.set_status(project, "ready")
        except BaseException:
            self.store.set_status(project, "error")
            raise
        return plan

    async def _apply(self, ctx: Ctx, op, ids: list[str]) -> None:
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
                pass  # process() always re-synthesises after a refinement
            case _:
                ctx.log(f"skipped incomplete op {op.action}")

    def undo(self, project: int) -> str | None:
        return self.store.restore(project)

    # --------------------------------------------------------------- Q & A

    async def ask(self, project: int, question: str, max_nodes: int = 60) -> dict:
        """Answer a natural-language question from a retrieved subgraph."""
        graph = self.store.graph(project)
        nodes = {n["id"]: n for n in graph["nodes"]}
        adj = defaultdict(list)
        for e in graph["edges"]:
            adj[e["source"]].append(e)
            adj[e["target"]].append(e)

        seeds = [h["node"] for h in self.store.search(project, question, limit=20, mode="OR") if h["node"] in nodes]
        chosen = list(dict.fromkeys(seeds + [n for n, v in nodes.items() if v["kind"] == "question"][:8]))
        for s in seeds:
            if len(chosen) >= max_nodes:
                break
            chosen += [e["target"] if e["source"] == s else e["source"] for e in adj[s]][:6]
        chosen = list(dict.fromkeys(chosen))[:max_nodes]

        handle = {n: f"n{i}" for i, n in enumerate(chosen)}
        records = {r.node: r for r in self.store.papers(project, "included")}
        lines = []
        for n in chosen:
            v = nodes[n]
            if (r := records.get(n)) is not None:
                ex = r.extraction
                detail = clip(ex.contribution if ex else r.paper.abstract, 220)
                if ex:
                    detail += f" | datasets: {', '.join(d.name for d in ex.datasets[:5]) or '-'}"
                    detail += f" | limitations: {'; '.join(clip(l.text, 90) for l in ex.limitations[:3]) or '-'}"
                lines.append(f"{handle[n]} [paper {r.paper.year or '?'}] {clip(r.paper.title, 300)} — {detail}")
            else:
                lines.append(f"{handle[n]} [{v['kind']}] {clip(v['label'], 300)}")
        seen = set()
        for n in chosen:
            for e in adj[n]:
                key = (e["source"], e["target"], e["kind"])
                if key in seen or e["source"] not in handle or e["target"] not in handle:
                    continue
                seen.add(key)
                label = e["kind"] + (f"[{clip(e.get('change_type'), 20)}: {clip(e.get('summary'), 120)}]"
                                     if e["kind"] == "builds_on" else "")
                lines.append(f"{handle[e['source']]} -{label}-> {handle[e['target']]}")

        res = await self.llm.json(Answer, SYSTEM, answer_prompt(self.store.scope(project), lines, question))
        back = {h: n for n, h in handle.items()}
        answer = res.answer
        for h, n in back.items():
            answer = answer.replace(f"[{h}]", f"[{h}]({n})")
        return {"answer": answer, "nodes": [back[h] for h in res.nodes if h in back]}
