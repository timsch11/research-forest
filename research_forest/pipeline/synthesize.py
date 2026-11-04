"""Cross-paper synthesis: limitation themes and open research questions."""

from __future__ import annotations

from ..models import Synthesis
from ..prompts import SYSTEM, synthesize_prompt
from .context import Ctx


async def synthesize(ctx: Ctx) -> None:
    recs = [r for r in ctx.store.papers(ctx.project, "included")
            if r.extraction and (r.extraction.limitations or r.extraction.future_work)]
    if not recs:
        return
    recs.sort(key=lambda r: (r.paper.year or 0, r.paper.title))
    rows = [(r.paper, [l.text for l in r.extraction.limitations][:4], r.extraction.future_work[:2]) for r in recs]
    syn = await ctx.llm.json(Synthesis, SYSTEM, synthesize_prompt(ctx.store.scope(ctx.project), rows))

    def paper_nodes(refs: list[int]) -> list[str]:
        return [recs[i].node for i in dict.fromkeys(refs) if 0 <= i < len(recs)]

    store, project = ctx.store, ctx.project
    store.delete_entities(project, ["limitation", "question"])
    edges, themes = [], []
    for t in syn.themes:
        node = store.upsert_entity(project, "limitation", t.name, t.description)
        themes.append(node)
        edges += [(p, node, "has_limitation", None) for p in paper_nodes(t.papers)]
        edges += [(p, node, "addresses", None) for p in paper_nodes(t.addressed_by)]
    for q in syn.questions:
        node = store.upsert_entity(project, "question", q.question, q.rationale)
        edges += [(themes[i], node, "motivates", None) for i in dict.fromkeys(q.themes) if 0 <= i < len(themes)]
        edges += [(node, p, "about", None) for p in paper_nodes(q.papers)]
    store.add_edges(project, edges)
    ctx.log(f"synthesised {len(syn.themes)} limitation themes and {len(syn.questions)} open questions")
