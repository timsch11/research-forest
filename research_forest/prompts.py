"""Prompt templates. Items are referenced by small integer refs to keep prompts short
and make hallucinated ids easy to detect."""

from __future__ import annotations

from .models import Paper, Scope

SYSTEM = (
    "You are an expert research analyst who maps the landscape of a scientific field. "
    "Be precise and concise. Only state facts supported by the provided text; anything you infer must be "
    "marked as inferred where the schema allows it. Never invent papers, datasets or results."
)

CHANGE_TYPE_GUIDE = """change_type values:
- architecture: changes model structure/components
- objective: changes loss, training objective or problem formulation
- training: changes optimisation, schedule, fine-tuning or training procedure
- data: changes training data, augmentation or data collection
- efficiency: same goal, cheaper/faster/smaller (compute, memory, latency)
- evaluation: new benchmark, metric or evaluation protocol applied to the prior work
- theory: theoretical analysis/explanation of the prior work
- application: applies the prior method to a new domain or task
- other: none of the above"""


def clip(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def paper_line(i: int, p: Paper, abstract_chars: int = 500) -> str:
    return f"[{i}] ({p.year or '?'}) {p.title} — {clip(p.tldr or p.abstract, abstract_chars) or 'no abstract'}"


def scope_block(scope: Scope) -> str:
    return (f"Research area: {scope.title}\n{scope.summary}\n"
            f"Include: {scope.include_criteria}\nExclude: {scope.exclude_criteria}")


def scope_prompt(prompt: str) -> str:
    return (
        "A user wants to explore the research landscape described below. Define its scope and produce search "
        "queries for Semantic Scholar (short keyword queries, not sentences; cover sub-areas, synonyms and key "
        "method names) plus exact titles of landmark papers you are confident exist.\n\n"
        f"User request:\n{prompt}")


def rescope_prompt(scope: Scope, instruction: str) -> str:
    return (f"Current scope:\n{scope_block(scope)}\nQueries: {scope.search_queries}\n\n"
            f"The user refines the focus: {instruction}\n\n"
            "Return the updated scope. search_queries should only contain NEW queries needed for the new focus.")


def screen_prompt(scope: Scope, papers: list[Paper], extra: str = "") -> str:
    lines = "\n".join(paper_line(i, p, 400) for i, p in enumerate(papers))
    return (f"{scope_block(scope)}\n{extra}\n\nRate each paper's relevance to this research area "
            f"(0 off-topic, 1 tangential, 2 relevant, 3 core). Return one item per ref.\n\nPapers:\n{lines}")


def prune_prompt(criterion: str, papers: list[Paper]) -> str:
    lines = "\n".join(paper_line(i, p, 300) for i, p in enumerate(papers))
    return (f"The user wants to REMOVE papers matching this criterion: {criterion}\n\n"
            "For each paper, set relevance to how strongly it matches the removal criterion "
            f"(0 = does not match, 3 = clearly matches). Return one item per ref.\n\nPapers:\n{lines}")


def extract_prompt(p: Paper, contexts: list[str], passages: dict[str, list[str]] | None = None) -> str:
    parts = [f"Title: {p.title}", f"Year: {p.year or '?'}  Venue: {p.venue or '?'}",
             f"Abstract: {p.abstract or 'n/a'}"]
    if p.tldr:
        parts.append(f"TL;DR: {p.tldr}")
    if contexts:
        parts.append("How other papers describe this work (citation contexts):\n" +
                     "\n".join(f"- {clip(c, 350)}" for c in contexts))
    if passages:
        for field, texts in passages.items():
            parts.append(f"Full-text passages likely relevant to {field}:\n" +
                         "\n".join(f"- {clip(t, 1200)}" for t in texts))
    parts.append(
        "Extract the paper's contribution, the methods it proposes (canonical short names) and the existing "
        "methods/architectures/algorithms it builds on or compares against (not datasets, not tasks), the "
        "datasets/benchmarks it trains or evaluates on, its limitations (stated by authors, reported by citing "
        "papers, or inferred) and future work. Use widely-used canonical names (e.g. 'ImageNet-1k', not "
        "'the ImageNet dataset'). Add a field to `missing` only if the text gives essentially no information "
        "for it (e.g. no dataset is named at all); then give 3-6 short `search_terms` (single words or "
        "2-word phrases) likely to occur in the relevant full-text passages.")
    return "\n\n".join(parts)


def relate_prompt(citing: Paper, contribution: str | None, cited: list[tuple[Paper, str | None, dict]]) -> str:
    lines = []
    for i, (p, contrib, edge) in enumerate(cited):
        ctx = "; ".join(clip(c, 250) for c in edge.get("contexts", [])[:3]) or "none"
        flags = ", ".join(edge.get("intents") or []) + (" [influential]" if edge.get("influential") else "")
        lines.append(f"[{i}] ({p.year or '?'}) {p.title}\n    contribution: {clip(contrib or p.abstract, 300)}\n"
                     f"    citation contexts: {ctx}\n    intents: {flags or 'unknown'}")
    return (
        f"Citing paper ({citing.year or '?'}): {citing.title}\nContribution: {contribution or clip(citing.abstract, 600)}\n\n"
        "For each cited paper decide whether the citing paper BUILDS UPON it: directly adopts and modifies, "
        "extends, re-implements more efficiently, or replaces a component of the cited work. Mere background, "
        "comparison baselines or dataset usage are NOT builds_on. If builds_on, give change_type, the component changed, and a one-sentence "
        "summary of what changes. Otherwise set those fields to null.\n\n"
        f"{CHANGE_TYPE_GUIDE}\n\nCited papers:\n" + "\n".join(lines))


def normalize_prompt(kind: str, items: list[tuple[str, str | None]]) -> str:
    lines = "\n".join(f"[{i}] {n}" + (f" — {clip(d, 120)}" if d else "") for i, (n, d) in enumerate(items))
    return (f"These are {kind} names extracted from different papers. Group entries that refer to the SAME "
            f"{kind} (aliases, abbreviations, spelling variants). Do not merge distinct variants or versions "
            "(e.g. 'GPT-2' vs 'GPT-3'). Only return groups with 2+ members; canonical is the best common name.\n\n"
            + lines)


def synthesize_prompt(scope: Scope, rows: list[tuple[Paper, list[str], list[str]]]) -> str:
    lines = []
    for i, (p, lims, future) in enumerate(rows):
        lines.append(f"[{i}] ({p.year or '?'}) {clip(p.title, 120)}\n    limitations: {'; '.join(lims) or '-'}"
                     + (f"\n    future work: {'; '.join(future)}" if future else ""))
    return (
        f"{scope_block(scope)}\n\nBelow are papers with their extracted limitations. Synthesise:\n"
        "1. 5-15 limitation THEMES shared across papers (name, description, refs of papers with the limitation, "
        "refs of later papers that address it).\n"
        "2. 5-12 OPEN RESEARCH QUESTIONS that remain unresolved, each motivated by themes (by index) and linked "
        "to relevant papers. Prefer specific, actionable questions over generic ones.\n\n" + "\n".join(lines))


def refine_prompt(scope: Scope, stats: str, papers: list[Paper], command: str) -> str:
    lines = "\n".join(f"[{i}] ({p.year or '?'}, {p.citation_count} cit.) {clip(p.title, 110)}"
                      for i, p in enumerate(papers))
    return f"""{scope_block(scope)}

Graph statistics: {stats}

Papers currently in the graph:
{lines}

The user wants to refine the graph: "{command}"

Translate this into a short list of operations (fields not used by an op are null / []):
- search: query = new Semantic Scholar keyword query; limit = max papers to add (default 10)
- expand: papers = refs to grow around; direction = references | citations | both; limit = papers per seed
- remove: papers = refs to delete
- filter: year_min / year_max / min_citations; removes papers outside the bounds
- prune: query = description of papers to REMOVE (semantic filter)
- deepen: papers = refs to re-analyse using full-text passages (better datasets/limitations)
- rescope: query = new focus description (re-screens papers and searches for the new focus)
- resynthesize: recompute limitation themes and open questions
Explain the plan in one or two sentences."""


def answer_prompt(scope: Scope, context: str, question: str) -> str:
    return (f"{scope_block(scope)}\n\nKnowledge graph excerpt (nodes are referenced by handle, e.g. n3):\n"
            f"{context}\n\nQuestion: {question}\n\nAnswer from the graph excerpt only; say so if it is "
            "insufficient. Cite supporting nodes inline as [n3] and list them in `nodes`.")
