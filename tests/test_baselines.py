"""Baseline relations: table citations, resolution to papers, linking, discovery and lineage hints."""

import pytest

from conftest import FakeBackend, make_forest, op
from research_forest.models import CitationLink, Extraction, Paper
from research_forest.pipeline import baselines
from research_forest.pipeline.baselines import match_reference, resolve, title_head
from research_forest.sources import fulltext_worker as worker
from research_forest.store import pid

# ------------------------------------------------------------ worker side

CITED_TABLE = """<html><body><article class="ltx_document">
<section class="ltx_section"><h2>4 Experiments</h2>
<figure class="ltx_table"><figcaption>Table 3: Results on Long Range Arena.</figcaption><table class="ltx_tabular">
<tr><th>Models</th><th>ListOps</th><th>Text</th></tr>
<tr><th><span>Linformer <cite>[<a href="#bib.bib84" class="ltx_ref">84</a>]</cite></span></th><td>35.6</td><td>55.9</td></tr>
<tr><th><span>Performer <cite>[<a href="#bib.bib12" class="ltx_ref">12</a>]</cite></span></th><td>36.8</td><td>63.6</td></tr>
<tr><th>Ours</th><td>37.0</td><td>64.0</td></tr></table></figure></section>
<section class="ltx_appendix"><h2>Appendix A</h2>
<figure class="ltx_table"><table class="ltx_tabular"><tr><th>Model</th><th>X</th></tr>
<tr><th>Hidden <cite>[<a href="#bib.bib12" class="ltx_ref">12</a>]</cite></th><td>1</td></tr></table></figure></section>
<section class="ltx_bibliography"><ul>
<li id="bib.bib12" class="ltx_bibitem"><span class="ltx_tag">[12]</span>
<span class="ltx_bibblock">K. Choromanski et al.</span><span class="ltx_bibblock">Rethinking attention with performers.</span>
<span class="ltx_bibblock">ICLR, 2021.</span></li>
<li id="bib.bib84" class="ltx_bibitem"><span class="ltx_tag">[84]</span><span class="ltx_bibblock">S. Wang et al.</span>
<span class="ltx_bibblock">Linformer: Self-attention with linear complexity.</span>
<span class="ltx_bibblock"><em>arXiv preprint arXiv:2006.04768</em>, 2020.</span></li></ul></section>
</article></body></html>"""


def test_cited_table_labels_resolve_to_bibliography():
    doc = worker.parse_document(CITED_TABLE.encode(), "text/html", "https://arxiv.org/html/2205.14135")
    assert doc.table_refs == [
        {"label": "Linformer", "title": "Linformer: Self-attention with linear complexity.", "arxiv": "2006.04768",
         "table": "Table 3"},
        {"label": "Performer", "title": "Rethinking attention with performers.", "arxiv": None, "table": "Table 3"}]
    assert "rows: Linformer, Performer, Ours" in doc.tables[0]  # citation markers stripped from the summary
    assert "SQuAD" not in doc.text and "Choromanski" not in doc.text  # bibliography still not in the text


def test_cite_markers_stripped_in_compact_tables():
    assert worker.compact_table("T", [["Model", "Acc"], ["Longformer [8]", "1"], ["BigBird (Zaheer et al., 2020)", "2"]]) \
        == "T — columns: Model | Acc — rows: Longformer, BigBird"


# ------------------------------------------------------------- resolution

REFS = [Paper(id="lin", title="Linformer: Self-Attention with Linear Complexity", arxiv_id="2006.04768"),
        Paper(id="perf", title="Rethinking Attention with Performers"),
        Paper(id="slim", title="Sub-Linear Memory: How to Make Performers SLiM"),
        Paper(id="ref", title="Reformer: The Efficient Transformer")]


def test_title_head_and_reference_matching():
    assert title_head("Linformer: Self-Attention with Linear Complexity") == "linformer"
    assert title_head("Big Bird — Transformers for Longer Sequences") == "bigbird"
    assert match_reference("Linformer: something else entirely", "2006.04768", REFS).id == "lin"  # arXiv id wins
    assert match_reference("Rethinking attention with performers.", None, REFS).id == "perf"
    assert match_reference("A completely different paper title", None, REFS) is None


def test_resolution_order():
    table = [{"label": "Performer", "title": "Rethinking attention with performers.", "arxiv": None}]
    # table citation disambiguates "Performer" (title matching alone would be ambiguous)
    assert resolve("Performer", REFS, table, {}, "p:self") == ("p:perf", "table citation")
    assert resolve("Reformer", REFS, [], {}, "p:self") == ("p:ref", "reference title")
    assert resolve("FlashAttention", REFS, [], {"flashattention": "p:fa"}, "p:self") == ("p:fa", "proposed in graph")
    assert resolve("FlashAttention", REFS, [], {"flashattention": "p:self"}, "p:self") == (None, "unresolved")
    assert resolve("Megatron-LM", REFS, [], {}, "p:self") == (None, "unresolved")


# ---------------------------------------------------------------- linking


async def test_build_links_baselines(forest, logs):
    project = await forest.build("efficient attention")
    edges = {(e["src"], e["dst"]): e["data"] for e in forest.store.edges(project, ["compares_with"])}
    assert edges[(pid("F"), pid("L"))] == {"name": "Linformer", "via": "proposed in graph"}  # L proposes it
    assert edges[(pid("F"), pid("P"))]["name"] == "Performer"
    megatron = next(d for (s, t), d in edges.items() if d["name"] == "Megatron-LM")
    assert megatron["via"] == "unresolved"
    assert edges[(pid("M"), pid("F"))]["via"] == "proposed in graph"
    assert any(m.startswith("Linked ") and "to method names only" in m for m in logs)
    # shown in the graph and the node details
    g = forest.store.graph(project)
    assert {"source": "p:F", "target": "p:L", "kind": "compares_with"} in g["edges"]
    kinds = {(nb["kind"], nb["direction"]) for nb in forest.store.node(project, pid("L"))["neighbors"]}
    assert ("compares_with", "in") in kinds


async def test_linking_is_idempotent(forest):
    project = await forest.build("efficient attention")
    before = sorted((e["src"], e["dst"]) for e in forest.store.edges(project, ["compares_with"]))
    baselines.link(forest.ctx(project))
    after = sorted((e["src"], e["dst"]) for e in forest.store.edges(project, ["compares_with"]))
    assert before == after


async def test_table_citations_take_priority(forest):
    project = await forest.build("efficient attention")
    store = forest.store
    # FlashAttention's results table cites the Sparse Transformer paper under the label "Performer"
    store.set_meta(project, "table_refs:F", [{"label": "Performer", "title": "Generating Long Sequences with "
                                              "Sparse Transformers", "arxiv": None, "table": "Table 3"}])
    store.add_papers(project, [Paper(id="S9", title="Generating Long Sequences with Sparse Transformers")])
    store.add_citations(project, [CitationLink(citing="F", cited="S9")])
    baselines.link(forest.ctx(project))
    edges = {(e["src"], e["dst"]): e["data"]["via"] for e in store.edges(project, ["compares_with"])}
    assert edges[(pid("F"), pid("S9"))] == "table citation"  # beats "proposed in graph" (P proposes Performer)


# --------------------------------------------------------------- discovery


async def test_baseline_papers_are_discovered(settings, fake_backend, logs):
    settings.target_papers = 3
    settings.baseline_papers = 4
    f = make_forest(settings, fake_backend, logs=logs)
    project = await f.build("efficient attention")
    included = {r.paper.id for r in f.store.papers(project, "included")}
    # the graph compares against Linformer/Performer/Sparse Transformer: they are pulled in
    assert {"L", "P"} & included and len(included) > 3
    assert any("compare against but the graph lacks" in m for m in logs)
    assert all(r.extraction for r in f.store.papers(project, "included"))  # new papers analysed too
    await f.aclose()


async def test_baseline_expansion_can_be_disabled(settings, fake_backend):
    settings.target_papers = 3
    settings.baseline_papers = 0
    f = make_forest(settings, fake_backend)
    project = await f.build("efficient attention")
    assert len(f.store.papers(project, "included")) == 3
    await f.aclose()


async def test_baseline_use_ranks_candidates(forest):
    store = forest.store
    project = store.create_project("p")
    store.add_papers(project, [Paper(id="a", title="A", year=2020, citation_count=500),
                               Paper(id="b", title="B", year=2020, citation_count=100),
                               Paper(id="x", title="X", year=2021, citation_count=10)])
    store.set_relevance(project, {"a": 3, "b": 3})
    store.update_papers(project, ["x"], status="included")
    store.add_edges(project, [("p:x", "p:b", "compares_with", {"name": "B", "via": "reference title"})])
    from research_forest.pipeline import discover
    assert discover.select(forest.ctx(project), 1) == ["b"]  # compared against beats more citations


# ------------------------------------------------------------------ lineage


async def test_relate_prompt_flags_baselines(forest, fake_backend):
    await forest.build("efficient attention")
    prompts = [u for n, u in fake_backend.calls if n == "Relations" and "FlashAttention" in u]
    assert any("[used as a baseline in the citing paper's results]" in u for u in prompts)


# --------------------------------------------------------------- evaluation


def test_baseline_scores():
    from research_forest.evaluation import score_paper, summarise
    ex = Extraction.model_validate({"contribution": "c", "methods_proposed": [], "methods_used": [], "datasets": [],
                                    "limitations": [], "future_work": [], "missing": [], "search_terms": [],
                                    "baselines": ["Linformer", "Performer", "Megatron-LM"]})
    gold = {"id": "x", "title": "T", "datasets": [["X"]], "methods_proposed": [["Y"]],
            "baselines": [["Linformer"], ["Performer"], ["SMYRF", "Smyrf"]]}
    p = score_paper(gold, ex, [], {"Linformer": "Linformer: ...", "Performer": "Rethinking ..."})
    assert p["baseline_recall"] == pytest.approx(2 / 3) and p["baselines_missed"] == ["SMYRF"]
    assert p["baseline_resolution"] == pytest.approx(2 / 3)
    assert score_paper({**gold, "baselines": []}, ex)["baseline_recall"] is None
    s = summarise([p], 0)
    assert s["baseline_recall"] == pytest.approx(0.667, abs=1e-3) and s["baseline_resolution"] == pytest.approx(0.667, abs=1e-3)


async def test_evaluate_reports_baselines(settings, fake_backend, tmp_path):
    import json
    from research_forest.evaluation import evaluate, format_report
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"papers": [
        {"id": "F", "title": "FlashAttention", "datasets": [["WikiText-103"]], "methods_proposed": [["FlashAttention"]],
         "baselines": [["Linformer"], ["Performer"], ["SMYRF"]]}]}))
    f = make_forest(settings, fake_backend)
    report = await evaluate(settings, gold, forest=f, on_log=lambda m: None)
    await f.aclose()
    paper = report["papers"][0]
    assert paper["baseline_recall"] == pytest.approx(2 / 3)
    assert paper["baselines_resolved"] == {"Linformer": "Linformer: Self-Attention with Linear Complexity",
                                           "Performer": "Performer: Rethinking Attention with Kernels"}
    text = format_report(report)
    assert "Baseline recall" in text and "Baselines resolved to papers" in text
