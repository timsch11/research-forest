from research_forest.store import pid


async def build(forest):
    return await forest.build("efficient attention for long sequences")


def by_title(forest, project):
    return {r.paper.title: r for r in forest.store.papers(project)}


async def test_build_creates_screened_graph(forest):
    project = await build(forest)
    store = forest.store
    assert store.project(project)["status"] == "ready"
    papers = by_title(forest, project)
    assert papers["A Survey of Cooking Recipes"].status != "included"
    included = {t for t, r in papers.items() if r.status == "included"}
    assert "FlashAttention: Fast and Memory-Efficient Exact Attention" in included
    assert len(included) <= forest.settings.target_papers
    assert all(r.extraction for r in store.papers(project, "included"))


async def test_lineage_edges_have_typed_changes(forest):
    project = await build(forest)
    builds = forest.store.edges(project, ["builds_on"])
    assert (pid("F"), pid("T")) in {(e["src"], e["dst"]) for e in builds}
    assert all(e["data"]["change_type"] == "efficiency" and e["data"]["summary"] for e in builds)
    # every in-graph citation was judged exactly once
    included = {r.node for r in forest.store.papers(project, "included")}
    cites = [e for e in forest.store.edges(project, ["cites"]) if e["src"] in included and e["dst"] in included]
    assert cites and all(e["data"]["judged"] for e in cites)


async def test_entities_are_normalised(forest):
    project = await build(forest)
    datasets = {e["name"] for e in forest.store.entities(project, "dataset")}
    assert "WikiText-103" in datasets
    assert not any("WT103" in d for d in datasets)  # alias merged by the LLM grouping
    methods = {e["name"] for e in forest.store.entities(project, "method")}
    assert {"FlashAttention", "Transformer"} <= methods


async def test_gap_filling_uses_snippets(forest, fake_backend):
    project = await build(forest)
    flash = forest.store.papers(project, ids=["F"])[0]
    assert flash.extraction.limitations[0].text == "Relies on custom CUDA kernels"
    assert any("Full-text passages" in u for name, u in fake_backend.calls if name == "Extraction")


async def test_synthesis_links_limitations_and_questions(forest):
    project = await build(forest)
    g = forest.store.graph(project)
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"paper", "method", "dataset", "limitation", "question"} <= kinds
    edge_kinds = {e["kind"] for e in g["edges"]}
    assert {"has_limitation", "motivates", "about", "evaluates_on", "proposes"} <= edge_kinds


async def test_search_and_node_details(forest):
    project = await build(forest)
    hits = forest.store.search(project, "flash")
    assert pid("F") in {h["node"] for h in hits}
    node = forest.store.node(project, pid("F"))
    kinds = {nb["kind"] for nb in node["neighbors"]}
    assert {"builds_on", "proposes", "evaluates_on"} <= kinds


async def test_process_is_incremental(forest, fake_backend):
    project = await build(forest)
    before = len(fake_backend.calls)
    await forest.process(project)
    new_calls = [n for n, _ in fake_backend.calls[before:]]
    assert "Extraction" not in new_calls and "Relations" not in new_calls


async def test_refine_filter_and_undo(forest):
    project = await build(forest)
    before = {r.paper.id for r in forest.store.papers(project, "included")}
    plan = await forest.refine(project, "only work since 2020")
    assert plan.ops[0].year_min == 2020
    after = forest.store.papers(project, "included")
    assert all(r.paper.year >= 2020 for r in after)
    assert forest.store.snapshots(project)
    assert forest.undo(project) == "only work since 2020"
    assert {r.paper.id for r in forest.store.papers(project, "included")} == before
    assert forest.undo(project) is None


async def test_refine_prune(forest):
    project = await build(forest)
    await forest.refine(project, "remove old foundational papers")
    assert forest.store.papers(project, ids=["T"])[0].status == "excluded"


async def test_ask_links_cited_nodes(forest):
    project = await build(forest)
    res = await forest.ask(project, "Which methods make attention efficient?")
    assert res["nodes"] and res["nodes"][0] in res["answer"]
