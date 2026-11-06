import pytest

from research_forest.models import CitationLink, Extraction, Paper, Scope
from research_forest.store import Store, entity_key, fts_query, pid, valid_node


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "s.db"))


@pytest.fixture
def project(store):
    p = store.create_project("prompt")
    store.set_scope(p, Scope(title="T", summary="S", search_queries=["q"], seed_papers=[],
                             include_criteria="i", exclude_criteria="e", year_min=None))
    store.add_papers(p, [Paper(id=i, title=f"Paper {i} about attention", abstract=f"abstract {i}", year=2020 + n)
                         for n, i in enumerate("ABC")])
    store.update_papers(p, ["A", "B", "C"], status="included")
    return p


def test_projects(store, project):
    assert store.project(project)["title"] == "T"
    assert store.scope(project).summary == "S"
    store.set_status(project, "ready")
    assert [p["status"] for p in store.projects()] == ["ready"]
    assert store.project(999) is None


def test_add_papers_reports_new_and_refreshes_metadata(store, project):
    assert store.add_papers(project, [Paper(id="A", title="A2", abstract="new"), Paper(id="D", title="D")]) == ["D"]
    assert store.papers(project, ids=["A"])[0].paper.abstract == "new"
    store.add_papers(project, [Paper(id="A", title="A3")])  # no abstract: keep the richer record
    assert store.papers(project, ids=["A"])[0].paper.title == "A2"
    assert store.papers(project, ids=[]) == []


def test_entities_upsert_merge_and_prune(store, project):
    m1 = store.upsert_entity(project, "method", "Flash Attention", "first")
    assert store.upsert_entity(project, "method", "flash-attention", "second") == m1  # same key
    assert store.entities(project, "method")[0]["description"] == "first"
    m2 = store.upsert_entity(project, "method", "FA")
    store.add_edges(project, [(pid("A"), m1, "uses", None), (pid("B"), m2, "uses", None),
                              (pid("A"), m2, "uses", None)])
    store.merge_entities(project, m1, [m2], "FlashAttention")
    ents = store.entities(project, "method")
    assert [e["name"] for e in ents] == ["FlashAttention"]
    assert {(e["src"], e["dst"]) for e in store.edges(project, ["uses"])} == {(pid("A"), m1), (pid("B"), m1)}
    store.exclude_papers(project, ["A", "B"])
    assert store.entities(project, "method") == []  # orphaned once its papers are gone
    assert store.edges(project, ["uses"]) == []


def test_exclude_keeps_citations_but_drops_lineage(store, project):
    store.add_citations(project, [CitationLink(citing="B", cited="A", contexts=["ctx"])])
    store.add_edges(project, [(pid("B"), pid("A"), "builds_on", {"change_type": "data"})])
    store.exclude_papers(project, ["A"])
    assert store.edges(project, ["builds_on"]) == []
    assert len(store.edges(project, ["cites"])) == 1  # raw citation data is kept for later re-inclusion
    assert "p:A" not in {n["id"] for n in store.graph(project)["nodes"]}


def test_graph_only_contains_connected_entities(store, project):
    used = store.upsert_entity(project, "dataset", "WikiText-103")
    store.upsert_entity(project, "dataset", "Unused")
    store.add_edges(project, [(pid("A"), used, "evaluates_on", {"usage": "evaluation"})])
    g = store.graph(project)
    labels = {n["label"] for n in g["nodes"]}
    assert "WikiText-103" in labels and "Unused" not in labels
    assert g["edges"] == [{"source": "p:A", "target": used, "kind": "evaluates_on"}]


@pytest.mark.parametrize("query", ['"', "a AND", "NEAR(a b)", "*", "-x", "col:value", "') OR 1=1 --", "ü日本"])
def test_fts_query_is_injection_safe(store, project, query):
    store.reindex(project)
    store.search(project, query)  # must not raise
    store.search(project, query, mode="OR")


def test_fts_query_format():
    assert fts_query('flash "attention"') == '"flash"* AND "attention"*'
    assert fts_query("what is the attention", "OR") == '"attention"*'
    assert fts_query("a") == ""


def test_search_ranks_and_filters_kinds(store, project):
    store.upsert_entity(project, "method", "Attention pooling")
    store.reindex(project)
    assert {h["kind"] for h in store.search(project, "attention")} == {"paper", "method"}
    assert {h["kind"] for h in store.search(project, "attention", kinds=["method"])} == {"method"}
    hit = store.search(project, "abstract", kinds=["paper"])[0]
    assert "\x02" in hit["snippet"] and "\x03" in hit["snippet"]


def test_reindex_sanitises_text(store, project):
    store.add_papers(project, [Paper(id="Z", title="Bad\x02title", abstract="x\x03y")])
    store.update_papers(project, ["Z"], status="included")
    store.reindex(project)
    hit = [h for h in store.search(project, "badtitle") if h["node"] == "p:Z"][0]
    assert "\x02" not in hit["title"]


@pytest.mark.parametrize("node, ok", [("p:abc123", True), ("e:42", True), ("e:x", False), ("p:", False),
                                      ("x:1", False), ("p:a b", False), ("e:1;DROP", False)])
def test_node_validation(store, project, node, ok):
    assert valid_node(node) is ok
    if not ok:
        assert store.node(project, node) is None


def test_node_details(store, project):
    store.set_extraction(project, "A", Extraction(contribution="c", methods_proposed=[], methods_used=[],
                                                  datasets=[], limitations=[], future_work=[], missing=[],
                                                  search_terms=[]))
    store.add_edges(project, [(pid("B"), pid("A"), "builds_on", {"change_type": "data"})])
    info = store.node(project, pid("A"))
    assert info["extraction"]["contribution"] == "c"
    assert info["neighbors"] == [{"node": "p:B", "label": "Paper B about attention", "kind": "builds_on",
                                  "direction": "in", "data": {"change_type": "data"}}]


def test_snapshots_roundtrip(store, project):
    v1 = store.snapshot(project, "first")
    store.exclude_papers(project, ["A"])
    v2 = store.snapshot(project, "second")
    store.exclude_papers(project, ["B"])
    assert [s["version"] for s in store.snapshots(project)] == [v2, v1]
    assert store.restore(project) == "second"
    assert {r.paper.id for r in store.papers(project, "included")} == {"B", "C"}
    assert store.restore(project, v1) == "first"
    assert {r.paper.id for r in store.papers(project, "included")} == {"A", "B", "C"}
    assert store.snapshots(project) == [] and store.restore(project) is None


def test_restore_ignores_unknown_columns(store, project):
    import json
    store.snapshot(project, "s")
    row = store.q("SELECT data FROM snapshots WHERE project = ?", (project,))[0]
    dump = json.loads(row["data"])
    for r in dump["papers"]:
        r["evil); DROP TABLE papers; --"] = 1
    with store.tx() as c:
        c.execute("UPDATE snapshots SET data = ? WHERE project = ?", (json.dumps(dump), project))
    assert store.restore(project) == "s"
    assert len(store.papers(project)) == 3


def test_entity_key():
    assert entity_key("Flash-Attention 2") == entity_key("flash attention2") == "flashattention2"
    assert entity_key("+++") == "+++"


def test_projects_are_isolated(store, project):
    other = store.create_project("other")
    store.add_papers(other, [Paper(id="A", title="Different")])
    assert store.papers(project, ids=["A"])[0].paper.title.startswith("Paper A")
    assert len(store.papers(other)) == 1


def test_delete_project_removes_everything(store, project):
    other = store.create_project("other")
    store.add_papers(other, [Paper(id="A", title="Different")])
    store.add_citations(project, [CitationLink(citing="B", cited="A")])
    store.upsert_entity(project, "method", "FlashAttention")
    store.snapshot(project, "x")
    store.set_meta(project, "k", 1)
    assert store.snapshots(project) and store.q("SELECT 1 FROM edges WHERE project = ?", (project,))
    store.reindex(project)
    assert store.delete_project(project)
    assert store.project(project) is None and store.papers(project) == []
    assert store.meta(project, "k") is None and store.snapshots(project) == []
    assert not store.q("SELECT 1 FROM edges WHERE project = ?", (project,))
    assert not store.q("SELECT 1 FROM entities WHERE project = ?", (project,))
    assert not store.q("SELECT 1 FROM fts WHERE project = ?", (project,))
    assert len(store.papers(other)) == 1  # other landscapes are untouched
    assert not store.delete_project(project)
