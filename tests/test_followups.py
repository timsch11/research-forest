"""Recent follow-up work found through influential citations of key graph papers."""

import datetime

import httpx
from conftest import make_forest, op, s2_handler

from research_forest.models import Paper
from research_forest.pipeline import followups
from research_forest.pipeline.discover import influential_per_year, rank_score
from research_forest.sources.semantic_scholar import SemanticScholar
from research_forest.store import pid

YEAR = datetime.date.today().year

# Citing papers outside the corpus: id -> (title, year, influential, cited seed ids)
CITERS = {
    "N1": ("HybridFlash: Selective State Spaces Meet IO-Aware Attention", YEAR, True, {"F", "M"}),
    "N2": ("TileFlash: Faster Exact Attention on New Accelerators", YEAR, True, {"F"}),
    "N3": ("A Benchmark Mentioning FlashAttention in Passing", YEAR, False, {"F", "M"}),
    "N4": ("Cooking Recipes Generated with Fast Attention", YEAR, True, {"F"}),  # screened out
    "N5": ("An Older Follow-up of FlashAttention", YEAR - 4, True, {"F"}),  # too old
}


def citer(i: str) -> dict:
    title, year, *_ = CITERS[i]
    return {"paperId": i, "title": title, "abstract": f"We study {title.lower()}.", "year": year,
            "citationCount": 0, "influentialCitationCount": 0, "externalIds": {}}


def handler(seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        path, params = request.url.path, request.url.params
        if path.endswith("/citations"):
            seed = path.split("/")[-2]
            seen.append((seed, dict(params)))
            year_min = int(params["publicationDateOrYear"].rstrip(":")) if "publicationDateOrYear" in params else 0
            rows = [{"citingPaper": citer(i), "contexts": [f"we extend {seed}", "following {seed}"],
                     "intents": ["methodology"], "isInfluential": infl}
                    for i, (_, year, infl, cited) in CITERS.items() if seed in cited and year >= year_min]
            return httpx.Response(200, json={"data": rows})
        return s2_handler(request)
    return handle


async def test_links_filter_influential_citations_server_and_client_side():
    seen = []
    s2 = SemanticScholar(rps=0, transport=httpx.MockTransport(handler(seen)))
    res = await s2.links("F", "citations", 1000, year_min=YEAR - 1, influential_only=True)
    assert {p.id for p, _ in res} == {"N1", "N2", "N4"}  # N3 isn't influential, N5 too old
    assert all(link.influential and link.cited == "F" for _, link in res)
    assert seen[0][1]["publicationDateOrYear"] == f"{YEAR - 1}:" and seen[0][1]["limit"] == "1000"
    await s2.aclose()


async def test_build_adds_recent_followups(settings, fake_backend, logs):
    settings.target_papers = 5
    settings.followup_papers = 2
    seen = []
    f = make_forest(settings, fake_backend, handler(seen), logs)
    project = await f.build("efficient attention")
    included = {r.paper.id for r in f.store.papers(project, "included")}
    assert {"N1", "N2"} <= included and not {"N3", "N4", "N5"} & included
    assert {s for s, _ in seen} <= included - {"N1", "N2"}  # only graph papers were scanned
    record = f.store.meta(project, followups.FOLLOWUPS)
    assert record["N1"] == ["F", "M"] and record["N2"] == ["F"]
    assert f.store.node(project, pid("N1"))["followup"] and not f.store.node(project, pid("F"))["followup"]
    assert all(r.extraction for r in f.store.papers(project, "included"))  # follow-ups are analysed too
    influential = [e for e in f.store.edges(project, ["cites"]) if e["src"] == pid("N1")]
    assert {e["dst"] for e in influential} == {pid("F"), pid("M")} and all(e["data"]["influential"] for e in influential)
    assert any("cite graph papers influentially (1 cite several)" in m for m in logs)
    await f.aclose()


async def test_refine_followups_of_chosen_papers(settings, fake_backend):
    settings.target_papers = 5
    seen = []
    f = make_forest(settings, fake_backend, handler(seen))
    project = await f.build("efficient attention")
    papers = sorted((r.paper for r in f.store.papers(project, "included")), key=lambda p: (p.year or 0, p.title))
    ref = next(i for i, p in enumerate(papers) if p.id == "M")
    fake_backend.next_plan = {"explanation": "x", "ops": [op("followups", papers=[ref], limit=3)]}
    await f.refine(project, "find recent work building on Mamba")
    assert [s for s, _ in seen] == ["M"]
    assert "N1" in {r.paper.id for r in f.store.papers(project, "included")}
    await f.aclose()


async def test_no_followups_without_influential_citations(forest, logs):
    forest.on_log = lambda _p, m: logs.append(m)
    project = await forest.build("efficient attention")  # the default corpus has no influential citers
    assert await followups.find(forest.ctx(project), 3) == []
    assert any("No recent influential citations" in m for m in logs)


def test_seeds_prefer_papers_the_graph_builds_on(forest):
    store = forest.store
    project = store.create_project("p")
    store.add_papers(project, [Paper(id=i, title=i, influential_count=n) for i, n in [("A", 0), ("B", 0), ("C", 50)]])
    store.update_papers(project, ["A", "B", "C"], status="included")
    store.add_edges(project, [(pid("B"), pid("A"), "builds_on", {}), (pid("C"), pid("A"), "builds_on", {})])
    assert followups.seeds(forest.ctx(project), 2) == ["A", "C"]


def test_influential_citations_lift_the_ranking():
    plain = Paper(id="a", title="a", year=YEAR - 2, citation_count=300)
    used = plain.model_copy(update={"influential_count": 60})
    assert influential_per_year(used, YEAR) == 20
    assert rank_score(used, 2, YEAR) > rank_score(plain, 2, YEAR)
    assert rank_score(plain, 3, YEAR) > rank_score(used, 2, YEAR)  # relevance still comes first
