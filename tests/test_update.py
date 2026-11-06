"""Checking a landscape for work published since the last check."""

import datetime
import time

import httpx
from fastapi.testclient import TestClient

from conftest import make_forest, s2_handler
from research_forest import cli
from research_forest.store import LAST_CHECKED, LATEST_NEW, pid
from research_forest.web import create_app

TODAY = datetime.date.today().isoformat()


def recent(i: str, title: str) -> dict:
    return {"paperId": i, "title": title, "abstract": f"We study {title.lower()}.", "year": int(TODAY[:4]),
            "publicationDate": TODAY, "citationCount": 0, "externalIds": {}}


def handler(seen: list):
    """New work only appears for date-filtered requests: a search hit and an influential citer."""
    def handle(request: httpx.Request) -> httpx.Response:
        path, params = request.url.path, request.url.params
        since = params.get("publicationDateOrYear")
        if since:
            seen.append((path.rsplit("/", 1)[-1], since))
        if path.endswith("/paper/search") and since:
            return httpx.Response(200, json={"data": [recent("Z", "Ring Attention for Linear Long Contexts")]})
        if path.endswith("/citations") and since and path.split("/")[-2] == "F":
            return httpx.Response(200, json={"data": [{"citingPaper": recent("N", "TileFlash: Faster Exact Attention"),
                                                       "contexts": ["we extend FlashAttention"],
                                                       "intents": ["methodology"], "isInfluential": True}]})
        return s2_handler(request)
    return handle


async def test_update_adds_and_marks_new_work_since_the_last_check(settings, fake_backend, logs):
    settings.target_papers = 5
    seen = []
    f = make_forest(settings, fake_backend, handler(seen), logs)
    project = await f.build("efficient attention")
    assert f.store.meta(project, LAST_CHECKED) == TODAY and f.last_checked(project) == TODAY
    f.store.set_meta(project, LAST_CHECKED, "2026-01-01")  # pretend the last check was a while ago
    seen.clear()

    new = await f.update(project)
    assert set(new) == {"Z", "N"}
    assert {s for _, s in seen} == {"2026-01-01:"}  # every request is restricted to new work
    assert {kind for kind, _ in seen} == {"search", "citations"}
    assert f.store.meta(project, LATEST_NEW) == {"since": "2026-01-01", "papers": ["N", "Z"]}
    assert f.store.meta(project, LAST_CHECKED) == TODAY
    flagged = {n["id"] for n in f.store.graph(project)["nodes"] if n.get("new")}
    assert flagged == {pid("N"), pid("Z")}
    assert all(r.extraction for r in f.store.papers(project, "included"))  # new papers are analysed
    assert any("Found 2 new papers published since 2026-01-01" in m for m in logs)

    assert f.undo(project) == "check for new work since 2026-01-01"
    included = {r.paper.id for r in f.store.papers(project, "included")}
    assert not {"Z", "N"} & included
    assert f.store.meta(project, LAST_CHECKED) == "2026-01-01"  # the next check finds them again
    assert f.store.meta(project, LATEST_NEW) is None
    await f.aclose()


async def test_update_without_new_work_skips_synthesis(settings, fake_backend, logs):
    def nothing_new(request: httpx.Request) -> httpx.Response:
        if "publicationDateOrYear" in request.url.params:
            return httpx.Response(200, json={"data": []})
        return s2_handler(request)

    settings.target_papers = 3
    f = make_forest(settings, fake_backend, nothing_new, logs)
    project = await f.build("efficient attention")
    before, calls = {r.paper.id for r in f.store.papers(project, "included")}, len(fake_backend.calls)
    assert await f.update(project) == []
    assert {r.paper.id for r in f.store.papers(project, "included")} == before
    assert not [n for n, _ in fake_backend.calls[calls:] if n == "Synthesis"]  # nothing changed: no synthesis
    assert f.store.meta(project, LATEST_NEW) == {"since": TODAY, "papers": []}
    assert any(f"No new relevant papers since {TODAY}" in m for m in logs)
    await f.aclose()


def test_update_endpoint_and_project_fields(settings, fake_backend):
    settings.target_papers = 3
    f = make_forest(settings, fake_backend, handler([]))
    with TestClient(create_app(settings, f, token="t")) as c:
        h = {"Authorization": "Bearer t"}
        project = c.post("/api/projects", json={"prompt": "efficient attention"}, headers=h).json()["id"]

        def wait() -> dict:
            for _ in range(400):
                if not (p := c.get(f"/api/projects/{project}", headers=h).json())["running"]:
                    return p
                time.sleep(0.05)
            raise TimeoutError

        assert wait()["last_checked"] == TODAY
        f.store.set_meta(project, LAST_CHECKED, "2026-01-01")
        assert c.post(f"/api/projects/{project}/update", headers=h).json() == {"started": True}
        p = wait()
        assert p["latest_new"]["since"] == "2026-01-01" and "Z" in p["latest_new"]["papers"]
        assert p["progress"]["job"] == "update"
        assert c.post("/api/projects/999/update", headers=h).status_code == 404


def test_cli_update(tmp_path, capsys, monkeypatch, settings, fake_backend):
    from research_forest import engine

    async def fake_update(self, project):
        return []

    monkeypatch.setattr(engine.Forest, "update", fake_update)
    db = str(tmp_path / "cli.db")
    store = engine.Store(db)
    project = store.create_project("efficient attention")
    cli.main(["--db", db, "update", str(project)])
    assert f"0 new papers since {TODAY}" in capsys.readouterr().out
