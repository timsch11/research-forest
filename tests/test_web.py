import asyncio
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from research_forest.web import create_app
from research_forest.web.server import COOKIE, Jobs

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(settings, forest):
    with TestClient(create_app(settings, forest, token=TOKEN), base_url="http://127.0.0.1:8000") as c:
        c.headers.update(AUTH)
        yield c


def wait_ready(client, pid, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = client.get(f"/api/projects/{pid}").json()
        if not p["running"] and p["status"] in ("ready", "error"):
            return p
        time.sleep(0.05)
    raise TimeoutError


@pytest.fixture
def built(client):
    pid = client.post("/api/projects", json={"prompt": "efficient attention"}).json()["id"]
    wait_ready(client, pid)
    return pid


def test_web_flow(client, built):
    pid = built
    assert client.get(f"/api/projects/{pid}").json()["log"][-1]["msg"] == "done"
    graph = client.get(f"/api/projects/{pid}/graph").json()
    assert any(e["kind"] == "builds_on" for e in graph["edges"])
    hits = client.get(f"/api/projects/{pid}/search", params={"q": "flash"}).json()
    assert hits
    assert client.get(f"/api/projects/{pid}/nodes/{hits[0]['node']}").json()["neighbors"]
    assert client.post(f"/api/projects/{pid}/ask", json={"text": "what is efficient?"}).json()["nodes"]
    client.post(f"/api/projects/{pid}/refine", json={"text": "only since 2020"})
    wait_ready(client, pid)
    assert client.get(f"/api/projects/{pid}").json()["snapshots"]
    assert client.post(f"/api/projects/{pid}/undo").json()["undone"] == "only since 2020"
    assert client.post(f"/api/projects/{pid}/undo").status_code == 400
    assert client.get("/api/projects/999").status_code == 404


def test_log_since_filters(client, built):
    full = client.get(f"/api/projects/{built}").json()["log"]
    later = client.get(f"/api/projects/{built}", params={"since": full[-2]["t"]}).json()["log"]
    assert later == full[-1:]


# ------------------------------------------------------------------- auth


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"},
                                     {"Cookie": f"{COOKIE}=wrong"}])
def test_api_requires_token(settings, forest, headers):
    with TestClient(create_app(settings, forest, token=TOKEN)) as c:
        for method, path in [("GET", "/api/projects"), ("POST", "/api/projects"), ("GET", "/api/projects/1/graph"),
                             ("POST", "/api/projects/1/undo")]:
            assert c.request(method, path, headers=headers, json={"prompt": "x" * 5}).status_code == 401


def test_url_token_is_exchanged_for_cookie(settings, forest):
    with TestClient(create_app(settings, forest, token=TOKEN)) as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert c.get("/api/projects").status_code == 200  # cookie now authenticates
        assert c.get("/?token=nope", follow_redirects=False).status_code == 401


def test_token_is_generated_when_unset(settings, forest):
    a, b = create_app(settings, forest), create_app(settings, forest)
    assert len(a.state.token) >= 24 and a.state.token != b.state.token


def test_static_ui_is_public(settings, forest):
    with TestClient(create_app(settings, forest, token=TOKEN)) as c:
        assert "Research Forest" in c.get("/").text
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


# ----------------------------------------------------------- CSRF / headers


def test_cross_origin_writes_are_rejected(client):
    evil = {"Origin": "http://evil.example"}
    assert client.post("/api/projects", json={"prompt": "efficient attention"}, headers=evil).status_code == 403
    assert client.get("/api/projects", headers=evil).status_code == 200  # reads need the token anyway
    same = {"Origin": "http://127.0.0.1:8000"}
    assert client.post("/api/projects/999/undo", headers=same).status_code == 404  # passes the origin check


def test_security_headers(client):
    for path in ("/", "/api/projects", "/static/app.js"):
        h = client.get(path).headers
        assert "script-src 'self'" in h["content-security-policy"] and "unsafe-eval" not in h["content-security-policy"]
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"
    assert client.get("/api/projects").headers["cache-control"] == "no-store"


def test_index_has_no_inline_scripts():
    from research_forest.web.server import STATIC
    html = (STATIC / "index.html").read_text()
    assert "<script>" not in html and "onclick" not in html.lower()


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("path, body", [
    ("/api/projects", {"prompt": "x"}),
    ("/api/projects", {"prompt": "x" * 2001}),
    ("/api/projects", {"prompt": "valid prompt", "target_papers": 10_000}),
    ("/api/projects", {}),
    ("/api/projects/1/ask", {"text": "y" * 2001}),
    ("/api/projects/1/refine", {"text": 123}),
])
def test_input_validation(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_bad_path_and_query_params(client, built):
    assert client.get("/api/projects/abc").status_code == 422
    assert client.get(f"/api/projects/{built}/nodes/e:notanint").status_code == 404
    assert client.get(f"/api/projects/{built}/nodes/x:1").status_code == 404
    assert client.get(f"/api/projects/{built}/search", params={"q": "q" * 201}).status_code == 422
    assert client.get(f"/api/projects/{built}/search", params={"q": '") OR 1=1 --'}).status_code == 200


# ------------------------------------------------------------------- jobs


async def test_job_limits():
    jobs = Jobs(limit=2)
    gate = asyncio.Event()
    jobs.start(1, gate.wait)
    with pytest.raises(HTTPException) as e:
        jobs.start(1, gate.wait)
    assert e.value.status_code == 409
    jobs.start(2, gate.wait)
    with pytest.raises(HTTPException) as e:
        jobs.start(3, gate.wait)
    assert e.value.status_code == 429
    gate.set()
    await asyncio.sleep(0.01)
    assert not jobs.running(1) and jobs.logs[1][-1]["msg"] == "done"
    jobs.start(3, gate.wait)


async def test_job_errors_are_logged_not_raised():
    jobs = Jobs()

    async def boom():
        raise ValueError("bad thing")

    jobs.start(1, boom)
    await asyncio.sleep(0.01)
    assert jobs.logs[1][-1]["msg"] == "error: ValueError: bad thing"


def test_conflicting_actions_while_running(client, built, fake_backend, monkeypatch):
    gate = threading.Event()
    original = fake_backend.generate

    async def gated(system, user, schema, name):
        if name == "RefinePlan":
            await asyncio.to_thread(gate.wait, 10)
        return await original(system, user, schema, name)

    monkeypatch.setattr(fake_backend, "generate", gated)
    assert client.post(f"/api/projects/{built}/refine", json={"text": "only since 2020"}).json()["started"]
    assert client.get(f"/api/projects/{built}").json()["running"] is True
    assert client.post(f"/api/projects/{built}/undo").status_code == 409
    assert client.post(f"/api/projects/{built}/refine", json={"text": "again"}).status_code == 409
    gate.set()
    assert wait_ready(client, built)["status"] == "ready"
