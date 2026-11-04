import time

from fastapi.testclient import TestClient

from research_forest.web import create_app


def wait_ready(client, pid, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = client.get(f"/api/projects/{pid}").json()
        if not p["running"] and p["status"] in ("ready", "error"):
            return p
        time.sleep(0.05)
    raise TimeoutError


def test_web_flow(settings, forest):
    with TestClient(create_app(settings, forest)) as client:
        assert "Research Forest" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200

        pid = client.post("/api/projects", json={"prompt": "efficient attention"}).json()["id"]
        project = wait_ready(client, pid)
        assert project["status"] == "ready" and project["log"][-1]["msg"] == "done"

        graph = client.get(f"/api/projects/{pid}/graph").json()
        assert any(e["kind"] == "builds_on" for e in graph["edges"])

        hits = client.get(f"/api/projects/{pid}/search", params={"q": "flash"}).json()
        assert hits
        node = client.get(f"/api/projects/{pid}/nodes/{hits[0]['node']}").json()
        assert node["neighbors"]

        assert client.post(f"/api/projects/{pid}/ask", json={"text": "what is efficient?"}).json()["nodes"]

        client.post(f"/api/projects/{pid}/refine", json={"text": "only since 2020"})
        wait_ready(client, pid)
        assert client.get(f"/api/projects/{pid}").json()["snapshots"]
        assert client.post(f"/api/projects/{pid}/undo").json()["undone"] == "only since 2020"
        assert client.post(f"/api/projects/{pid}/undo").status_code == 400
        assert client.get("/api/projects/999").status_code == 404
