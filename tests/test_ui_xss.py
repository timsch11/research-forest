"""Browser-level XSS test: hostile data written straight into the DB must never execute.

Runs only where Node.js and Playwright (with Chromium) are available.
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time

import pytest
import uvicorn
from conftest import FakeBackend, make_forest

from research_forest.models import Extraction, LimitationItem, Paper, Scope
from research_forest.store import pid

XSS = '<img src=x onerror="window.__xss=1">'


def _node_env() -> dict:
    env = dict(os.environ)
    if shutil.which("npm") and "NODE_PATH" not in env:  # find a globally installed playwright
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        env["NODE_PATH"] = root
    return env


def _playwright_available() -> bool:
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('playwright')"], capture_output=True,
                          env=_node_env()).returncode == 0


pytestmark = pytest.mark.skipif(not _playwright_available(), reason="node + playwright not available")

SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  const [url] = process.argv.slice(2);
  const b = await chromium.launch();
  const page = await b.newPage();
  const out = { dialogs: 0, errors: [], cspViolations: 0 };
  page.on('dialog', d => { out.dialogs++; d.dismiss(); });
  page.on('pageerror', e => out.errors.push(String(e)));
  page.on('console', m => { if (/Content Security Policy/.test(m.text())) out.cspViolations++; });
  await page.goto(url);
  await page.waitForTimeout(800);
  await page.fill('#search', 'hostile'); await page.waitForTimeout(500);
  await page.click('#results li >> nth=0'); await page.waitForTimeout(500);
  await page.click('#questions li >> nth=0'); await page.waitForTimeout(400);
  await page.evaluate(() => document.querySelector('[data-node^="p:"]')?.click());
  await page.waitForTimeout(400);
  out.xss = await page.evaluate(() => window.__xss === 1);
  out.injectedImgs = await page.evaluate(() => document.querySelectorAll('img[src="x"]').length);
  out.hrefs = await page.evaluate(() => [...document.querySelectorAll('a[href]')].map(a => a.getAttribute('href')));
  out.detailsText = await page.evaluate(() => document.querySelector('#details').innerText);
  out.graphNodes = await page.evaluate(() => document.querySelectorAll('#cy canvas').length);
  console.log(JSON.stringify(out));
  await b.close();
})();
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def hostile_server(settings):
    forest = make_forest(settings, FakeBackend())
    store = forest.store
    p = store.create_project(f"hostile {XSS}")
    store.set_scope(p, Scope(title=f"hostile {XSS}", summary="s", search_queries=[], seed_papers=[],
                             include_criteria="i", exclude_criteria="e", year_min=None))
    # bypass ingestion sanitisers on purpose: the UI itself must be safe
    store.add_papers(p, [Paper(id="X", title=f"hostile paper {XSS}", abstract=XSS, year=2024, venue=XSS,
                               authors=[XSS], url="javascript:window.__xss=1", pdf_url="javascript:alert(1)")])
    store.update_papers(p, ["X"], status="included")
    store.set_extraction(p, "X", Extraction.model_construct(
        contribution=XSS, methods_proposed=[], methods_used=[], datasets=[], future_work=[XSS], missing=[],
        search_terms=[], limitations=[LimitationItem.model_construct(text=XSS, source="stated")]))
    m = store.upsert_entity(p, "method", f"hostile method {XSS}", XSS)
    q = store.upsert_entity(p, "question", f"hostile question {XSS}", XSS)
    store.add_edges(p, [(pid("X"), m, "proposes", None), (q, pid("X"), "about", None),
                        (pid("X"), pid("X"), "builds_on", {"change_type": XSS, "summary": XSS, "component": XSS})])
    store.reindex(p)
    store.set_status(p, "ready")

    from research_forest.web import create_app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings, forest, token="tok"), port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/?token=tok"
    server.should_exit = True


def test_hostile_content_does_not_execute(hostile_server, tmp_path):
    script = tmp_path / "xss.js"
    script.write_text(SCRIPT)
    proc = subprocess.run(["node", str(script), hostile_server], capture_output=True, text=True, timeout=90,
                          cwd=str(tmp_path), env=_node_env())
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["xss"] is False and out["dialogs"] == 0 and out["injectedImgs"] == 0
    assert not any(h.lower().startswith("javascript") for h in out["hrefs"])
    assert "<img" in out["detailsText"]  # rendered as visible text, not markup
    assert out["errors"] == [] and out["cspViolations"] == 0 and out["graphNodes"] > 0
