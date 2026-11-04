import http.server
import threading

import pytest

from research_forest.config import Settings
from research_forest.sources import fulltext_worker as worker
from research_forest.sources.fulltext import run_worker

FILLER = " ".join(f"word{i}" for i in range(400))
DOC = (f"<html><head><script>var limitation = 1;</script></head><body><p>Introduction {FILLER}</p>"
       f"<p>We evaluate on the WikiText-103 benchmark and the Long Range Arena dataset.</p><p>{FILLER}</p>"
       f"<p>A main limitation of our approach is that it fails on very long inputs.</p><p>{FILLER}</p></body></html>")


def test_bm25_finds_relevant_passages():
    text = worker.extract_text(DOC.encode(), "text/html")
    assert "var limitation" not in text
    res = worker.search(text, {"datasets": "dataset benchmark evaluate", "limitations": "limitation fails"}, 1)
    assert "WikiText-103" in res["datasets"][0]["text"]
    assert "fails on very long inputs" in res["limitations"][0]["text"]


@pytest.fixture
def http_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body, ctype = (DOC.encode(), "text/html") if self.path == "/paper" else (b"\0" * 10, "image/png")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


async def test_sandboxed_worker_end_to_end(http_server, tmp_path):
    settings = Settings(db_path=str(tmp_path / "db"), fulltext_timeout=20)
    res = await run_worker([f"{http_server}/image", f"{http_server}/paper"],
                           {"limitations": "limitation fails"}, settings, top_k=2)
    assert res["error"] is None and res["source"].endswith("/paper")
    assert "fails on very long inputs" in res["passages"]["limitations"][0]["text"]


async def test_worker_rejects_bad_content(http_server, tmp_path):
    settings = Settings(db_path=str(tmp_path / "db"), fulltext_timeout=20)
    res = await run_worker([f"{http_server}/image", "file:///etc/passwd"], {"x": "y"}, settings)
    assert res["source"] is None
    assert "content-type" in res["error"] and "unsupported scheme" in res["error"]
