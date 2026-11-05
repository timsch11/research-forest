import http.server
import socket
import sys
import threading

import httpx
import pytest

from research_forest.config import Settings
from research_forest.models import Paper
from research_forest.sources import fulltext
from research_forest.sources import fulltext_worker as worker
from research_forest.sources.fulltext import PassageFinder, run_worker

FILLER = " ".join(f"word{i}" for i in range(400))
DOC = (f"<html><head><script>var limitation = 1;</script></head><body><p>Introduction {FILLER}</p>"
       f"<p>We evaluate on the WikiText-103 benchmark and the Long Range Arena dataset.</p><p>{FILLER}</p>"
       f"<p>A main limitation of our approach is that it fails on very long inputs.</p><p>{FILLER}</p></body></html>")


# ------------------------------------------------------------- BM25 / text


def test_bm25_finds_relevant_passages():
    text = worker.extract_text(DOC.encode(), "text/html")
    assert "var limitation" not in text
    res = worker.search(text, {"datasets": "dataset benchmark evaluate", "limitations": "limitation fails"}, 1)
    assert "WikiText-103" in res["datasets"][0]["text"]
    assert "fails on very long inputs" in res["limitations"][0]["text"]


def test_bm25_no_match_and_empty_text():
    assert worker.search("alpha beta gamma " * 50, {"x": "zebra"}, 3) == {"x": []}
    assert worker.search("", {"x": "zebra"}, 3) == {"x": []}


def test_chunking_overlaps_and_dehyphenates():
    words = [f"w{i}" for i in range(300)]
    chunks = worker.chunk(" ".join(words))
    assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w299"
    assert len(chunks) == 4
    assert worker.chunk("trans-\nformer") == ["transformer"]


def test_top_k_skips_adjacent_windows():
    text = " ".join(["filler"] * 2000 + ["needle"] * 5 + ["filler"] * 2000 + ["needle"] * 5)
    hits = worker.BM25(worker.chunk(text)).top("needle", 5)
    idx = sorted(i for _, i in hits)
    assert all(b - a > 1 for a, b in zip(idx, idx[1:]))


def test_plain_text_and_pdf_magic():
    assert worker.extract_text(b"hello world", "text/plain") == "hello world"
    with pytest.raises(Exception):
        worker.extract_text(b"%PDF-1.4 garbage", "application/octet-stream")


def test_resource_limits_are_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)  # simulate Windows: `import resource` fails
    worker.limit_resources(512, 10)  # must not raise


# ---------------------------------------------------------------- SSRF


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://localhost/x", "http://10.0.0.5/x", "http://192.168.1.1/x",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/x", "http://[::ffff:127.0.0.1]/x",
    "http://0.0.0.0/x", "http://100.64.0.1/x", "http://[fd00::1]/x", "http://224.0.0.1/x",
])
def test_check_url_blocks_internal_addresses(url):
    with pytest.raises(ValueError, match="non-public|cannot resolve"):
        worker.check_url(url)


@pytest.mark.parametrize("url, msg", [
    ("ftp://example.org/x", "scheme"), ("file:///etc/passwd", "scheme"), ("gopher://x/", "scheme"),
    ("https://8.8.8.8:8443/x", "port"), ("http://user:pw@8.8.8.8/x", "invalid host"), ("https:///x", "invalid host"),
])
def test_check_url_rejects_bad_urls(url, msg):
    with pytest.raises(ValueError, match=msg):
        worker.check_url(url)


def test_check_url_allows_public_and_uses_dns(monkeypatch):
    worker.check_url("https://8.8.8.8/paper.pdf")
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p: [(None, None, None, None, ("93.184.215.14", 0))])
    worker.check_url("https://example.org/paper.pdf")
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p: [(None, None, None, None, ("93.184.215.14", 0)),
                                                             (None, None, None, None, ("10.1.1.1", 0))])
    with pytest.raises(ValueError, match="non-public"):  # any private answer (DNS rebinding) blocks
        worker.check_url("https://example.org/paper.pdf")


def test_check_url_unresolvable_depends_on_proxy(monkeypatch):
    def fail(h, p):
        raise socket.gaierror("no dns")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="cannot resolve"):
        worker.check_url("https://example.org/x")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    worker.check_url("https://example.org/x")  # resolution is the proxy's job


def mock_client(routes: dict[str, httpx.Response]) -> httpx.Client:
    def handler(request):
        return routes.get(str(request.url), httpx.Response(404))
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_redirect_to_internal_address_is_blocked():
    client = mock_client({"https://8.8.8.8/a": httpx.Response(302, headers={"location": "http://169.254.169.254/"})})
    with pytest.raises(ValueError, match="non-public"):
        worker.download("https://8.8.8.8/a", 10**6, 5, client=client)


def test_redirects_are_followed_and_limited():
    ok = {"https://8.8.8.8/a": httpx.Response(301, headers={"location": "/b"}),
          "https://8.8.8.8/b": httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hello")}
    assert worker.download("https://8.8.8.8/a", 10**6, 5, client=mock_client(ok)) == (b"hello", "text/plain")
    loop = {"https://8.8.8.8/a": httpx.Response(302, headers={"location": "/a"})}
    with pytest.raises(ValueError, match="too many redirects"):
        worker.download("https://8.8.8.8/a", 10**6, 5, client=mock_client(loop))


@pytest.mark.parametrize("response, msg", [
    (httpx.Response(200, headers={"content-type": "image/png"}, content=b"x"), "content-type"),
    (httpx.Response(200, headers={"content-type": "text/plain", "content-length": "999999"}, content=b"x"), "too large"),
    (httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5000), "too large"),
])
def test_download_rejects_bad_responses(response, msg):
    with pytest.raises(ValueError, match=msg):
        worker.download("https://8.8.8.8/a", 1000, 5, client=mock_client({"https://8.8.8.8/a": response}))


def test_download_size_limit_applies_to_decompressed_bytes():
    import gzip
    bomb = gzip.compress(b"\0" * 200_000)
    response = httpx.Response(200, headers={"content-type": "text/plain", "content-encoding": "gzip"}, content=bomb)
    with pytest.raises(ValueError, match="too large"):
        worker.download("https://8.8.8.8/a", 50_000, 5, client=mock_client({"https://8.8.8.8/a": response}))


# ------------------------------------------------------- worker process


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


def local_settings(tmp_path, **kw):
    """Settings for tests against a local server (SSRF guard relaxed)."""
    return Settings(**{"db_path": str(tmp_path / "db"), "fulltext_timeout": 20,
                       "fulltext_allow_private": True, **kw})


async def test_sandboxed_worker_end_to_end(http_server, tmp_path):
    res = await run_worker([f"{http_server}/image", f"{http_server}/paper"],
                           {"limitations": "limitation fails"}, local_settings(tmp_path), top_k=2)
    assert res["error"] is None and res["source"].endswith("/paper")
    assert "fails on very long inputs" in res["passages"]["limitations"][0]["text"]


async def test_worker_blocks_local_server_by_default(http_server, tmp_path):
    res = await run_worker([f"{http_server}/paper"], {"x": "limitation"},
                           Settings(db_path=str(tmp_path / "db"), fulltext_timeout=20))
    assert res["source"] is None and "not allowed" in res["error"]


async def test_worker_rejects_bad_content(http_server, tmp_path):
    res = await run_worker([f"{http_server}/image", "file:///etc/passwd"], {"x": "y"}, local_settings(tmp_path))
    assert res["source"] is None
    assert "content-type" in res["error"] and "unsupported scheme" in res["error"]


async def test_worker_environment_is_minimal(tmp_path, monkeypatch):
    script = tmp_path / "w.py"
    script.write_text("import json, os, sys; sys.stdin.read(); "
                      "print(json.dumps({'source': os.getcwd(), 'error': ','.join(sorted(os.environ)), "
                      "'passages': {}}))")
    monkeypatch.setattr(fulltext, "WORKER", script)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("S2_API_KEY", "s2-secret")
    res = await run_worker(["https://x"], {"x": "y"}, local_settings(tmp_path))
    assert "OPENAI_API_KEY" not in res["error"] and "S2_API_KEY" not in res["error"]
    assert "rf-fulltext-" in res["source"]


@pytest.mark.parametrize("output, ok", [
    ("not json", False),
    ('{"source": "u", "error": null, "passages": {"x": [{"text": "a"}, "junk"], "evil": [{"text": "b"}]}}', True),
])
async def test_worker_output_is_validated(tmp_path, monkeypatch, output, ok):
    script = tmp_path / "w.py"
    script.write_text(f"import sys; sys.stdin.read(); sys.stdout.write({output!r})")
    monkeypatch.setattr(fulltext, "WORKER", script)
    res = await run_worker(["https://x"], {"x": "y"}, local_settings(tmp_path))
    if ok:
        assert res["passages"] == {"x": [{"text": "a"}]}  # unknown fields and malformed hits dropped
    else:
        assert res["error"] == "worker returned invalid JSON"


async def test_worker_timeout_and_crash(tmp_path, monkeypatch):
    script = tmp_path / "w.py"
    script.write_text("import time; time.sleep(30)")
    monkeypatch.setattr(fulltext, "WORKER", script)
    monkeypatch.setattr(fulltext, "WORKER_GRACE", 0)
    res = await run_worker(["https://x"], {"x": "y"}, local_settings(tmp_path, fulltext_timeout=0.5))
    assert res["error"] == "worker timed out"
    script.write_text("raise SystemExit(7)")
    res = await run_worker(["https://x"], {"x": "y"}, local_settings(tmp_path))
    assert "worker exit 7" in res["error"]


# ------------------------------------------------------- passage finder


class FakeS2:
    def __init__(self, snippets):
        self._snippets, self.queries = snippets, []

    async def snippets(self, query, paper_id, limit=5):
        self.queries.append(query)
        return self._snippets


async def test_passage_finder_uses_snippets_first_without_arxiv_html(tmp_path, monkeypatch):
    called = []

    async def fake_worker(*a, **k):
        called.append(a)
        return {"passages": {}}

    monkeypatch.setattr(fulltext, "run_worker", fake_worker)
    finder = PassageFinder(FakeS2(["snippet text"]), local_settings(tmp_path))
    paper = Paper(id="p", title="t", pdf_url="https://example.org/p.pdf")  # no arXiv rendering
    found = await finder.find(paper, ["limitations"], ["kernel"])
    assert found == {"limitations": ["snippet text"]} and not called
    assert "kernel" in finder.s2.queries[0]


async def test_passage_finder_reads_arxiv_html_before_snippets(tmp_path, monkeypatch):
    seen = {}

    async def fake_worker(urls, queries, settings, top_k=4):
        seen.update(urls=urls, queries=queries)
        return {"source": urls[0], "error": None, "passages": {"datasets": [{"text": "Table 2: SST-2 | IMDB"}]}}

    monkeypatch.setattr(fulltext, "run_worker", fake_worker)
    s2 = FakeS2(["snippet text"])
    finder = PassageFinder(s2, local_settings(tmp_path))
    found = await finder.find(Paper(id="p", title="t", arxiv_id="2006.04768"), ["datasets", "limitations"], [])
    assert seen["urls"][0] == "https://arxiv.org/html/2006.04768"
    assert found["datasets"] == ["Table 2: SST-2 | IMDB"]  # from the HTML, not the snippet
    assert found["limitations"] == ["snippet text"] and len(s2.queries) == 1  # snippets only for what is left


async def test_passage_finder_falls_back_to_worker(tmp_path, monkeypatch):
    seen = {}

    async def fake_worker(urls, queries, settings, top_k=4):
        seen.update(urls=urls, queries=queries)
        return {"source": urls[0], "error": None, "passages": {"datasets": [{"text": "uses ImageNet"}]}}

    monkeypatch.setattr(fulltext, "run_worker", fake_worker)
    finder = PassageFinder(FakeS2([]), local_settings(tmp_path))
    paper = Paper(id="p", title="t", arxiv_id="2205.14135", pdf_url="https://arxiv.org/pdf/2205.14135")
    assert await finder.find(paper, ["datasets"], []) == {"datasets": ["uses ImageNet"]}
    assert seen["urls"] == ["https://arxiv.org/html/2205.14135", "https://arxiv.org/pdf/2205.14135"]  # deduplicated


async def test_passage_finder_without_urls_or_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(fulltext, "run_worker", lambda *a, **k: pytest.fail("worker must not run"))
    finder = PassageFinder(FakeS2([]), local_settings(tmp_path))
    assert await finder.find(Paper(id="p", title="t"), ["datasets"], []) == {}
    finder = PassageFinder(FakeS2([]), local_settings(tmp_path, fulltext_enabled=False))
    assert await finder.find(Paper(id="p", title="t", arxiv_id="2205.14135"), ["datasets"], []) == {}


# ------------------------------------------------------------ arXiv HTML

LATEXML = """<html><head><title>T</title><style>.x{}</style></head><body>
<header class="desktop_header">arXiv navigation</header>
<article class="ltx_document">
<div class="ltx_authors"><span class="ltx_role_author">Jane Doe</span> <span class="ltx_role_affiliation">Uni</span></div>
<section class="ltx_section"><h2>4 Experiments</h2>
<p>We evaluate on <span class="ltx_text">WikiHop</span> and HotpotQA.<br>New line &amp; more.</p>
<p>Loss is <math alttext="x^2"><mi>x</mi></math> small.</p>
<figure class="ltx_table"><table class="ltx_tabular">
<thead><tr><th>Model</th>
<th></th>
<th>HotpotQA</th><th>NaturalQ</th></tr></thead>
<tbody><tr><td>BigBird</td><td></td><td>75.7</td><td>86.8</td></tr></tbody></table>
<figcaption>Table 2: QA results on HotpotQA and Natural Questions.</figcaption></figure>
<p>Unclosed <b>bold <i>italic</p>
<img src="x.png"><p>After image.</p>
</section>
<section class="ltx_bibliography"><ul><li>[1] SQuAD: 100,000+ questions. 2016.</li></ul></section>
</article>
<footer class="ltx_page_footer">Generated by LaTeXML</footer></body></html>"""


def test_arxiv_html_extraction_keeps_structure_and_drops_noise():
    text = worker.extract_text(LATEXML.encode(), "text/html", "https://arxiv.org/html/2007.14062")
    lines = text.splitlines()
    assert "Model | HotpotQA | NaturalQ" in lines and "BigBird | 75.7 | 86.8" in lines  # table rows
    assert "Table 2: QA results on HotpotQA and Natural Questions." in lines
    assert "We evaluate on WikiHop and HotpotQA." in lines and "New line & more." in lines
    assert "Loss is small." in lines  # math markup dropped
    for noise in ("SQuAD", "Jane Doe", "Uni", "LaTeXML", "navigation", ".x{}"):
        assert noise not in text
    assert "After image." in lines  # void tags and unclosed tags do not derail skipping


def test_missing_arxiv_rendering_falls_back():
    with pytest.raises(ValueError, match="no arXiv HTML"):
        worker.extract_text(b"<html><body>" + b"No HTML for this paper. " * 50 + b"</body></html>", "text/html",
                            "https://arxiv.org/html/1234.56789")
    # other sites are parsed normally
    assert "Hello" in worker.extract_text(b"<p>Hello</p>", "text/html", "https://example.org/a.html")


def test_document_url_order():
    from research_forest.sources.fulltext import document_urls
    p = Paper(id="p", title="t", arxiv_id="2205.14135", pdf_url="https://example.org/paper.pdf")
    assert document_urls(p) == ["https://arxiv.org/html/2205.14135", "https://example.org/paper.pdf",
                                "https://arxiv.org/pdf/2205.14135"]
    assert document_urls(p, prefer_html=False)[0] == "https://example.org/paper.pdf"
    assert document_urls(Paper(id="q", title="t", pdf_url="https://example.org/q.pdf")) == ["https://example.org/q.pdf"]
    assert document_urls(Paper(id="r", title="t")) == []
