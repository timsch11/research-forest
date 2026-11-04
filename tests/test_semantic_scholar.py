import asyncio
import json
import time

import httpx
import pytest

from research_forest.sources.semantic_scholar import SemanticScholar, to_paper


def client(handler, rps=0):
    return SemanticScholar(rps=rps, transport=httpx.MockTransport(handler))


def test_to_paper_sanitises_untrusted_fields():
    p = to_paper({"paperId": "x", "title": "  Title‮​  ", "abstract": "a\x00b", "year": "2020",
                  "url": "javascript:alert(1)", "openAccessPdf": {"url": "data:text/html,x"},
                  "externalIds": {"ArXiv": "../../etc/passwd"}, "authors": [{"name": "A⁦B"}, {}]})
    assert p.title == "Title" and p.abstract == "ab" and p.year is None
    assert p.url is None and p.pdf_url is None and p.arxiv_id is None
    assert p.authors == ["AB"]


@pytest.mark.parametrize("arxiv", ["2205.14135", "2205.14135v2", "hep-th/9901001", "math.GT/0309136"])
def test_to_paper_accepts_real_arxiv_ids(arxiv):
    p = to_paper({"paperId": "x", "title": "T", "externalIds": {"ArXiv": arxiv}})
    assert p.arxiv_id == arxiv and p.pdf_url == f"https://arxiv.org/pdf/{arxiv}"


@pytest.mark.parametrize("d", [{}, {"paperId": "x"}, {"paperId": "x", "title": "  "}, {"title": "t"},
                               {"paperId": "x", "title": 5}])
def test_to_paper_rejects_incomplete(d):
    assert to_paper(d) is None


async def test_retries_on_429_and_5xx(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _s: real_sleep(0))
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429 if len(calls) == 1 else 503)
        return httpx.Response(200, json={"data": [{"paperId": "a", "title": "A"}]})

    s2 = client(handler)
    papers = await s2.search("q")
    assert [p.id for p in papers] == ["a"] and len(calls) == 3
    await s2.aclose()


async def test_404_is_empty_and_4xx_raises():
    s2 = client(lambda r: httpx.Response(404))
    assert await s2.match_title("nothing") is None
    assert await s2.links("x", "references") == []
    await s2.aclose()
    s2 = client(lambda r: httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        await s2.search("q")
    await s2.aclose()


async def test_batch_is_chunked():
    sizes = []

    def handler(request):
        ids = json.loads(request.content)["ids"]
        sizes.append(len(ids))
        return httpx.Response(200, json=[{"paperId": i, "title": i} for i in ids] + [None])

    s2 = client(handler)
    papers = await s2.batch(f"id{i}" for i in range(1200))
    assert sizes == [500, 500, 200] and len(papers) == 1200
    await s2.aclose()


async def test_links_direction_mapping():
    def handler(request):
        key = "citedPaper" if request.url.path.endswith("references") else "citingPaper"
        return httpx.Response(200, json={"data": [
            {key: {"paperId": "o", "title": "Other"}, "contexts": ["c1", None, ""], "intents": ["methodology"],
             "isInfluential": True},
            {key: {"paperId": None}}]})

    s2 = client(handler)
    (other, link), = await s2.links("me", "references")
    assert (link.citing, link.cited, link.contexts, link.influential) == ("me", "o", ["c1"], True)
    (_, link), = await s2.links("me", "citations")
    assert (link.citing, link.cited) == ("o", "me")
    await s2.aclose()


async def test_snippets_degrade_gracefully():
    s2 = client(lambda r: httpx.Response(403))
    assert await s2.snippets("q", "x") == []
    await s2.aclose()
    s2 = client(lambda r: httpx.Response(200, json={"data": [{"snippet": {"text": "t‮"}}, {"snippet": {}},
                                                             {"snippet": {"text": 3}}]}))
    assert await s2.snippets("q", "x") == ["t"]
    await s2.aclose()


async def test_throttle_spaces_requests():
    s2 = client(lambda r: httpx.Response(200, json={"data": []}), rps=20)
    start = time.monotonic()
    for _ in range(4):
        await s2.search("q")
    assert time.monotonic() - start >= 0.14  # 3 gaps of 50ms
    await s2.aclose()


async def test_api_key_header():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"data": []})

    s2 = SemanticScholar(api_key="secret", rps=0, transport=httpx.MockTransport(handler))
    await s2.search("q")
    assert seen["x-api-key"] == "secret"
    await s2.aclose()
