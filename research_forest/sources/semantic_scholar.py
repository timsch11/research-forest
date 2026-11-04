"""Async Semantic Scholar Graph API client with client-side rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Literal

import httpx

from ..models import CitationLink, Paper

log = logging.getLogger(__name__)

API = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,url,externalIds,openAccessPdf,tldr"
EDGE_PAPER_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,url,externalIds"
EDGE_FIELDS = "contexts,intents,isInfluential," + EDGE_PAPER_FIELDS


def to_paper(d: dict) -> Paper | None:
    if not d or not d.get("paperId") or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    pdf = (d.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{arxiv}" if arxiv else None)
    return Paper(
        id=d["paperId"], title=d["title"].strip(), abstract=d.get("abstract"),
        tldr=(d.get("tldr") or {}).get("text"), year=d.get("year"), venue=d.get("venue") or None,
        authors=[a["name"] for a in (d.get("authors") or [])[:8] if a.get("name")],
        citation_count=d.get("citationCount") or 0, url=d.get("url"), arxiv_id=arxiv, pdf_url=pdf)


class SemanticScholar:
    def __init__(self, api_key: str | None = None, rps: float = 1.0, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        headers = {"x-api-key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(base_url=API, headers=headers, timeout=timeout, transport=transport)
        self._interval = 1.0 / rps if rps > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def _request(self, method: str, path: str, **kw) -> dict:
        for attempt in range(6):
            await self._throttle()
            try:
                r = await self._client.request(method, path, **kw)
            except httpx.TransportError as e:
                log.warning("S2 transport error %s (attempt %d)", e, attempt + 1)
            else:
                if r.status_code == 404:
                    return {}
                if r.status_code != 429 and r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                log.info("S2 %s on %s, backing off", r.status_code, path)
            await asyncio.sleep(min(60, 2 ** attempt * 1.5))
        raise RuntimeError(f"Semantic Scholar request failed: {method} {path}")

    # ----------------------------------------------------------- queries ----

    async def search(self, query: str, limit: int = 40, year_min: int | None = None) -> list[Paper]:
        params = {"query": query, "limit": min(limit, 100), "fields": PAPER_FIELDS}
        if year_min:
            params["year"] = f"{year_min}-"
        data = await self._request("GET", "/paper/search", params=params)
        return [p for d in data.get("data", []) if (p := to_paper(d))]

    async def match_title(self, title: str) -> Paper | None:
        data = await self._request("GET", "/paper/search/match", params={"query": title, "fields": PAPER_FIELDS})
        hits = data.get("data") or []
        return to_paper(hits[0]) if hits else None

    async def batch(self, ids: Iterable[str]) -> list[Paper]:
        ids, out = list(ids), []
        for i in range(0, len(ids), 500):
            data = await self._request("POST", "/paper/batch", params={"fields": PAPER_FIELDS},
                                       json={"ids": ids[i:i + 500]})
            out += [p for d in (data or []) if d and (p := to_paper(d))]
        return out

    async def links(self, paper_id: str, direction: Literal["references", "citations"],
                    limit: int = 500) -> list[tuple[Paper, CitationLink]]:
        """References (papers this one cites) or citations (papers citing it), with citation contexts."""
        data = await self._request("GET", f"/paper/{paper_id}/{direction}",
                                   params={"fields": EDGE_FIELDS, "limit": min(limit, 1000)})
        key = "citedPaper" if direction == "references" else "citingPaper"
        out = []
        for row in data.get("data") or []:
            if not (other := to_paper(row.get(key) or {})):
                continue
            citing, cited = (paper_id, other.id) if direction == "references" else (other.id, paper_id)
            out.append((other, CitationLink(
                citing=citing, cited=cited, contexts=[c for c in row.get("contexts") or [] if c][:6],
                intents=row.get("intents") or [], influential=bool(row.get("isInfluential")))))
        return out

    async def snippets(self, query: str, paper_id: str, limit: int = 5) -> list[str]:
        """Full-text snippets from S2's index for one paper (empty if unavailable)."""
        try:
            data = await self._request("GET", "/snippet/search",
                                       params={"query": query, "paperIds": paper_id, "limit": limit})
        except (httpx.HTTPStatusError, RuntimeError) as e:
            log.info("snippet search unavailable for %s: %s", paper_id, e)
            return []
        return [t for row in data.get("data") or [] if (t := (row.get("snippet") or {}).get("text"))]

    async def aclose(self) -> None:
        await self._client.aclose()
