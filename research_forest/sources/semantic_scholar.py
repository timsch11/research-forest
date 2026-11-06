"""Async Semantic Scholar Graph API client with client-side rate limiting."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Iterable
from typing import Literal

import httpx

from ..models import CitationLink, Paper
from ..safety import safe_url, sanitize

log = logging.getLogger(__name__)

API = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = ("paperId,title,abstract,year,venue,authors,citationCount,influentialCitationCount,url,externalIds,"
                "openAccessPdf,tldr")
EDGE_PAPER_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,influentialCitationCount,url,externalIds"
EDGE_FIELDS = "contexts,intents,isInfluential," + EDGE_PAPER_FIELDS
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?")


def to_paper(d: dict) -> Paper | None:
    if not d or not d.get("paperId") or not isinstance(d.get("title"), str) or not d["title"].strip():
        return None
    ext = d.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv and not _ARXIV_ID.fullmatch(arxiv):
        arxiv = None
    pdf = safe_url((d.get("openAccessPdf") or {}).get("url")) or (f"https://arxiv.org/pdf/{arxiv}" if arxiv else None)
    year = d.get("year")
    return Paper(
        id=str(d["paperId"])[:100], title=sanitize(d["title"], 500), abstract=sanitize(d.get("abstract"), 10_000) or None,
        tldr=sanitize((d.get("tldr") or {}).get("text"), 1000) or None,
        year=year if isinstance(year, int) else None, venue=sanitize(d.get("venue"), 200) or None,
        authors=[sanitize(a["name"], 100) for a in (d.get("authors") or [])[:8] if a.get("name")],
        citation_count=d.get("citationCount") or 0, influential_count=d.get("influentialCitationCount") or 0,
        url=safe_url(d.get("url")), arxiv_id=arxiv, pdf_url=pdf)


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

    async def _request(self, method: str, path: str, attempts: int = 6, **kw) -> dict:
        for attempt in range(attempts):
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

    async def search(self, query: str, limit: int = 40, year_min: int | None = None,
                     since: str | None = None) -> list[Paper]:
        """Keyword search; ``since`` (YYYY-MM-DD) keeps only papers published on or after that date."""
        params = {"query": query, "limit": min(limit, 100), "fields": PAPER_FIELDS}
        if year_min:
            params["year"] = f"{year_min}-"
        if since:
            params["publicationDateOrYear"] = f"{since}:"
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

    async def links(self, paper_id: str, direction: Literal["references", "citations"], limit: int = 500,
                    year_min: int | str | None = None,
                    influential_only: bool = False) -> list[tuple[Paper, CitationLink]]:
        """References (papers this one cites) or citations (papers citing it), with citation contexts.

        Citations come newest first, so ``limit`` bounds how far back they reach; ``year_min``
        (a year or a YYYY-MM-DD date) restricts them further. The API can't filter by ``isInfluential``; ``influential_only``
        drops the rest here.
        """
        params: dict = {"fields": EDGE_FIELDS, "limit": min(limit, 1000)}
        if year_min:
            params["publicationDateOrYear"] = f"{year_min}:"
        data = await self._request("GET", f"/paper/{paper_id}/{direction}", params=params)
        key = "citedPaper" if direction == "references" else "citingPaper"
        out = []
        for row in data.get("data") or []:
            if influential_only and not row.get("isInfluential"):
                continue
            if not (other := to_paper(row.get(key) or {})):
                continue
            citing, cited = (paper_id, other.id) if direction == "references" else (other.id, paper_id)
            out.append((other, CitationLink(
                citing=citing, cited=cited,
                contexts=[sanitize(c, 1000) for c in row.get("contexts") or [] if isinstance(c, str) and c][:6],
                intents=row.get("intents") or [], influential=bool(row.get("isInfluential")))))
        return out

    async def snippets(self, query: str, paper_id: str, limit: int = 5) -> list[str]:
        """Full-text snippets from S2's index for one paper (empty if unavailable)."""
        try:
            # optional source: fail fast instead of minutes of back-off on the shared rate limit
            data = await self._request("GET", "/snippet/search", attempts=2,
                                       params={"query": query, "paperIds": paper_id, "limit": limit})
        except (httpx.HTTPStatusError, RuntimeError) as e:
            log.info("snippet search unavailable for %s: %s", paper_id, e)
            return []
        return [sanitize(t, 5000) for row in data.get("data") or []
                if isinstance(t := (row.get("snippet") or {}).get("text"), str) and t]

    async def aclose(self) -> None:
        await self._client.aclose()
