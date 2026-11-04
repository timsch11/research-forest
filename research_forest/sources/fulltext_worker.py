"""Sandboxed full-text worker: download -> extract text -> BM25 passage search.

Runs as an isolated child process (``python -I``) so untrusted downloads and
PDF parsing never execute inside the main application. It imports nothing from
the research_forest package. Protocol: one JSON request on stdin, one JSON
response on stdout.

Request:  {"urls": [...], "queries": {field: query}, "top_k": int,
           "max_bytes": int, "timeout": float, "mem_mb": int, "paths": [...]}
Response: {"source": url | null, "passages": {field: [{"text", "score"}]}, "error": str | null}
"""

import io
import json
import math
import re
import resource
import sys
from collections import Counter
from html.parser import HTMLParser

MAX_PAGES = 80
CHUNK_WORDS, CHUNK_STRIDE = 120, 80
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were we with "
    "which our these those their than then also can not".split())


def limit_resources(mem_mb: int, cpu_s: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))  # no meaningful disk writes
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# ------------------------------------------------------------- download ----

def download(url: str, max_bytes: int, timeout: float) -> tuple[bytes, str]:
    import httpx

    if not url.startswith(("https://", "http://")):
        raise ValueError("unsupported scheme")
    headers = {"User-Agent": "research-forest/0.1 (fulltext passage search)"}
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in ("application/pdf", "text/html", "application/xhtml+xml", "text/plain"):
            raise ValueError(f"unsupported content-type {ctype!r}")
        if int(r.headers.get("content-length") or 0) > max_bytes:
            raise ValueError("document too large")
        buf = bytearray()
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > max_bytes:
                raise ValueError("document too large")
    return bytes(buf), ctype


# ---------------------------------------------------------- text extract ----

class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "math"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def extract_text(data: bytes, ctype: str) -> str:
    if ctype == "application/pdf" or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PAGES])
    text = data.decode("utf-8", errors="replace")
    if ctype == "text/plain":
        return text
    parser = _HTMLText()
    parser.feed(text)
    return " ".join(parser.parts)


# ------------------------------------------------------------------ BM25 ----

def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def chunk(text: str) -> list[str]:
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate PDF line breaks
    words = text.split()
    return [" ".join(words[i:i + CHUNK_WORDS])
            for i in range(0, max(1, len(words) - CHUNK_WORDS + CHUNK_STRIDE), CHUNK_STRIDE)]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, k: int) -> list[tuple[float, int]]:
        q = set(tokenize(query))
        scores = []
        for i, (tf, ln) in enumerate(zip(self.tfs, self.lens)):
            s = 0.0
            for t in q & tf.keys():
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        picked: list[tuple[float, int]] = []
        for s, i in scores:  # drop heavily overlapping neighbouring windows
            if all(abs(i - j) > 1 for _, j in picked):
                picked.append((s, i))
            if len(picked) == k:
                break
        return picked


def search(text: str, queries: dict[str, str], top_k: int) -> dict[str, list[dict]]:
    index = BM25(chunk(text))
    return {field: [{"text": index.docs[i], "score": round(s, 3)} for s, i in index.top(q, top_k)]
            for field, q in queries.items()}


# ------------------------------------------------------------------ main ----

def main() -> None:
    req = json.loads(sys.stdin.read())
    limit_resources(int(req.get("mem_mb", 1024)), int(req.get("timeout", 90)))
    sys.path.extend(p for p in req.get("paths", []) if p not in sys.path)
    errors = []
    for url in req.get("urls", []):
        try:
            data, ctype = download(url, int(req.get("max_bytes", 25 * 2**20)), float(req.get("timeout", 90)))
            text = extract_text(data, ctype)
            if len(text) < 500:
                raise ValueError("no extractable text")
            out = {"source": url, "passages": search(text, req["queries"], int(req.get("top_k", 4))), "error": None}
            break
        except Exception as e:  # noqa: BLE001 - report any failure to the parent
            errors.append(f"{url}: {type(e).__name__}: {e}")
    else:
        out = {"source": None, "passages": {}, "error": "; ".join(errors) or "no urls"}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
