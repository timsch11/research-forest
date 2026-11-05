"""Sandboxed full-text worker: download -> extract text -> BM25 passage search.

Runs as an isolated child process (``python -I``) so untrusted downloads and
PDF parsing never execute inside the main application. It imports nothing from
the research_forest package. Protocol: one JSON request on stdin, one JSON
response on stdout.

Request:  {"urls": [...], "queries": {field: query}, "top_k": int, "max_bytes": int,
           "timeout": float, "mem_mb": int, "paths": [...], "allow_private": bool}
Response: {"source": url | null, "passages": {field: [{"text", "score"}]}, "error": str | null}
"""

import io
import ipaddress
import json
import math
import os
import re
import socket
import sys
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

MAX_PAGES = 80
CHUNK_WORDS, CHUNK_STRIDE = 120, 80
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were we with "
    "which our these those their than then also can not".split())


def limit_resources(mem_mb: int, cpu_s: int) -> None:
    """Best-effort rlimits. POSIX only; some platforms (e.g. macOS for RLIMIT_AS) refuse individual limits."""
    try:
        import resource
    except ImportError:  # Windows: rely on the parent's wall-clock timeout and size caps
        return
    limits = [(resource.RLIMIT_AS, mem_mb * 2**20), (resource.RLIMIT_CPU, cpu_s),
              (resource.RLIMIT_FSIZE, 2**20), (resource.RLIMIT_CORE, 0)]  # FSIZE: no meaningful disk writes
    for res, value in limits:
        try:
            _, hard = resource.getrlimit(res)
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(res, (value, value))
        except (ValueError, OSError):
            pass


# ------------------------------------------------------------- download ----

MAX_REDIRECTS = 5
MAX_TEXT_CHARS = 3_000_000
ALLOWED_TYPES = ("application/pdf", "text/html", "application/xhtml+xml", "text/plain")


def _proxied(scheme: str) -> bool:
    return any(os.environ.get(k) for k in (f"{scheme.upper()}_PROXY", f"{scheme}_proxy", "ALL_PROXY", "all_proxy"))


def check_url(url: str, allow_private: bool = False) -> None:
    """SSRF guard: http(s) on default ports only, and the host must resolve to public addresses.

    When the host cannot be resolved locally but an egress proxy is configured, resolution is
    the proxy's job and the request is allowed (the proxy enforces its own policy).
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme {scheme!r}")
    host = parts.hostname
    if not host or parts.username or parts.password:
        raise ValueError("invalid host")
    if allow_private:  # local testing only
        return
    if parts.port not in (None, 80, 443):
        raise ValueError(f"port {parts.port} not allowed")
    try:
        addrs = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addrs = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in socket.getaddrinfo(host, None)]
        except (socket.gaierror, UnicodeError):
            if _proxied(scheme):
                return
            raise ValueError(f"cannot resolve {host!r}")
    for addr in addrs:
        addr = getattr(addr, "ipv4_mapped", None) or addr
        if not addr.is_global or addr.is_multicast:
            raise ValueError(f"non-public address {addr} for {host!r}")


def download(url: str, max_bytes: int, timeout: float, allow_private: bool = False,
             client=None) -> tuple[bytes, str]:
    import httpx

    headers = {"User-Agent": "research-forest/0.1 (fulltext passage search)"}
    own = client is None
    client = client or httpx.Client(follow_redirects=False, timeout=timeout, headers=headers)
    try:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(url, allow_private)  # re-checked on every redirect hop
            with client.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in ALLOWED_TYPES:
                    raise ValueError(f"unsupported content-type {ctype!r}")
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    raise ValueError("document too large")
                buf = bytearray()
                for chunk in r.iter_bytes():  # decoded bytes: also bounds gzip bombs
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise ValueError("document too large")
                return bytes(buf), ctype
        raise ValueError("too many redirects")
    finally:
        if own:
            client.close()


# ---------------------------------------------------------- text extract ----

_CELLS = re.compile(r"(?:\s*\|\s*)+")  # runs of (empty) table cells -> one separator
_SPACES = re.compile(r"\s+")


class _HTMLText(HTMLParser):
    """Readable text from HTML, tuned for arXiv's LaTeXML renderings.

    Drops page chrome and parts that only add noise to passage search (bibliography, author and
    affiliation blocks, table of contents, math markup), turns block elements into line breaks
    and keeps table cells apart with " | " so a table row stays one readable line.
    """

    SKIP_TAGS = {"head", "script", "style", "nav", "header", "footer", "noscript", "svg", "math", "button", "form"}
    SKIP_CLASSES = {"ltx_bibliography", "ltx_authors", "ltx_TOC", "ltx_page_footer", "ltx_page_header",
                    "ltx_dates", "ltx_role_affiliation", "ltx_role_email", "ltx_note_mark"}
    BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table",
                  "figcaption", "caption", "figure", "blockquote", "br", "dd", "dt", "pre"}
    CELL_TAGS = {"td", "th"}
    BREAK = "\x00"  # block boundary marker; never occurs in decoded page text we keep
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
                 "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[tuple[str, bool]] = []  # (tag, starts a skipped subtree)
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append(self.BREAK)
        elif tag in self.CELL_TAGS:
            self.parts.append(" | ")
        if tag in self.VOID_TAGS:
            return
        classes = set((dict(attrs).get("class") or "").split())
        skipping = tag in self.SKIP_TAGS or bool(classes & self.SKIP_CLASSES)
        self._stack.append((tag, skipping))
        self._skip += skipping

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append(self.BREAK)
        if tag in self.VOID_TAGS or not any(t == tag for t, _ in self._stack):
            return
        while self._stack:  # pop to the matching tag (tolerates unclosed children)
            t, skipping = self._stack.pop()
            self._skip -= skipping
            if t == tag:
                break

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(_SPACES.sub(" ", data))  # source newlines are just spaces

    def text(self) -> str:
        lines = "".join(self.parts).split(self.BREAK)
        cleaned = (_CELLS.sub(" | ", " ".join(line.split())).strip(" |") for line in lines)
        return "\n".join(line for line in cleaned if line)


def is_arxiv_html(url: str) -> bool:
    return "arxiv.org/html/" in url


def extract_text(data: bytes, ctype: str, url: str = "") -> str:
    if ctype == "application/pdf" or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PAGES])
    text = data.decode("utf-8", errors="replace")
    if ctype == "text/plain":
        return text
    if is_arxiv_html(url) and "ltx_document" not in text:
        raise ValueError("no arXiv HTML rendering for this paper")
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    return parser.text()


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
            data, ctype = download(url, int(req.get("max_bytes", 25 * 2**20)), float(req.get("timeout", 90)),
                                   bool(req.get("allow_private", False)))
            text = extract_text(data, ctype, url)[:MAX_TEXT_CHARS]
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
