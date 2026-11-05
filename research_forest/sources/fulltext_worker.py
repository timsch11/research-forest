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
_NUMERIC = re.compile(r"[\d.,%±+\-–—()x×~*†‡\s]*")  # table cells that are just numbers/markers
_APPENDIX_HEADING = re.compile(r"^\s*(?:[A-Z]\.?\s+)?(appendix|appendices|supplementary|supplemental)\b", re.I)
_LIMITATION_HEADING = re.compile(
    r"\b(limitations?|conclusions?|concluding remarks|discussion|future work|future directions|"
    r"broader impacts?|societal impacts?)\b", re.I)
MAX_TABLES, MAX_SECTIONS, SECTION_CHARS, MAX_TABLE_REFS = 15, 3, 2000, 80
# citation markers inside table cells: "[84]", "[12, 13]", "(Wang et al., 2020)"
_CITE_MARK = re.compile(r"\s*\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\s*\(?[A-Z][\w-]+(?: and [A-Z][\w-]+| et al\.)?,? \(?\d{4}[a-z]?\)\)?")
_ARXIV_ID = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})", re.I)


class Document:
    """Text of a paper plus the structure gap filling uses: main-body table summaries and
    sections (heading, text)."""

    def __init__(self, text: str, tables: list[str] | None = None, sections: list[tuple[str, str]] | None = None,
                 table_refs: list[dict] | None = None):
        self.text, self.tables, self.sections = text, tables or [], sections or []
        self.table_refs = table_refs or []  # cited labels in main-body tables: {label, title, arxiv, table}

    def limitation_sections(self) -> list[str]:
        """Text of Limitations / Conclusion / Discussion / Future-work sections, limitations first."""
        hits = [(h, t) for h, t in self.sections if _LIMITATION_HEADING.search(h) and t.strip()]
        hits.sort(key=lambda ht: "limitation" not in ht[0].lower())
        return [f"{h}: {t}"[:SECTION_CHARS] for h, t in hits[:MAX_SECTIONS]]


def _clean(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def compact_table(caption: str, rows: list[list[str]]) -> str | None:
    """'Caption — columns: A | B | C — rows: x, y, z': where datasets and benchmarks are named."""
    rows = [[_clean(_CITE_MARK.sub("", c)) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    text_cells = lambda row: [c for c in row if c and not _NUMERIC.fullmatch(c)]  # noqa: E731
    header_i = next((i for i, row in enumerate(rows) if len(text_cells(row)) >= 2), None)
    parts = [_clean(caption)[:160]] if _clean(caption) else []
    if header_i is not None:
        parts.append("columns: " + " | ".join(text_cells(rows[header_i]))[:220])
        labels = list(dict.fromkeys(row[0] for row in rows[header_i + 1:] if row and row[0]
                                    and not _NUMERIC.fullmatch(row[0])))
        if labels:
            parts.append("rows: " + ", ".join(labels[:12])[:220])
    return " — ".join(parts) if len(parts) > 1 or (parts and header_i is None) else None


class _HTMLText(HTMLParser):
    """Readable text from HTML, tuned for arXiv's LaTeXML renderings.

    Drops page chrome and parts that only add noise to passage search (bibliography, author and
    affiliation blocks, table of contents, math markup), turns block elements into line breaks
    and keeps table cells apart with " | " so a table row stays one readable line. Also records
    section headings and, for tables in the main body (not appendices/supplementary), their
    caption, header row and row labels.
    """

    SKIP_TAGS = {"head", "script", "style", "nav", "header", "footer", "noscript", "svg", "math", "button", "form"}
    SKIP_CLASSES = {"ltx_bibliography", "ltx_authors", "ltx_TOC", "ltx_page_footer", "ltx_page_header",
                    "ltx_dates", "ltx_role_affiliation", "ltx_role_email", "ltx_note_mark"}
    BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table",
                  "figcaption", "caption", "figure", "blockquote", "br", "dd", "dt", "pre"}
    CELL_TAGS = {"td", "th"}
    HEADING_TAGS = {"h2", "h3", "h4", "h5"}
    BREAK = "\x00"  # block boundary marker; never occurs in decoded page text we keep
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
                 "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[tuple[str, bool, bool, bool, str | None]] = []  # (tag, skips, appendix, table, role)
        self._skip = 0
        self._appendix_after = False  # an "Appendix"/"Supplementary" heading was seen
        self._heading: list[str] | None = None
        self._heading_at = 0
        self._section_starts: list[tuple[str, int, int]] = []  # (heading, heading start, body start) in parts
        self._table: dict | None = None
        self._cell: dict | None = None  # {"text": [...], "cites": [bib ids]}
        self._table_cites: list[tuple[str, str, list[str]]] = []  # (table name, label, bib ids)
        self._bib: dict[str, list[str]] = {}  # bibliography: id -> text blocks (authors, title, venue)
        self._bib_id: str | None = None
        self._bib_block: list[str] | None = None
        self._caption = False
        self.tables: list[str] = []

    def _in_appendix(self) -> bool:
        return self._appendix_after or any(appendix for _, _, appendix, _, _ in self._stack)

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append(self.BREAK)
        elif tag in self.CELL_TAGS:
            self.parts.append(" | ")
        if tag in self.VOID_TAGS:
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        skipping = tag in self.SKIP_TAGS or bool(classes & self.SKIP_CLASSES)
        appendix = "ltx_appendix" in classes
        table = tag == "figure" and "ltx_table" in classes and self._table is None and not self._skip
        role = None
        if tag == "li" and "ltx_bibitem" in classes and attributes.get("id"):
            role, self._bib_id = "bibitem", attributes["id"]
            self._bib[self._bib_id] = []
        elif self._bib_id and "ltx_bibblock" in classes:
            role, self._bib_block = "bibblock", []
        self._stack.append((tag, skipping, appendix, table, role))
        self._skip += skipping
        if tag == "a" and self._cell is not None and (attributes.get("href") or "").startswith("#bib."):
            self._cell["cites"].append(attributes["href"][1:])
        if table:
            self._table = {"rows": [], "caption": [], "appendix": self._in_appendix()}
        elif self._table is not None:
            if tag == "tr":
                self._table["rows"].append([])
            elif tag in self.CELL_TAGS and self._table["rows"]:
                self._cell = {"text": [], "cites": []}
                self._table["rows"][-1].append(self._cell)
            elif tag == "figcaption":
                self._caption = True
        if tag in self.HEADING_TAGS and not self._skip:
            self._heading, self._heading_at = [], len(self.parts)

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append(self.BREAK)
        if tag in self.CELL_TAGS:
            self._cell = None
        elif tag == "figcaption":
            self._caption = False
        if tag in self.HEADING_TAGS and self._heading is not None:
            heading = _clean("".join(self._heading))
            self._heading = None
            if heading:
                self._section_starts.append((heading, self._heading_at, len(self.parts)))
                if _APPENDIX_HEADING.match(heading):
                    self._appendix_after = True
        if tag in self.VOID_TAGS or not any(t == tag for t, *_ in self._stack):
            return
        while self._stack:  # pop to the matching tag (tolerates unclosed children)
            t, skipping, _, table, role = self._stack.pop()
            self._skip -= skipping
            if table:
                self._finish_table()
            if role == "bibblock" and self._bib_block is not None and self._bib_id:
                self._bib[self._bib_id].append(_clean("".join(self._bib_block)))
                self._bib_block = None
            elif role == "bibitem":
                self._bib_id = None
            if t == tag:
                break

    def _finish_table(self) -> None:
        tbl, self._table, self._cell, self._caption = self._table, None, None, False
        if tbl and not tbl["appendix"] and len(self.tables) < MAX_TABLES:
            caption = "".join(tbl["caption"])
            rows = [["".join(cell["text"]) for cell in row] for row in tbl["rows"]]
            if summary := compact_table(caption, rows):
                self.tables.append(summary)
            name = m.group(0) if (m := re.match(r"\s*Table\s*[\w.]+", caption)) else "Table"
            for row in tbl["rows"]:
                for cell in row:
                    label = _clean(_CITE_MARK.sub("", "".join(cell["text"])))
                    if cell["cites"] and label and not _NUMERIC.fullmatch(label):
                        self._table_cites.append((_clean(name), label[:120], cell["cites"]))

    def handle_data(self, data):
        if self._bib_block is not None:  # bibliography text is captured even though it is skipped
            self._bib_block.append(data)
        if self._skip:
            return
        text = _SPACES.sub(" ", data)  # source newlines are just spaces
        self.parts.append(text)
        if self._heading is not None:
            self._heading.append(text)
        if self._table is not None:
            if self._caption:
                self._table["caption"].append(text)
            elif self._cell is not None:
                self._cell["text"].append(text)

    @classmethod
    def _join(cls, parts: list[str]) -> str:
        lines = "".join(parts).split(cls.BREAK)
        cleaned = (_CELLS.sub(" | ", " ".join(line.split())).strip(" |") for line in lines)
        return "\n".join(line for line in cleaned if line)

    def text(self) -> str:
        return self._join(self.parts)

    def table_refs(self) -> list[dict]:
        """Cited labels in main-body tables, resolved to their bibliography entries: usually the
        baselines a paper compares against ("Linformer [84]" -> "Linformer: Self-attention ...")."""
        refs, seen = [], set()
        for table, label, cites in self._table_cites:
            for bib_id in cites:
                blocks = [b for b in self._bib.get(bib_id, []) if b]
                if not blocks or (label, bib_id) in seen:
                    continue
                seen.add((label, bib_id))
                arxiv = m.group(1) if (m := _ARXIV_ID.search(" ".join(blocks))) else None
                title = blocks[1] if len(blocks) > 1 else blocks[0]
                refs.append({"label": label, "title": title[:300], "arxiv": arxiv, "table": table})
        return refs[:MAX_TABLE_REFS]

    def sections(self) -> list[tuple[str, str]]:
        """(heading, body) pairs; a body ends where the next heading begins."""
        ends = [start for _, start, _ in self._section_starts[1:]] + [len(self.parts)]
        return [(h, self._join(self.parts[body:end])) for (h, _, body), end in zip(self._section_starts, ends)]


_PDF_SECTION = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)?\s*(limitations?|conclusions?(?: and future work)?|concluding remarks|"
    r"discussion(?: and conclusions?)?|future work|broader impacts?|societal impacts?)\s*$", re.I)
_PDF_NEXT_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+[A-Z][A-Za-z ,&-]{2,60}$|^\s*(references|"
                               r"acknowledge?ments?|bibliography)\s*$", re.I)


def pdf_sections(text: str, max_words: int = 400) -> list[tuple[str, str]]:
    """Best-effort Limitations/Conclusion/Discussion sections from PDF text (headings on own lines)."""
    lines, found = text.splitlines(), []
    for i, line in enumerate(lines):
        if not _PDF_SECTION.match(line):
            continue
        body, words = [], 0
        for nxt in lines[i + 1:]:
            if _PDF_NEXT_HEADING.match(nxt) or words >= max_words:
                break
            body.append(nxt)
            words += len(nxt.split())
        if body:
            found.append((_clean(line), _clean(" ".join(body))))
    return found


def is_arxiv_html(url: str) -> bool:
    return "arxiv.org/html/" in url


def parse_document(data: bytes, ctype: str, url: str = "") -> Document:
    if ctype == "application/pdf" or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PAGES])
        return Document(text, sections=pdf_sections(text))
    text = data.decode("utf-8", errors="replace")
    if ctype == "text/plain":
        return Document(text, sections=pdf_sections(text))
    if is_arxiv_html(url) and "ltx_document" not in text:
        raise ValueError("no arXiv HTML rendering for this paper")
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    return Document(parser.text(), parser.tables, parser.sections(), parser.table_refs())


def extract_text(data: bytes, ctype: str, url: str = "") -> str:
    return parse_document(data, ctype, url).text


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
            doc = parse_document(data, ctype, url)
            text = doc.text[:MAX_TEXT_CHARS]
            if len(text) < 500:
                raise ValueError("no extractable text")
            out = {"source": url, "passages": search(text, req["queries"], int(req.get("top_k", 4))), "error": None,
                   "tables": doc.tables if req.get("tables") else [],
                   "table_refs": doc.table_refs if req.get("tables") else [],
                   "sections": doc.limitation_sections() if req.get("sections") else []}
            break
        except Exception as e:  # noqa: BLE001 - report any failure to the parent
            errors.append(f"{url}: {type(e).__name__}: {e}")
    else:
        out = {"source": None, "passages": {}, "error": "; ".join(errors) or "no urls"}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
