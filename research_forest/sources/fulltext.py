"""Gap-filling passage retrieval: S2 snippet search first, then the sandboxed BM25 worker."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import Settings
from ..models import Paper
from .semantic_scholar import SemanticScholar

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("fulltext_worker.py")
WORKER_GRACE = 15.0  # seconds on top of the download timeout before the worker is killed

# Per missing field: a keyword query for local BM25 (bag of words) and a natural-language query
# for Semantic Scholar snippet search. Both are extended with LLM-suggested search terms and with
# names already known in the graph (the best vocabulary for finding e.g. dataset mentions).
FIELD_QUERIES = {
    "datasets": "evaluate evaluated evaluation experiments results benchmark benchmarks dataset datasets corpus "
                "trained training pretrained test split table accuracy",
    "limitations": "limitation limitations limited drawback fails failure however cannot restrict assumption "
                   "future work weakness",
    "methods": "propose proposed method approach architecture model framework algorithm baseline compare",
    # second dataset query: pre-training / training corpora live in setup sections, not results
    "datasets_training": "pre-trained pretrained pre-training pretraining corpus corpora training data trained "
                         "tokens crawl wikipedia books web text collected",
}
SNIPPET_QUERIES = {
    "datasets": "we evaluate on datasets and benchmarks; experiments are trained and tested on",
    "datasets_training": "we pre-train on the corpus; the training data consists of",
    "limitations": "limitations of our approach; it fails when; future work",
    "methods": "we propose a method; we compare against baselines",
}
MAX_VOCABULARY = 12
TRAINING_TOP_K = 3  # passages from the training-data query, added to the dataset passages
TABLES, SECTIONS, TABLE_REFS = "tables", "limitation_sections", "table_refs"  # reserved keys in find() results
MAX_TABLE_REFS = 80
MAX_TABLES, TABLE_CHARS, MAX_SECTIONS, SECTION_CHARS = 15, 600, 3, 2500
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy",
                    "http_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
                    "SYSTEMROOT", "WINDIR", "COMSPEC")  # Windows needs SYSTEMROOT for sockets/crypto


def _package_paths() -> list[str]:
    """site-packages dirs that hold the worker's third-party deps (needed under ``python -I``)."""
    paths = []
    for mod in ("httpx", "pypdf", "httpcore", "anyio", "h11", "certifi", "idna", "sniffio"):
        if (spec := importlib.util.find_spec(mod)) and spec.origin:
            parent = str(Path(spec.origin).parent.parent)
            if parent not in paths:
                paths.append(parent)
    return paths


@functools.lru_cache(maxsize=1)
def _bwrap_prefix() -> tuple[str, ...]:
    """bubblewrap command prefix if available and functional, else empty (rlimits only)."""
    if not sys.platform.startswith("linux") or not (bwrap := shutil.which("bwrap")):
        return ()
    binds: list[str] = []
    for p in {"/usr", "/lib", "/lib64", "/bin", "/etc", sys.prefix, sys.base_prefix, *_package_paths()}:
        if os.path.exists(p):
            binds += ["--ro-bind", p, p]
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if (f := os.environ.get(var)) and os.path.exists(f):
            binds += ["--ro-bind", f, f]
    prefix = (bwrap, *binds, "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--unshare-all",
              "--share-net", "--die-with-parent", "--new-session", "--chdir", "/tmp")
    try:
        subprocess.run([*prefix, sys.executable, "-I", "-c", "pass"], check=True, timeout=10,
                       capture_output=True)
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("bubblewrap present but unusable (%s); worker runs with rlimits only", e)
        return ()
    return prefix


async def run_worker(urls: list[str], queries: dict[str, str], settings: Settings, top_k: int = 4,
                     tables: bool = False, sections: bool = False) -> dict:
    """Run the full-text worker in an isolated subprocess and return its JSON response.

    ``tables``: also return compact summaries of main-body tables (HTML only; appendix and
    supplementary tables are skipped). ``sections``: also return Limitations / Conclusion /
    Discussion sections.
    """
    request = {"urls": urls, "queries": queries, "top_k": top_k, "tables": tables, "sections": sections,
               "max_bytes": settings.fulltext_max_mb * 2**20,
               "timeout": settings.fulltext_timeout, "mem_mb": settings.fulltext_mem_mb, "paths": _package_paths(),
               "allow_private": settings.fulltext_allow_private}
    env = {k: v for k in _ENV_PASSTHROUGH if (v := os.environ.get(k))}
    with tempfile.TemporaryDirectory(prefix="rf-fulltext-") as tmp:
        env["HOME"] = env["TMPDIR"] = env["TEMP"] = env["TMP"] = tmp
        cmd = [*_bwrap_prefix(), sys.executable, "-I", str(WORKER)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=tmp, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()),
                                              settings.fulltext_timeout + WORKER_GRACE)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"source": None, "passages": {}, "error": "worker timed out"}
    if proc.returncode != 0:
        return {"source": None, "passages": {}, "error": f"worker exit {proc.returncode}: {err.decode()[-300:]}"}
    try:
        res = json.loads(out or b"{}")
    except ValueError:
        return {"source": None, "passages": {}, "error": "worker returned invalid JSON"}
    # The worker processes untrusted data: validate the shape of what it hands back.
    passages = {f: [{"text": str(h.get("text", ""))[:5000]} for h in hits[:top_k] if isinstance(h, dict)]
                for f, hits in (res.get("passages") or {}).items() if f in queries and isinstance(hits, list)}

    def strings(key: str, count: int, chars: int) -> list[str]:
        items = res.get(key) if isinstance(res.get(key), list) else []
        return [str(x)[:chars] for x in items[:count] if isinstance(x, str) and x.strip()]

    return {"source": res.get("source"), "passages": passages, "error": res.get("error"),
            "tables": strings("tables", MAX_TABLES, TABLE_CHARS) if tables else [],
            "table_refs": _table_refs(res.get("table_refs")) if tables else [],
            "sections": strings("sections", MAX_SECTIONS, SECTION_CHARS) if sections else []}


def _table_refs(raw) -> list[dict]:
    """Validate the worker's cited table labels: {label, title, arxiv, table} with bounded strings."""
    out = []
    for r in (raw if isinstance(raw, list) else [])[:MAX_TABLE_REFS]:
        if isinstance(r, dict) and isinstance(r.get("label"), str) and isinstance(r.get("title"), str):
            arxiv = r.get("arxiv") if isinstance(r.get("arxiv"), str) and len(r.get("arxiv")) <= 20 else None
            out.append({"label": r["label"][:120], "title": r["title"][:300], "arxiv": arxiv,
                        "table": str(r.get("table") or "")[:30]})
    return out


def document_urls(paper: Paper, prefer_html: bool = True) -> list[str]:
    """Full-text sources to try in order. arXiv's HTML rendering (also available for most older
    papers) gives clean text with readable tables, where datasets are typically listed; PDF text
    extraction scrambles tables. The worker falls through to the next URL when one fails."""
    html = f"https://arxiv.org/html/{paper.arxiv_id}" if paper.arxiv_id else None
    arxiv_pdf = f"https://arxiv.org/pdf/{paper.arxiv_id}" if paper.arxiv_id else None
    order = [html, paper.pdf_url, arxiv_pdf] if prefer_html else [paper.pdf_url, arxiv_pdf, html]
    return list(dict.fromkeys(u for u in order if u))


class PassageFinder:
    """Finds text passages that can fill the fields an abstract-only extraction missed."""

    def __init__(self, s2: SemanticScholar, settings: Settings):
        self.s2, self.settings = s2, settings
        self._sem = asyncio.Semaphore(2)

    async def find(self, paper: Paper, fields: list[str], terms: list[str], vocabulary: dict[str, list[str]] | None = None,
                   top_k: int = 4, sections: bool = False) -> dict[str, list[str]]:
        """Passages per field, plus reserved keys when the document was read:

        * ``"tables"``: caption, header row and row labels of each main-body table (when datasets
          are requested; HTML sources only);
        * ``"limitation_sections"``: Limitations / Conclusion / Discussion sections (``sections``).

        ``vocabulary`` maps a field to names already known in the graph (e.g. datasets other
        papers use); they are added to the queries so passages that name them rank first. For
        datasets a second query targets training / pre-training corpora.
        """
        extra = " ".join(terms)
        keys = list(fields) + (["datasets_training"] if "datasets" in fields else [])
        vocab = lambda k: " ".join((vocabulary or {}).get(k.split("_")[0], [])[:MAX_VOCABULARY])  # noqa: E731
        queries = {k: " ".join(x for x in (FIELD_QUERIES.get(k, k), extra, vocab(k)) if x) for k in keys}
        snippet_queries = {k: " ".join(x for x in (SNIPPET_QUERIES.get(k, k), extra, vocab(k)) if x)[:300]
                           for k in keys}
        limits = {k: TRAINING_TOP_K if k == "datasets_training" else top_k for k in keys}

        found: dict[str, list[str]] = {}
        urls = document_urls(paper, self.settings.fulltext_prefer_html) if self.settings.fulltext_enabled else []
        # An arXiv HTML rendering beats S2's snippets (whole document, readable tables), so read it
        # first when available; snippets then only cover what is still missing. Otherwise snippets
        # come first, since they avoid downloading and parsing a PDF.
        html_first = bool(urls) and self.settings.fulltext_prefer_html and "arxiv.org/html/" in urls[0]
        order = ("document", "snippets") if html_first else ("snippets", "document")
        document_read = False
        for source in order:
            remaining = [k for k in keys if k not in found]
            if not remaining and (document_read or not sections):
                break
            if source == "snippets":
                lists = await asyncio.gather(*(self.s2.snippets(snippet_queries[k], paper.id, limits[k])
                                               for k in remaining))
                found.update({k: snippets for k, snippets in zip(remaining, lists) if snippets})
            elif urls:
                async with self._sem:
                    res = await run_worker(urls, {k: queries[k] for k in remaining or keys}, self.settings,
                                           max(limits.values()), tables="datasets" in fields, sections=sections)
                if res.get("error"):
                    log.info("full-text worker for %s: %s", paper.id, res["error"])
                document_read = bool(res.get("source"))
                for k, hits in (res.get("passages") or {}).items():
                    if hits and k not in found:
                        found[k] = [h["text"] for h in hits][:limits.get(k, top_k)]
                if res.get("tables"):
                    found[TABLES] = res["tables"]
                if res.get("table_refs"):
                    found[TABLE_REFS] = res["table_refs"]
                if res.get("sections"):
                    found[SECTIONS] = res["sections"]
        if training := found.pop("datasets_training", None):  # merge into the dataset passages
            found["datasets"] = list(dict.fromkeys(found.get("datasets", []) + training))
        return found
