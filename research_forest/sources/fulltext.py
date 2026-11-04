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

# Generic BM25 queries per missing field; extended with LLM-suggested search terms.
FIELD_QUERIES = {
    "datasets": "dataset datasets benchmark corpus evaluate evaluation experiments trained train test split",
    "limitations": "limitation limitations limited drawback fails failure however cannot restrict assumption "
                   "future work weakness",
    "methods": "propose proposed method approach architecture model framework algorithm",
}
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


async def run_worker(urls: list[str], queries: dict[str, str], settings: Settings, top_k: int = 4) -> dict:
    """Run the full-text worker in an isolated subprocess and return its JSON response."""
    request = {"urls": urls, "queries": queries, "top_k": top_k, "max_bytes": settings.fulltext_max_mb * 2**20,
               "timeout": settings.fulltext_timeout, "mem_mb": settings.fulltext_mem_mb, "paths": _package_paths()}
    env = {k: v for k in _ENV_PASSTHROUGH if (v := os.environ.get(k))}
    with tempfile.TemporaryDirectory(prefix="rf-fulltext-") as tmp:
        env["HOME"] = env["TMPDIR"] = env["TEMP"] = env["TMP"] = tmp
        cmd = [*_bwrap_prefix(), sys.executable, "-I", str(WORKER)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=tmp, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(json.dumps(request).encode()),
                                              settings.fulltext_timeout + 15)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"source": None, "passages": {}, "error": "worker timed out"}
    if proc.returncode != 0:
        return {"source": None, "passages": {}, "error": f"worker exit {proc.returncode}: {err.decode()[-300:]}"}
    return json.loads(out or b"{}")


class PassageFinder:
    """Finds text passages that can fill the fields an abstract-only extraction missed."""

    def __init__(self, s2: SemanticScholar, settings: Settings):
        self.s2, self.settings = s2, settings
        self._sem = asyncio.Semaphore(2)

    async def find(self, paper: Paper, fields: list[str], terms: list[str], top_k: int = 4) -> dict[str, list[str]]:
        extra = " ".join(terms)
        queries = {f: f"{FIELD_QUERIES.get(f, f)} {extra}".strip() for f in fields}

        found: dict[str, list[str]] = {}
        snippet_lists = await asyncio.gather(*(self.s2.snippets(q, paper.id, top_k) for q in queries.values()))
        for field, snippets in zip(queries, snippet_lists):
            if snippets:
                found[field] = snippets

        remaining = {f: q for f, q in queries.items() if f not in found}
        urls = [u for u in (paper.pdf_url, f"https://arxiv.org/pdf/{paper.arxiv_id}" if paper.arxiv_id else None) if u]
        if remaining and urls and self.settings.fulltext_enabled:
            async with self._sem:
                res = await run_worker(list(dict.fromkeys(urls)), remaining, self.settings, top_k)
            if res.get("error"):
                log.info("full-text worker for %s: %s", paper.id, res["error"])
            for field, hits in (res.get("passages") or {}).items():
                if hits:
                    found[field] = [h["text"] for h in hits]
        return found
