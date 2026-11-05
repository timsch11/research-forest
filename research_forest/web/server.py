"""FastAPI app: REST API for the graph plus the single-page explorer UI.

Security model (the server can spend the user's LLM budget and exposes their data):

* every ``/api`` request needs the access token — as an HttpOnly ``SameSite=Strict`` cookie
  (set by opening ``/?token=…`` once) or an ``Authorization: Bearer`` header. This also
  defeats CSRF and DNS-rebinding attacks from websites the user visits;
* state-changing requests with a foreign ``Origin`` are rejected;
* strict security headers (CSP without inline scripts, no framing, no referrer, nosniff);
* bounded inputs and at most ``max_jobs`` concurrent background jobs.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Settings
from ..engine import Forest
from ..store import valid_node

STATIC = Path(__file__).with_name("static")
COOKIE = "rf_token"
log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class Jobs:
    """At most one background job per project and ``limit`` overall, with a rolling progress log."""

    def __init__(self, limit: int = 2) -> None:
        self.logs: dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self.tasks: dict[int, asyncio.Task] = {}
        self.limit = limit

    def log(self, project: int, msg: str) -> None:
        self.logs[project].append({"t": time.time(), "msg": msg})

    def running(self, project: int) -> bool:
        return project in self.tasks and not self.tasks[project].done()

    def check_capacity(self, project: int | None = None) -> None:
        if project is not None and self.running(project):
            raise HTTPException(409, "a job is already running for this project")
        if sum(not t.done() for t in self.tasks.values()) >= self.limit:
            raise HTTPException(429, "too many jobs running; try again later")

    def start(self, project: int, job: Callable[[], Awaitable]) -> None:
        self.check_capacity(project)

        async def wrapper():
            try:
                await job()
                self.log(project, "done")
            except Exception as e:  # surfaced through the progress log
                log.exception("job failed")
                self.log(project, f"error: {type(e).__name__}: {str(e)[:300]}")

        self.tasks[project] = asyncio.create_task(wrapper())


class BuildRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    target_papers: int | None = Field(default=None, ge=5, le=400)


class TextRequest(BaseModel):
    text: str = Field(min_length=2, max_length=2000)


class PreviousTurn(BaseModel):
    question: str = Field(max_length=2000)
    nodes: list[str] = Field(default_factory=list, max_length=40)


class AskRequest(TextRequest):
    previous: PreviousTurn | None = None  # last question + the nodes its answer cited


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin is None:  # non-browser clients; still need the token
        return True
    return urlsplit(origin).netloc == request.headers.get("host")


def create_app(settings: Settings, forest: Forest | None = None, token: str | None = None) -> FastAPI:
    token = token or settings.api_token or secrets.token_urlsafe(24)
    jobs = Jobs(settings.max_jobs)
    forest = forest or Forest(settings)
    if (previous := forest.on_log) is None:
        forest.on_log = jobs.log
    else:  # keep the caller's logger and also feed the UI's progress log
        forest.on_log = lambda project, msg: (previous(project, msg), jobs.log(project, msg))
    store = forest.store

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await forest.aclose()

    app = FastAPI(title="research-forest", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = token

    def authorized(request: Request) -> bool:
        supplied = request.cookies.get(COOKIE) or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        return bool(supplied) and hmac.compare_digest(supplied.encode(), token.encode())

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if not authorized(request):
                response: Response = JSONResponse({"detail": "unauthorized: open the URL printed by "
                                                   "`research-forest serve` (it contains the access token)"},
                                                  status_code=401)
            elif request.method not in ("GET", "HEAD") and not _same_origin(request):
                response = JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
            else:
                response = await call_next(request)
                response.headers["Cache-Control"] = "no-store"
        else:
            response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def project_or_404(pid: int) -> dict:
        if not (p := store.project(pid)):
            raise HTTPException(404, "project not found")
        return p

    @app.get("/")
    async def index(token_param: str | None = Query(default=None, alias="token", max_length=200)):
        if token_param is not None:  # exchange the one-time URL token for a cookie, then drop it from the URL
            if not hmac.compare_digest(token_param.encode(), token.encode()):
                raise HTTPException(401, "invalid token")
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=30 * 24 * 3600)
            return resp
        return FileResponse(STATIC / "index.html")

    @app.get("/api/projects")
    async def projects():
        return store.projects()

    @app.post("/api/projects")
    async def create(req: BuildRequest):
        jobs.check_capacity()
        pid = store.create_project(req.prompt)
        jobs.start(pid, lambda: forest.build(req.prompt, req.target_papers, project=pid))
        return {"id": pid}

    @app.get("/api/projects/{pid}")
    async def project(pid: int, since: float = 0.0):
        p = project_or_404(pid)
        return {**p, "running": jobs.running(pid), "log": [l for l in jobs.logs[pid] if l["t"] > since],
                "snapshots": store.snapshots(pid)}

    @app.get("/api/projects/{pid}/graph")
    async def graph(pid: int):
        project_or_404(pid)
        return store.graph(pid)

    @app.get("/api/projects/{pid}/nodes/{node}")
    async def node(pid: int, node: str):
        if not valid_node(node) or not (info := store.node(pid, node)):
            raise HTTPException(404, "node not found")
        return info

    @app.get("/api/projects/{pid}/search")
    async def search(pid: int, q: str = Query(max_length=200), kind: str | None = Query(default=None, max_length=20)):
        project_or_404(pid)
        return store.search(pid, q, kinds=[kind] if kind else None)

    @app.post("/api/projects/{pid}/refine")
    async def refine(pid: int, req: TextRequest):
        project_or_404(pid)
        jobs.start(pid, lambda: forest.refine(pid, req.text))
        return {"started": True}

    @app.post("/api/projects/{pid}/ask")
    async def ask(pid: int, req: AskRequest):
        project_or_404(pid)
        previous = req.previous.model_dump() if req.previous else None
        if previous:
            previous["nodes"] = [n for n in previous["nodes"] if valid_node(n)]
        return await forest.ask(pid, req.text, previous)

    @app.post("/api/projects/{pid}/undo")
    async def undo(pid: int):
        project_or_404(pid)
        if jobs.running(pid):
            raise HTTPException(409, "cannot undo while a job is running")
        if (label := forest.undo(pid)) is None:
            raise HTTPException(400, "nothing to undo")
        jobs.log(pid, f"undid: {label}")
        return {"undone": label}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
