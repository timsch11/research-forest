"""FastAPI app: REST API for the graph plus the single-page explorer UI."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Settings
from ..engine import Forest

STATIC = Path(__file__).with_name("static")
log = logging.getLogger(__name__)


class Jobs:
    """At most one background job per project, with a rolling progress log."""

    def __init__(self) -> None:
        self.logs: dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self.tasks: dict[int, asyncio.Task] = {}

    def log(self, project: int, msg: str) -> None:
        self.logs[project].append({"t": time.time(), "msg": msg})

    def running(self, project: int) -> bool:
        return project in self.tasks and not self.tasks[project].done()

    def start(self, project: int, job: Callable[[], Awaitable]) -> None:
        if self.running(project):
            raise HTTPException(409, "a job is already running for this project")

        async def wrapper():
            try:
                await job()
                self.log(project, "done")
            except Exception as e:  # surfaced through the progress log
                log.exception("job failed")
                self.log(project, f"error: {type(e).__name__}: {e}")

        self.tasks[project] = asyncio.create_task(wrapper())


class BuildRequest(BaseModel):
    prompt: str = Field(min_length=3)
    target_papers: int | None = Field(default=None, ge=5, le=400)


class TextRequest(BaseModel):
    text: str = Field(min_length=2)


def create_app(settings: Settings, forest: Forest | None = None) -> FastAPI:
    jobs = Jobs()
    forest = forest or Forest(settings, on_log=jobs.log)
    forest.on_log = forest.on_log or jobs.log
    store = forest.store

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await forest.aclose()

    app = FastAPI(title="research-forest", lifespan=lifespan)

    def project_or_404(pid: int) -> dict:
        if not (p := store.project(pid)):
            raise HTTPException(404, "project not found")
        return p

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/projects")
    async def projects():
        return store.projects()

    @app.post("/api/projects")
    async def create(req: BuildRequest):
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
        if not (info := store.node(pid, node)):
            raise HTTPException(404, "node not found")
        return info

    @app.get("/api/projects/{pid}/search")
    async def search(pid: int, q: str, kind: str | None = None):
        project_or_404(pid)
        return store.search(pid, q, kinds=[kind] if kind else None)

    @app.post("/api/projects/{pid}/refine")
    async def refine(pid: int, req: TextRequest):
        project_or_404(pid)
        jobs.start(pid, lambda: forest.refine(pid, req.text))
        return {"started": True}

    @app.post("/api/projects/{pid}/ask")
    async def ask(pid: int, req: TextRequest):
        project_or_404(pid)
        return await forest.ask(pid, req.text)

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
