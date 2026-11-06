from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLM
from ..sources.fulltext import PassageFinder
from ..sources.semantic_scholar import SemanticScholar
from ..store import Store

log = logging.getLogger("research_forest")


@dataclass
class Ctx:
    """Everything a pipeline stage needs to operate on one project."""

    store: Store
    llm: LLM
    s2: SemanticScholar
    finder: PassageFinder | None
    settings: Settings
    project: int
    on_log: Callable[[int, str], None] | None = field(default=None, repr=False)
    on_progress: Callable[[int, dict], None] | None = field(default=None, repr=False)
    _stage: str = field(default="", repr=False)
    _label: str = field(default="", repr=False)
    _done: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)

    def log(self, msg: str) -> None:
        log.info("[project %d] %s", self.project, msg)
        if self.on_log:
            self.on_log(self.project, msg)

    def warn(self, msg: str) -> None:
        """Something the user should know about the result (e.g. lost searches): logged and kept
        on the progress card until the next job."""
        self.log(f"warning: {msg}")
        if self.on_progress:
            self.on_progress(self.project, {"warning": msg})

    def s2_hint(self) -> str:
        return "" if self.settings.s2_api_key else " Setting S2_API_KEY raises Semantic Scholar's rate limit."

    # Progress: the engine opens a stage; the step inside it declares its units of work and ticks.

    def stage(self, key: str, label: str) -> None:
        """Enter a pipeline stage (one of the job's declared stages)."""
        self._stage, self._label, self._done, self._total = key, label, 0, 0
        self.log(label)
        self._emit()

    def work(self, total: int, label: str | None = None) -> None:
        """Declare the units of work of the current step (e.g. papers or prompts)."""
        self._done, self._total = 0, max(0, total)
        if label:
            self._label = label
        self._emit()

    def tick(self, n: int = 1) -> None:
        self._done = min(self._total, self._done + n) if self._total else self._done + n
        self._emit()

    def _emit(self) -> None:
        if self.on_progress:
            self.on_progress(self.project, {"stage": self._stage, "label": self._label,
                                            "done": self._done, "total": self._total})


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
