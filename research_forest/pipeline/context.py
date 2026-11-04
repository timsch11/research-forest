from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

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

    def log(self, msg: str) -> None:
        log.info("[project %d] %s", self.project, msg)
        if self.on_log:
            self.on_log(self.project, msg)


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
