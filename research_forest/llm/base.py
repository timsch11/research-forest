"""Backend-agnostic structured-output LLM client with caching, retries and validation."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_OUTPUT_CHARS = 1_000_000


class TransientLLMError(RuntimeError):
    """Retryable failure (rate limit, 5xx, CLI crash)."""


class Backend(ABC):
    """Produces a JSON document conforming to ``schema`` for a system/user prompt."""

    name: str = "backend"
    model: str | None = None

    @abstractmethod
    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> str | dict:
        ...

    async def aclose(self) -> None:  # pragma: no cover - optional hook
        pass


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema with inlined refs, all properties required and no extra keys.

    This is the common denominator accepted by OpenAI strict structured outputs,
    Codex ``--output-schema`` and Claude Code ``--json-schema``.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return walk({**target, **extra})
        # Drop the "default"/"title" *keywords*, but never entries of "properties": those are field
        # names, and a field may well be called "title".
        out = {k: ({name: walk(sub) for name, sub in v.items()} if k == "properties" else walk(v))
               for k, v in node.items() if k not in ("default", "title")}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(raw)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> Any:
    text = _FENCE.sub("", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


class JsonCache:
    """Tiny thread-safe SQLite key/value cache for LLM responses."""

    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM llm_cache WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)", (key, json.dumps(value)))


def _brief(e: Exception) -> str:
    """One-line description of a validation/JSON error."""
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, err['loc'])) or 'output'}: {err['msg']}" for err in e.errors()[:5])
    return str(e).splitlines()[0][:200]


class LLM:
    """High-level client: ``await llm.json(Schema, system, user) -> Schema``."""

    def __init__(self, backend: Backend, *, concurrency: int = 4, retries: int = 2,
                 cache: JsonCache | None = None):
        self.backend = backend
        self.retries = retries
        self.cache = cache
        self._sem = asyncio.Semaphore(concurrency)

    def _key(self, schema: dict, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (self.backend.name, self.backend.model or "", json.dumps(schema, sort_keys=True), system, user):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()

    async def json(self, model: type[T], system: str, user: str, *, use_cache: bool = True) -> T:
        schema = strict_schema(model)
        key = self._key(schema, system, user)
        if use_cache and self.cache and (hit := self.cache.get(key)) is not None:
            try:
                return model.model_validate(hit)
            except ValidationError:
                pass

        prompt, last_err = user, None
        for attempt in range(self.retries + 1):
            try:
                async with self._sem:
                    raw = await self.backend.generate(system, prompt, schema, model.__name__)
                if isinstance(raw, str) and len(raw) > MAX_OUTPUT_CHARS:
                    raise ValueError(f"output too large ({len(raw)} chars)")
                data = parse_json(raw) if isinstance(raw, str) else raw
                result = model.model_validate(data)
            except TransientLLMError as e:
                last_err = e
                await asyncio.sleep(2 ** attempt * 2)
                continue
            except (ValidationError, ValueError) as e:  # JSONDecodeError is a ValueError
                last_err = e
                log.warning("%s returned invalid %s (attempt %d/%d): %s — retrying", self.backend.name,
                            model.__name__, attempt + 1, self.retries + 1, _brief(e))
                prompt = (f"{user}\n\nYour previous answer was invalid ({str(e)[:500]}). "
                          "Reply with JSON that matches the schema exactly.")
                continue
            if self.cache:
                self.cache.put(key, result.model_dump(mode="json"))
            return result
        raise RuntimeError(f"{self.backend.name} did not return a valid {model.__name__} after "
                           f"{self.retries + 1} attempts ({_brief(last_err) if last_err else 'no output'})")

    async def aclose(self) -> None:
        await self.backend.aclose()
