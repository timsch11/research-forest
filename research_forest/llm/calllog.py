"""Append-only JSONL log of every LLM call: inputs, outputs and their sizes.

One line per attempt (cache hits and failed attempts included)::

    {"ts": "...", "backend": "openai", "model": "gpt-6-luna", "call": "Extraction", "attempt": 1,
     "cached": false, "seconds": 3.2, "error": null,
     "words": {"system": 85, "user": 612, "output": 140}, "chars": {...},
     "system": "...", "user": "...", "output": "..."}

Sizes are reported in words and characters: exact token counts depend on the backend's
tokenizer and are not reported by every backend (as a rule of thumb, tokens ≈ 1.3 × words).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _words(text: str) -> int:
    return len(text.split())


class CallLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, *, backend: str, model: str | None, call: str, attempt: int, system: str, user: str,
               output: Any = None, error: BaseException | str | None = None, cached: bool = False,
               started: float | None = None) -> None:
        out = _text(output)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "backend": backend, "model": model, "call": call, "attempt": attempt, "cached": cached,
            "seconds": round(time.monotonic() - started, 3) if started is not None else 0.0,
            "error": None if error is None else (str(error)[:2000] or type(error).__name__),
            "words": {"system": _words(system), "user": _words(user), "output": _words(out)},
            "chars": {"system": len(system), "user": len(user), "output": len(out)},
            "system": system, "user": user, "output": out,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_entries(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize(path: str | Path) -> str:
    """Per call type: number of calls, cache hits, errors, words in/out and time."""
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in read_entries(path):
        s = stats[e["call"]]
        s["calls"] += 1
        s["cached"] += e["cached"]
        s["errors"] += e["error"] is not None
        s["words_in"] += e["words"]["system"] + e["words"]["user"]
        s["words_out"] += e["words"]["output"]
        s["seconds"] += e["seconds"]
    if not stats:
        return "no LLM calls logged"
    header = f"{'call':<14}{'calls':>7}{'cached':>8}{'errors':>8}{'words in':>11}{'words out':>11}{'avg in':>8}{'seconds':>9}"
    lines = [header, "-" * len(header)]
    total = defaultdict(float)
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["words_in"]):
        lines.append(f"{name:<14}{s['calls']:>7.0f}{s['cached']:>8.0f}{s['errors']:>8.0f}{s['words_in']:>11.0f}"
                     f"{s['words_out']:>11.0f}{s['words_in'] / s['calls']:>8.0f}{s['seconds']:>9.1f}")
        for k, v in s.items():
            total[k] += v
    lines += ["-" * len(header),
              f"{'total':<14}{total['calls']:>7.0f}{total['cached']:>8.0f}{total['errors']:>8.0f}"
              f"{total['words_in']:>11.0f}{total['words_out']:>11.0f}{total['words_in'] / total['calls']:>8.0f}"
              f"{total['seconds']:>9.1f}",
              "", "Sizes in words (tokens ≈ 1.3 × words). Cached calls cost nothing."]
    return "\n".join(lines)
