"""Append-only JSONL log of every LLM call: inputs, outputs and their sizes.

One line per attempt (cache hits and failed attempts included)::

    {"ts": "...", "backend": "openai", "model": "gpt-6-luna", "call": "Extraction", "attempt": 1,
     "cached": false, "seconds": 3.2, "error": null,
     "words": {"system": 85, "user": 612, "output": 140}, "chars": {...},
     "tokens": {"input": 1450, "cached": 900, "output": 410, "reasoning": 220, "cost_usd": 0.0012, "turns": 2},
     "system": "...", "user": "...", "output": "..."}

``tokens`` holds what the backend itself reported: ``input`` includes cached prompt tokens and
everything the CLI adds around our prompt; ``output`` includes reasoning/thinking tokens. All
backends report it when their API does (``cost_usd`` and ``turns`` only Claude Code); words and
characters are always there, as a fallback (tokens ≈ 1.3 × words).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
               started: float | None = None, tokens: dict | None = None) -> None:
        out = _text(output)
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "backend": backend, "model": model, "call": call, "attempt": attempt, "cached": cached,
            "seconds": round(time.monotonic() - started, 3) if started is not None else 0.0,
            "error": None if error is None else (str(error)[:2000] or type(error).__name__),
            "words": {"system": _words(system), "user": _words(user), "output": _words(out)},
            "chars": {"system": len(system), "user": len(user), "output": len(out)},
            "tokens": tokens or None,
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
    """Per call type: calls, cache hits, errors, words in/out, reported tokens and time."""
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in read_entries(path):
        s = stats[e["call"]]
        s["calls"] += 1
        s["cached"] += e["cached"]
        s["errors"] += e["error"] is not None
        s["words_in"] += e["words"]["system"] + e["words"]["user"]
        s["words_out"] += e["words"]["output"]
        s["seconds"] += e["seconds"]
        if not e["cached"] and (t := e.get("tokens")):
            s["metered"] += 1
            for k in ("input", "cached", "output", "reasoning", "cost_usd"):
                s[f"tok_{k}"] += t.get(k) or 0
    if not stats:
        return "no LLM calls logged"
    cols = [("calls", "calls", 6, 0), ("cached", "cached", 7, 0), ("errors", "errors", 7, 0),
            ("words in", "words_in", 10, 0), ("words out", "words_out", 10, 0),
            ("tokens in", "tok_input", 11, 0), ("prompt cache", "tok_cached", 13, 0),
            ("tokens out", "tok_output", 11, 0), ("reasoning", "tok_reasoning", 10, 0), ("seconds", "seconds", 9, 1)]
    width = max(14, *(len(n) + 1 for n in stats))
    header = f"{'call':<{width}}" + "".join(f"{label:>{w}}" for label, _, w, _ in cols)
    total: dict[str, float] = defaultdict(float)

    def row(name: str, s: dict[str, float]) -> str:
        return f"{name:<{width}}" + "".join(f"{s.get(k, 0):>{w}.{d}f}" for _, k, w, d in cols)

    lines = [header, "-" * len(header)]
    for name, s in sorted(stats.items(), key=lambda kv: -(kv[1]["tok_input"] or kv[1]["words_in"])):
        lines.append(row(name, s))
        for k, v in s.items():
            total[k] += v
    lines += ["-" * len(header), row("total", total), ""]
    uncached = total["calls"] - total["cached"]
    if total["metered"]:
        lines.append(f"Tokens as reported by the backend for {total['metered']:.0f} of {uncached:.0f} uncached calls: "
                     "input includes the prompt cache and everything the CLI adds; output includes reasoning.")
        if total["tok_cost_usd"]:
            lines.append(f"Reported cost: ${total['tok_cost_usd']:.4f} (estimated at API list prices; a subscription "
                         "isn't charged this, but it shows the relative usage)")
    else:
        lines.append("The backend reported no token counts: sizes are in words (tokens ≈ 1.3 × words).")
    lines.append("Cached calls cost nothing.")
    return "\n".join(lines)
