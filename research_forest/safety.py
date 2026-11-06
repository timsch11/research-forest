"""Defences for untrusted content: prompt-injection containment, text/URL sanitisation.

Everything that originates outside the user (paper metadata, abstracts, citation
contexts, full-text passages, web pages) *and* everything an LLM derived from it
(entity names, limitations, summaries) is treated as untrusted. It is:

1. sanitised: control, zero-width and bidi characters are stripped (they hide text
   from humans while staying visible to the model) and anything resembling our own
   delimiter tags is removed, so data can never "close" its block;
2. fenced: interpolated only inside ``<untrusted_data>`` blocks, which the system
   prompt declares to be inert data;
3. bounded: LLM outputs are length/count-capped and validated (see ``models.py``),
   and ids are only accepted from the enumerated refs we offered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlsplit

DATA_TAG = "untrusted_data"

# Our delimiter tags (any spacing/case/separator variant), so data cannot forge or close them.
_TAG_RE = re.compile(
    r"<\s*/?\s*(?:untrusted[\s_-]*data|user[\s_-]*(?:command|request|(?:previous[\s_-]*)?question))\b[^>]*>", re.IGNORECASE)
# C0/C1 controls (except \t \n \r), zero-width chars, bidi overrides/isolates, BOM, tag characters.
_INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤⁦-⁩﻿"
    "\U000e0000-\U000e007f]")
_SUSPICIOUS_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:instructions?|prompts?|rules|guidelines)\b"
    r"|\byou\s+are\s+now\b|\bsystem\s+prompt\b|\bnew\s+instructions?\b|\bdeveloper\s+mode\b"
    r"|<\s*/?\s*(?:system|assistant|instructions?)\s*>|\b(?:respond|answer|reply)\s+only\s+with\b",
    re.IGNORECASE)


def sanitize(text: str | None, limit: int | None = None, keep_newlines: bool = False) -> str:
    """Strip invisible characters and delimiter look-alikes, collapse whitespace, optionally clip."""
    text = _INVISIBLE_RE.sub("", text or "")
    while True:  # repeat: removing one tag may assemble another ("<untr<untrusted_data>usted_data>")
        cleaned = _TAG_RE.sub(" ", text)
        if cleaned == text:
            break
        text = cleaned
    if keep_newlines:
        text = "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()
    else:
        text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def data_block(source: str, lines: Iterable[str]) -> str:
    """Fence already-sanitised untrusted lines. ``source`` is a fixed label chosen by our code."""
    body = "\n".join(lines) or "(none)"
    return f'<{DATA_TAG} source="{source}">\n{body}\n</{DATA_TAG}>'


def trusted_block(tag: str, text: str) -> str:
    """Wrap text typed by the user (still sanitised so it cannot forge data blocks)."""
    return f"<{tag}>\n{sanitize(text)}\n</{tag}>"


def looks_like_injection(text: str | None) -> bool:
    """Heuristic flag for instruction-like text inside data. Used for logging only, never for
    deletion: papers *about* prompt injection legitimately contain such phrases."""
    return bool(text) and bool(_SUSPICIOUS_RE.search(text))


def safe_url(url: str | None) -> str | None:
    """Keep only absolute http(s) URLs (blocks javascript:, data:, file: ...)."""
    if not url:
        return None
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc or _INVISIBLE_RE.search(url):
        return None
    return url
