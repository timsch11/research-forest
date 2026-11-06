"""Backend for any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, vLLM, Ollama, ...)."""

from __future__ import annotations

import json
import logging

import httpx

from .base import Backend, Reply, TransientLLMError, usage

# Structured-output modes, from most to least constrained. Servers that reject a
# mode (HTTP 400/422) are transparently downgraded to the next one.
_MODES = ("strict", "schema", "object", "plain")
log = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-6-luna"


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, base_url: str, api_key: str | None, model: str | None, timeout: float = 300.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.model = model or DEFAULT_MODEL
        host = httpx.URL(base_url).host
        if api_key and httpx.URL(base_url).scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
            log.warning("sending the API key over plain HTTP to %s", host)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
                                         transport=transport)
        self._mode = 0

    def _body(self, system: str, user: str, schema: dict, name: str) -> dict:
        mode = _MODES[self._mode]
        if mode in ("object", "plain"):
            system += ("\n\nRespond with a single JSON object matching this JSON schema:\n"
                       + json.dumps(schema, separators=(",", ":")))
        body: dict = {"model": self.model,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if mode in ("strict", "schema"):
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": name, "schema": schema, "strict": mode == "strict"}}
        elif mode == "object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> Reply:
        while True:
            try:
                r = await self._client.post("/chat/completions", json=self._body(system, user, schema, schema_name))
            except httpx.TransportError as e:
                raise TransientLLMError(str(e)) from e
            if r.status_code in (400, 422) and self._mode < len(_MODES) - 1:
                self._mode += 1
                continue
            if r.status_code == 429 or r.status_code >= 500:
                raise TransientLLMError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            body = r.json()
            u = body.get("usage") or {}
            return Reply(body["choices"][0]["message"]["content"] or "", usage(
                input=u.get("prompt_tokens"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                output=u.get("completion_tokens"),
                reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or None)

    async def aclose(self) -> None:
        await self._client.aclose()
