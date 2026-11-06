"""Backends that shell out to the ``claude`` (Claude Code) and ``codex`` CLIs.

Both run in a throwaway working directory. Claude Code runs with all tools, MCP
servers and slash commands disabled. Codex runs in its read-only sandbox (no
writes, no network); it can still *read* files, which is why its output is only
ever used as schema-validated, length-capped data.

Cost: both CLIs are agents with a large fixed prompt, and by default they also load the
user's own setup (Claude Code: CLAUDE.md, memory, hooks, plugins; Codex: config.toml with
its MCP servers, model and effort) into *every* call. Unless ``user_config`` is set, that is
skipped, a small model is used unless one is configured, and reasoning effort is set low.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from .base import Backend, Reply, TransientLLMError, usage


async def _run(args: list[str], stdin: str, cwd: str, timeout: float) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin.encode()), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise TransientLLMError(f"{args[0]} timed out after {timeout}s") from None
    if proc.returncode != 0:
        raise TransientLLMError(f"{args[0]} exited {proc.returncode}: {err.decode(errors='replace')[-500:]}")
    return out.decode()


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"'{binary}' CLI not found on PATH")
    return path


class ClaudeCodeBackend(Backend):
    name = "claude-code"
    DEFAULT_MODEL = "haiku"

    def __init__(self, model: str | None = None, timeout: float = 300.0, binary: str = "claude", *,
                 effort: str = "medium", user_config: bool = False):
        self.model, self.timeout, self._bin = model or self.DEFAULT_MODEL, timeout, _require(binary)
        self.effort, self.user_config = effort, user_config

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> Reply:
        # No tools, no MCP servers, no skills/slash commands: injected text has nothing to act with.
        args = [self._bin, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                "--disable-slash-commands", "--no-session-persistence", "--model", self.model,
                "--json-schema", json.dumps(schema, separators=(",", ":")), "--system-prompt", system]
        if not self.user_config:  # no user/project settings: CLAUDE.md, memory, hooks, plugins
            args += ["--setting-sources", ""]
        if self.effort:
            args += ["--effort", self.effort]
        with tempfile.TemporaryDirectory(prefix="rf-claude-") as cwd:
            out = await _run(args, user, cwd, self.timeout)
        envelope = json.loads(out)
        if envelope.get("is_error"):
            raise TransientLLMError(f"claude error: {str(envelope.get('result'))[:300]}")
        u = envelope.get("usage") or {}
        tokens = usage(
            input=sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            if u else None,
            cached=u.get("cache_read_input_tokens"), output=u.get("output_tokens"),
            reasoning=(u.get("output_tokens_details") or {}).get("thinking_tokens"),
            cost_usd=envelope.get("total_cost_usd"), turns=envelope.get("num_turns"))
        return Reply(envelope.get("structured_output") or envelope.get("result", ""), tokens or None)


def _codex_usage(jsonl: str) -> dict | None:
    """Sum the ``usage`` of all ``turn.completed`` events in ``codex exec --json`` output."""
    total: dict[str, int] = {}
    for line in jsonl.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            for k, v in event["usage"].items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
    return usage(input=total.get("input_tokens"), cached=total.get("cached_input_tokens"),
                 output=total.get("output_tokens"), reasoning=total.get("reasoning_output_tokens")) or None


class CodexBackend(Backend):
    name = "codex"
    DEFAULT_MODEL = "gpt-6-luna"

    def __init__(self, model: str | None = None, timeout: float = 300.0, binary: str = "codex", *,
                 effort: str = "medium", user_config: bool = False):
        self.model, self.timeout, self._bin = model or self.DEFAULT_MODEL, timeout, _require(binary)
        self.effort, self.user_config = effort, user_config

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> Reply:
        with tempfile.TemporaryDirectory(prefix="rf-codex-") as cwd:
            schema_path, out_path = Path(cwd, "schema.json"), Path(cwd, "out.json")
            instructions = Path(cwd, "instructions.md")
            schema_path.write_text(json.dumps(schema, separators=(",", ":")))
            instructions.write_text(system, encoding="utf-8")
            # Our system prompt replaces Codex's long agent instructions (about 22k characters).
            args = [self._bin, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                    "--ephemeral", "--json", "--model", self.model,
                    "-c", f"model_instructions_file={json.dumps(str(instructions))}",
                    "--output-schema", str(schema_path), "--output-last-message", str(out_path)]
            if not self.user_config:  # skip ~/.codex/config.toml (MCP servers, model, effort, profiles)
                args.append("--ignore-user-config")
            if self.effort:
                args += ["-c", f"model_reasoning_effort={json.dumps(self.effort)}"]
            events = await _run(args + ["-"], user, cwd, self.timeout)
            if not out_path.exists():
                raise TransientLLMError("codex produced no final message")
            return Reply(out_path.read_text(), _codex_usage(events))
