"""Backends that shell out to the ``claude`` (Claude Code) and ``codex`` CLIs.

Both run with tools disabled / a read-only sandbox in a throwaway working
directory, so the agents cannot touch the user's files.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from .base import Backend, TransientLLMError


async def _run(args: list[str], stdin: str, cwd: str, timeout: float) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin.encode()), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TransientLLMError(f"{args[0]} timed out after {timeout}s")
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

    def __init__(self, model: str | None = None, timeout: float = 300.0, binary: str = "claude"):
        self.model, self.timeout, self._bin = model, timeout, _require(binary)

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> str | dict:
        args = [self._bin, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
                "--json-schema", json.dumps(schema), "--system-prompt", system]
        if self.model:
            args += ["--model", self.model]
        with tempfile.TemporaryDirectory(prefix="rf-claude-") as cwd:
            out = await _run(args, user, cwd, self.timeout)
        envelope = json.loads(out)
        if envelope.get("is_error"):
            raise TransientLLMError(f"claude error: {str(envelope.get('result'))[:300]}")
        return envelope.get("structured_output") or envelope.get("result", "")


class CodexBackend(Backend):
    name = "codex"

    def __init__(self, model: str | None = None, timeout: float = 300.0, binary: str = "codex"):
        self.model, self.timeout, self._bin = model, timeout, _require(binary)

    async def generate(self, system: str, user: str, schema: dict, schema_name: str) -> str:
        with tempfile.TemporaryDirectory(prefix="rf-codex-") as cwd:
            schema_path, out_path = Path(cwd, "schema.json"), Path(cwd, "out.json")
            schema_path.write_text(json.dumps(schema))
            args = [self._bin, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                    "--output-schema", str(schema_path), "--output-last-message", str(out_path)]
            if self.model:
                args += ["--model", self.model]
            await _run(args + ["-"], f"{system}\n\n---\n\n{user}", cwd, self.timeout)
            if not out_path.exists():
                raise TransientLLMError("codex produced no final message")
            return out_path.read_text()
