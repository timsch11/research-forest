"""Command-line interface: ``research-forest {build,refine,ask,search,list,undo,serve}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import Settings
from .engine import Forest


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(backend=args.backend, model=args.model, openai_base_url=args.base_url,
                             db_path=args.db, fulltext_enabled=False if args.no_fulltext else None)


async def _with_forest(args: argparse.Namespace, fn):
    forest = Forest(_settings(args), on_log=lambda _p, msg: print(f"  · {msg}", file=sys.stderr))
    try:
        return await fn(forest)
    finally:
        await forest.aclose()


def _summary(forest: Forest, project: int) -> str:
    g = forest.store.graph(project)
    counts: dict[str, int] = {}
    for n in g["nodes"]:
        counts[n["kind"]] = counts.get(n["kind"], 0) + 1
    builds = sum(e["kind"] == "builds_on" for e in g["edges"])
    questions = [n["label"] for n in g["nodes"] if n["kind"] == "question"]
    lines = [f"project {project}: {counts} · {builds} builds-on edges"]
    lines += [f"  ? {q}" for q in questions]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="research-forest", description=__doc__)
    p.add_argument("--db", default=None, help="SQLite database file (default: research.forest.db)")
    p.add_argument("--backend", choices=["openai", "codex", "claude-code"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    p.add_argument("--no-fulltext", action="store_true", help="disable full-text gap filling")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a new landscape from a prompt")
    b.add_argument("prompt")
    b.add_argument("-n", "--papers", type=int, default=None, help="target number of papers")
    r = sub.add_parser("refine", help="apply a natural-language refinement")
    r.add_argument("project", type=int)
    r.add_argument("command")
    a = sub.add_parser("ask", help="ask a question about a landscape")
    a.add_argument("project", type=int)
    a.add_argument("question")
    s = sub.add_parser("search", help="keyword search over a landscape")
    s.add_argument("project", type=int)
    s.add_argument("query")
    sub.add_parser("list", help="list landscapes")
    u = sub.add_parser("undo", help="undo the last refinement")
    u.add_argument("project", type=int)
    sv = sub.add_parser("serve", help="run the web explorer")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn

        from .web import create_app
        app = create_app(_settings(args))
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"warning: listening on {args.host} exposes the explorer to your network; "
                  "traffic is unencrypted, keep the token secret.", file=sys.stderr)
        shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        print(f"research-forest explorer: http://{shown}:{args.port}/?token={app.state.token}", file=sys.stderr)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return

    async def run(forest: Forest):
        match args.cmd:
            case "build":
                project = await forest.build(args.prompt, args.papers)
                print(_summary(forest, project))
            case "refine":
                plan = await forest.refine(args.project, args.command)
                print(plan.explanation)
                print(_summary(forest, args.project))
            case "ask":
                res = await forest.ask(args.project, args.question)
                print(res["answer"])
            case "search":
                for hit in forest.store.search(args.project, args.query):
                    print(f"{hit['kind']:<10} {hit['node']:<14} {hit['title']}")
            case "list":
                for proj in forest.store.projects():
                    print(f"{proj['id']:>3}  {proj['status']:<9} {proj['title'] or proj['prompt']}")
            case "undo":
                label = forest.undo(args.project)
                print(f"undid: {label}" if label else "nothing to undo")

    asyncio.run(_with_forest(args, run))


if __name__ == "__main__":
    main()
