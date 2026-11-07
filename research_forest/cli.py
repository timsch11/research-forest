"""Command-line interface: ``research-forest {build,refine,ask,search,list,undo,serve}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import TARGET_RANGE, ConfigError, Settings, untrusted_settings
from .engine import Forest


def _settings(args: argparse.Namespace) -> Settings:
    if (cached := getattr(args, "_settings", None)) is not None:
        return cached
    flags = {"backend": args.backend, "model": args.model, "openai_base_url": args.base_url, "db_path": args.db,
             "fulltext_enabled": False if args.no_fulltext else None,
             "fulltext_allow_private": True if args.allow_private else None}
    try:
        settings = Settings.load(args.config, **flags)
        risks = untrusted_settings(settings, args.config, {k for k, v in flags.items() if v is not None})
    except ConfigError as e:
        raise SystemExit(f"research-forest: {e}") from None
    if risks:
        _confirm(risks)
    args._settings = settings
    return settings


def _confirm(risks: list[str]) -> None:
    """Explicit consent for risky settings from the config file (a CLI flag counts as consent)."""
    for risk in risks:
        print(f"research-forest: warning: {risk}", file=sys.stderr)
    if not sys.stdin.isatty():
        raise SystemExit("research-forest: refusing to continue without confirmation; pass the flags above "
                         "on the command line if this is intended")
    if input("Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("research-forest: stopped")


def _papers(value: str) -> int:
    n = int(value)
    if not TARGET_RANGE[0] <= n <= TARGET_RANGE[1]:
        raise argparse.ArgumentTypeError(f"must be between {TARGET_RANGE[0]} and {TARGET_RANGE[1]}")
    return n


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
    p.add_argument("--config", default=None,
                   help="TOML config file (default: ./research-forest.toml if present)")
    p.add_argument("--db", default=None, help="SQLite database file (default: research.forest.db)")
    p.add_argument("--backend", choices=["openai", "codex", "claude-code"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    p.add_argument("--no-fulltext", action="store_true", help="disable full-text gap filling")
    p.add_argument("--allow-private", action="store_true",
                   help="let full-text downloads reach private/local addresses (turns off the SSRF guard)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a new landscape from a prompt")
    b.add_argument("prompt")
    b.add_argument("-n", "--papers", type=_papers, default=None,
                   help=f"target number of papers ({TARGET_RANGE[0]}-{TARGET_RANGE[1]})")
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
    nw = sub.add_parser("update", help="check a landscape for work published since the last check")
    nw.add_argument("project", type=int)
    u = sub.add_parser("undo", help="undo the last refinement")
    u.add_argument("project", type=int)
    d = sub.add_parser("delete", help="delete a landscape and all its data")
    d.add_argument("project", type=int)
    d.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    ev = sub.add_parser("evaluate", help="score extraction quality on the gold set (eval/gold.json)")
    ev.add_argument("--gold", default="eval/gold.json")
    ev.add_argument("--limit", type=int, default=None, help="only the first N gold papers")
    ev.add_argument("--out", default=None, help="write the full JSON report here")
    ev.add_argument("--compare", default=None, help="a previous JSON report to show deltas against")
    ev.add_argument("--sources", action="store_true",
                    help="compare full-text sources (arXiv HTML vs PDF) by passage recall, without an LLM")
    lg = sub.add_parser("llm-log", help="summarise the LLM call log (calls, words in/out, time per call type)")
    lg.add_argument("file", nargs="?", default=None, help="log file (default: [logging] llm_log_file)")
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

    if args.cmd == "llm-log":
        from .llm.calllog import summarize
        path = Path(args.file or _settings(args).llm_log_file)
        if not path.is_file():
            raise SystemExit(f"research-forest: no LLM call log at {path} (is [logging] llm_calls enabled?)")
        print(summarize(path))
        return
    if args.cmd == "evaluate" and args.sources:
        from .evaluation import evaluate_sources, format_sources_report
        report = asyncio.run(evaluate_sources(_settings(args), Path(args.gold), args.limit,
                                              on_log=lambda m: print(m, file=sys.stderr)))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(format_sources_report(report))
        return
    if args.cmd == "evaluate":
        from .evaluation import evaluate, format_report, load_report
        report = asyncio.run(evaluate(_settings(args), Path(args.gold), args.limit,
                                      on_log=lambda m: print(m, file=sys.stderr)))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(format_report(report, load_report(args.compare) if args.compare else None))
        return

    if args.cmd == "delete":
        from .store import Store
        store = Store(_settings(args).db_path)
        if not (proj := store.project(args.project)):
            raise SystemExit(f"research-forest: no landscape {args.project}")
        name = proj["title"] or proj["prompt"]
        if not args.yes and input(f"Delete landscape {args.project} ({name})? [y/N] ").strip().lower() not in ("y", "yes"):
            print("cancelled")
            return
        store.delete_project(args.project)
        print(f"deleted landscape {args.project} ({name})")
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
            case "update":
                since = forest.last_checked(args.project)
                new = await forest.update(args.project)
                titles = {r.paper.id: r.paper.title for r in forest.store.papers(args.project, ids=new)}
                print(f"{len(new)} new papers since {since}" + "".join(f"\n  {titles[i]}" for i in new))
            case "undo":
                label = forest.undo(args.project)
                print(f"undid: {label}" if label else "nothing to undo")

    asyncio.run(_with_forest(args, run))


if __name__ == "__main__":
    main()
