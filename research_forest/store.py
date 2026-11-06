"""SQLite persistence: papers, entities, typed edges, FTS5 search index and snapshots.

Node ids are ``p:<s2 paper id>`` for papers and ``e:<int>`` for entities
(methods, datasets, limitation themes, open questions).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .models import CitationLink, EntityKind, Extraction, Paper, Scope
from .safety import sanitize

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY, prompt TEXT NOT NULL, title TEXT, scope TEXT,
    status TEXT NOT NULL DEFAULT 'new', created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS papers (
    project INTEGER NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',      -- candidate | included | excluded
    relevance INTEGER,
    extraction TEXT, refs_fetched INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project, id));
CREATE INDEX IF NOT EXISTS papers_status ON papers(project, status);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project INTEGER NOT NULL, kind TEXT NOT NULL,
    name TEXT NOT NULL, key TEXT NOT NULL, description TEXT, data TEXT,
    UNIQUE (project, kind, key));
CREATE TABLE IF NOT EXISTS edges (
    project INTEGER NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL, data TEXT,
    PRIMARY KEY (project, src, dst, kind));
CREATE INDEX IF NOT EXISTS edges_dst ON edges(project, dst);
CREATE TABLE IF NOT EXISTS snapshots (
    project INTEGER NOT NULL, version INTEGER NOT NULL, label TEXT, created REAL NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (project, version));
CREATE TABLE IF NOT EXISTS meta (
    project INTEGER NOT NULL, key TEXT NOT NULL, value TEXT, PRIMARY KEY (project, key));
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    node UNINDEXED, project UNINDEXED, kind UNINDEXED, title, body, tokenize = 'porter unicode61');
"""

PAPER_EDGE_KINDS = ("cites", "builds_on")
ENTITY_EDGE_KINDS = ("proposes", "uses", "evaluates_on", "has_limitation", "addresses", "motivates", "about")


def pid(paper_id: str) -> str:
    return f"p:{paper_id}"


def eid(entity_id: int) -> str:
    return f"e:{entity_id}"


NODE_RE = re.compile(r"p:[A-Za-z0-9_.:-]{1,100}|e:\d{1,18}")
SNIPPET_START, SNIPPET_END = "\x02", "\x03"  # highlight markers that cannot occur in sanitised text


def valid_node(node: str) -> bool:
    return bool(NODE_RE.fullmatch(node))


def entity_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower()) or name.lower()


_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it of on or that the this to was were "
    "what when where which who why with any all most more papers paper".split())


def fts_query(text: str, mode: str = "AND") -> str:
    """Safe FTS5 query: quoted prefix tokens joined by AND/OR (stopwords dropped for OR queries)."""
    tokens = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
    if mode == "OR":
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return f" {mode} ".join(f'"{t}"*' for t in dict.fromkeys(tokens))


@dataclass
class PaperRecord:
    paper: Paper
    status: str
    relevance: int | None
    extraction: Extraction | None
    refs_fetched: bool

    @property
    def node(self) -> str:
        return pid(self.paper.id)


class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchall()

    # -------------------------------------------------------------- projects

    def create_project(self, prompt: str) -> int:
        now = time.time()
        with self.tx() as c:
            return c.execute("INSERT INTO projects (prompt, created, updated) VALUES (?, ?, ?)",
                             (prompt, now, now)).lastrowid

    def project(self, project: int) -> dict | None:
        rows = self.q("SELECT * FROM projects WHERE id = ?", (project,))
        if not rows:
            return None
        d = dict(rows[0])
        d["scope"] = json.loads(d["scope"]) if d["scope"] else None
        return d

    def projects(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT id, prompt, title, status, created, updated FROM projects "
                                        "ORDER BY updated DESC")]

    def delete_project(self, project: int) -> bool:
        """Delete a landscape and everything it owns (papers, entities, edges, snapshots, meta, index)."""
        with self.tx() as c:
            for table in ("papers", "entities", "edges", "snapshots", "meta", "fts"):
                c.execute(f"DELETE FROM {table} WHERE project = ?", (project,))
            return c.execute("DELETE FROM projects WHERE id = ?", (project,)).rowcount > 0

    def set_scope(self, project: int, scope: Scope) -> None:
        with self.tx() as c:
            c.execute("UPDATE projects SET scope = ?, title = ?, updated = ? WHERE id = ?",
                      (scope.model_dump_json(), scope.title, time.time(), project))

    def scope(self, project: int) -> Scope:
        return Scope.model_validate((self.project(project) or {})["scope"])

    def set_status(self, project: int, status: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE projects SET status = ?, updated = ? WHERE id = ?", (status, time.time(), project))

    def meta(self, project: int, key: str, default: Any = None) -> Any:
        rows = self.q("SELECT value FROM meta WHERE project = ? AND key = ?", (project, key))
        return json.loads(rows[0]["value"]) if rows else default

    def set_meta(self, project: int, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?, ?)", (project, key, json.dumps(value)))

    # ---------------------------------------------------------------- papers

    def add_papers(self, project: int, papers: Iterable[Paper]) -> list[str]:
        """Insert unseen papers as candidates (metadata of known ones is refreshed); returns new ids."""
        new = []
        with self.tx() as c:
            for p in papers:
                cur = c.execute("INSERT OR IGNORE INTO papers (project, id, data) VALUES (?, ?, ?)",
                                (project, p.id, p.model_dump_json()))
                if cur.rowcount:
                    new.append(p.id)
                elif p.abstract:
                    c.execute("UPDATE papers SET data = ? WHERE project = ? AND id = ?",
                              (p.model_dump_json(), project, p.id))
        return new

    def papers(self, project: int, status: str | None = None, ids: Iterable[str] | None = None) -> list[PaperRecord]:
        sql, args = "SELECT * FROM papers WHERE project = ?", [project]
        if status:
            sql += " AND status = ?"
            args.append(status)
        if ids is not None:
            if not (ids := list(ids)):
                return []
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args += ids
        return [PaperRecord(Paper.model_validate_json(r["data"]), r["status"], r["relevance"],
                            Extraction.model_validate_json(r["extraction"]) if r["extraction"] else None,
                            bool(r["refs_fetched"])) for r in self.q(sql, args)]

    def update_papers(self, project: int, ids: Iterable[str], **fields: Any) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.tx() as c:
            c.executemany(f"UPDATE papers SET {cols} WHERE project = ? AND id = ?",
                          [(*fields.values(), project, i) for i in ids])

    def set_relevance(self, project: int, scores: dict[str, int]) -> None:
        with self.tx() as c:
            c.executemany("UPDATE papers SET relevance = ? WHERE project = ? AND id = ?",
                          [(s, project, i) for i, s in scores.items()])

    def set_extraction(self, project: int, paper_id: str, ex: Extraction) -> None:
        self.update_papers(project, [paper_id], extraction=ex.model_dump_json())

    def exclude_papers(self, project: int, ids: Iterable[str]) -> None:
        """Remove papers from the graph (kept as 'excluded' so they are not re-added)."""
        ids = list(ids)
        self.update_papers(project, ids, status="excluded")
        with self.tx() as c:
            c.executemany("DELETE FROM edges WHERE project = ? AND src = ? AND kind NOT IN ('cites')",
                          [(project, pid(i)) for i in ids])
            c.executemany("DELETE FROM edges WHERE project = ? AND dst = ? AND kind = 'builds_on'",
                          [(project, pid(i)) for i in ids])
        self.prune_orphans(project)

    # ----------------------------------------------------------------- edges

    def add_citations(self, project: int, links: Iterable[CitationLink]) -> None:
        with self.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, 'cites', ?)",
                [(project, pid(l.citing), pid(l.cited),
                  json.dumps({"contexts": l.contexts, "intents": l.intents, "influential": l.influential}))
                 for l in links])

    def add_edges(self, project: int, edges: Iterable[tuple[str, str, str, dict | None]]) -> None:
        with self.tx() as c:
            c.executemany("INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?)",
                          [(project, s, d, k, json.dumps(data) if data else None) for s, d, k, data in edges])

    def delete_edges(self, project: int, kinds: Iterable[str], src: Iterable[str] | None = None) -> None:
        kinds = list(kinds)
        sql = f"DELETE FROM edges WHERE project = ? AND kind IN ({','.join('?' * len(kinds))})"
        with self.tx() as c:
            if src is None:
                c.execute(sql, (project, *kinds))
            else:
                c.executemany(sql + " AND src = ?", [(project, *kinds, s) for s in src])

    def edges(self, project: int, kinds: Iterable[str] | None = None) -> list[dict]:
        sql, args = "SELECT src, dst, kind, data FROM edges WHERE project = ?", [project]
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        return [{"src": r["src"], "dst": r["dst"], "kind": r["kind"],
                 "data": json.loads(r["data"]) if r["data"] else {}} for r in self.q(sql, args)]

    # -------------------------------------------------------------- entities

    def upsert_entity(self, project: int, kind: EntityKind, name: str, description: str | None = None,
                      data: dict | None = None) -> str:
        key = entity_key(name)
        with self.tx() as c:
            c.execute("INSERT INTO entities (project, kind, name, key, description, data) VALUES (?, ?, ?, ?, ?, ?) "
                      "ON CONFLICT (project, kind, key) DO UPDATE SET "
                      "description = COALESCE(entities.description, excluded.description), "
                      "data = COALESCE(excluded.data, entities.data)",
                      (project, kind, name.strip(), key, description, json.dumps(data) if data else None))
            row = c.execute("SELECT id FROM entities WHERE project = ? AND kind = ? AND key = ?",
                            (project, kind, key)).fetchone()
        return eid(row["id"])

    def entities(self, project: int, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM entities WHERE project = ?", [project]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        return [{**dict(r), "node": eid(r["id"]), "data": json.loads(r["data"]) if r["data"] else {}}
                for r in self.q(sql, args)]

    def delete_entities(self, project: int, kinds: Iterable[str]) -> None:
        kinds = list(kinds)
        marks = ",".join("?" * len(kinds))
        with self.tx() as c:
            nodes = [eid(r["id"]) for r in c.execute(
                f"SELECT id FROM entities WHERE project = ? AND kind IN ({marks})", (project, *kinds))]
            c.executemany("DELETE FROM edges WHERE project = ? AND (src = ? OR dst = ?)",
                          [(project, n, n) for n in nodes])
            c.execute(f"DELETE FROM entities WHERE project = ? AND kind IN ({marks})", (project, *kinds))

    def merge_entities(self, project: int, keep: str, drop: Iterable[str], name: str | None = None) -> None:
        """Re-point all edges of ``drop`` entities to ``keep`` and delete them."""
        drop = [d for d in drop if d != keep]
        with self.tx() as c:
            for d in drop:
                c.execute("INSERT OR IGNORE INTO edges SELECT project, ?, dst, kind, data FROM edges "
                          "WHERE project = ? AND src = ?", (keep, project, d))
                c.execute("INSERT OR IGNORE INTO edges SELECT project, src, ?, kind, data FROM edges "
                          "WHERE project = ? AND dst = ?", (keep, project, d))
                c.execute("DELETE FROM edges WHERE project = ? AND (src = ? OR dst = ?)", (project, d, d))
                c.execute("DELETE FROM entities WHERE project = ? AND id = ?", (project, int(d[2:])))
            if name:
                c.execute("UPDATE OR IGNORE entities SET name = ?, key = ? WHERE project = ? AND id = ?",
                          (name, entity_key(name), project, int(keep[2:])))

    def prune_orphans(self, project: int) -> None:
        """Drop method/dataset entities no longer linked to any included paper."""
        with self.tx() as c:
            c.execute("""
                DELETE FROM entities WHERE project = ?1 AND kind IN ('method', 'dataset') AND NOT EXISTS (
                    SELECT 1 FROM edges e JOIN papers p ON p.project = ?1 AND e.src = 'p:' || p.id
                    WHERE e.project = ?1 AND e.dst = 'e:' || entities.id AND p.status = 'included')""", (project,))
            c.execute("""DELETE FROM edges WHERE project = ?1 AND dst LIKE 'e:%'
                         AND CAST(substr(dst, 3) AS INTEGER) NOT IN (SELECT id FROM entities WHERE project = ?1)""",
                      (project,))

    # ---------------------------------------------------------------- search

    def reindex(self, project: int) -> None:
        rows = []
        for r in self.papers(project, "included"):
            p, ex = r.paper, r.extraction
            body = [p.abstract or "", p.tldr or "", " ".join(p.authors), p.venue or ""]
            if ex:
                body += [ex.contribution, *(l.text for l in ex.limitations), *ex.future_work]
            rows.append((r.node, project, "paper", sanitize(p.title), sanitize("\n".join(body))))
        rows += [(e["node"], project, e["kind"], sanitize(e["name"]), sanitize(e["description"]))
                 for e in self.entities(project)]
        with self.tx() as c:
            c.execute("DELETE FROM fts WHERE project = ?", (project,))
            c.executemany("INSERT INTO fts (node, project, kind, title, body) VALUES (?, ?, ?, ?, ?)", rows)

    def search(self, project: int, text: str, limit: int = 30, mode: str = "AND",
               kinds: Iterable[str] | None = None) -> list[dict]:
        if not (query := fts_query(text, mode)):
            return []
        sql = ("SELECT node, kind, title, snippet(fts, 4, ?, ?, '…', 14) AS snippet, bm25(fts, 5.0, 1.0) AS score "
               "FROM fts WHERE fts MATCH ? AND project = ?")
        args: list[Any] = [SNIPPET_START, SNIPPET_END, query, project]
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        return [dict(r) for r in self.q(sql + " ORDER BY score LIMIT ?", (*args, limit))]

    # ----------------------------------------------------------------- graph

    def graph(self, project: int) -> dict:
        """Included papers, the entities attached to them, and all edges among those nodes."""
        papers = self.papers(project, "included")
        nodes = {r.node: {"id": r.node, "kind": "paper", "label": r.paper.title, "year": r.paper.year,
                          "citations": r.paper.citation_count, "relevance": r.relevance, "venue": r.paper.venue}
                 for r in papers}
        entity_nodes = {e["node"]: {"id": e["node"], "kind": e["kind"], "label": e["name"]}
                        for e in self.entities(project)}
        edges, degree = [], defaultdict(int)
        for e in self.edges(project):
            src_ok = e["src"] in nodes or e["src"] in entity_nodes
            dst_ok = e["dst"] in nodes or e["dst"] in entity_nodes
            if not (src_ok and dst_ok):
                continue
            edges.append({"source": e["src"], "target": e["dst"], "kind": e["kind"], **_edge_view(e)})
            degree[e["src"]] += 1
            degree[e["dst"]] += 1
        for n, node in entity_nodes.items():
            if degree[n]:
                nodes[n] = {**node, "degree": degree[n]}
        return {"nodes": list(nodes.values()), "edges": edges}

    def node(self, project: int, node: str) -> dict | None:
        if not valid_node(node):
            return None
        if node.startswith("p:"):
            recs = self.papers(project, ids=[node[2:]])
            if not recs:
                return None
            r = recs[0]
            info = {"id": node, "kind": "paper", **r.paper.model_dump(), "status": r.status,
                    "relevance": r.relevance,
                    "extraction": r.extraction.model_dump() if r.extraction else None}
        else:
            rows = self.q("SELECT * FROM entities WHERE project = ? AND id = ?", (project, int(node[2:])))
            if not rows:
                return None
            r = rows[0]
            info = {"id": node, "kind": r["kind"], "name": r["name"], "description": r["description"],
                    "data": json.loads(r["data"]) if r["data"] else {}}
        labels = self._labels(project)
        neighbors = []
        for e in self.q("SELECT * FROM edges WHERE project = ? AND (src = ? OR dst = ?)", (project, node, node)):
            outgoing = e["src"] == node
            other = e["dst"] if outgoing else e["src"]
            if other in labels:
                neighbors.append({"node": other, "label": labels[other], "kind": e["kind"],
                                  "direction": "out" if outgoing else "in",
                                  "data": json.loads(e["data"]) if e["data"] else {}})
        info["neighbors"] = neighbors
        return info

    def _labels(self, project: int) -> dict[str, str]:
        labels = {pid(r["id"]): json.loads(r["data"])["title"]
                  for r in self.q("SELECT id, data FROM papers WHERE project = ? AND status = 'included'", (project,))}
        labels.update({eid(r["id"]): r["name"] for r in self.q("SELECT id, name FROM entities WHERE project = ?",
                                                                (project,))})
        return labels

    # ------------------------------------------------------------- snapshots

    def snapshot(self, project: int, label: str) -> int:
        dump = {t: [dict(r) for r in self.q(f"SELECT * FROM {t} WHERE project = ?", (project,))]
                for t in ("papers", "entities", "edges")}
        dump["scope"] = (self.project(project) or {}).get("scope")
        with self.tx() as c:
            version = c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM snapshots WHERE project = ?",
                                (project,)).fetchone()[0]
            c.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
                      (project, version, label, time.time(), json.dumps(dump)))
        return version

    def snapshots(self, project: int) -> list[dict]:
        return [dict(r) for r in self.q("SELECT version, label, created FROM snapshots WHERE project = ? "
                                        "ORDER BY version DESC", (project,))]

    def restore(self, project: int, version: int | None = None) -> str | None:
        """Restore a snapshot (default: latest) and drop it and every newer one. Returns its label."""
        sql = "SELECT * FROM snapshots WHERE project = ? " + ("AND version = ?" if version else
                                                               "ORDER BY version DESC LIMIT 1")
        rows = self.q(sql, (project, version) if version else (project,))
        if not rows:
            return None
        snap, dump = rows[0], json.loads(rows[0]["data"])
        with self.tx() as c:
            for table in ("papers", "entities", "edges"):
                allowed = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                c.execute(f"DELETE FROM {table} WHERE project = ?", (project,))
                if dump[table]:
                    cols = [k for k in dump[table][0] if k in allowed]
                    c.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                  [tuple(r[k] for k in cols) for r in dump[table]])
            if dump.get("scope"):
                c.execute("UPDATE projects SET scope = ? WHERE id = ?", (json.dumps(dump["scope"]), project))
            c.execute("DELETE FROM snapshots WHERE project = ? AND version >= ?", (project, snap["version"]))
        self.reindex(project)
        return snap["label"]


def _edge_view(e: dict) -> dict:
    d = e["data"]
    if e["kind"] == "builds_on":
        return {"change_type": d.get("change_type"), "summary": d.get("summary"), "component": d.get("component")}
    if e["kind"] == "cites":
        return {"influential": d.get("influential", False)}
    return {}
