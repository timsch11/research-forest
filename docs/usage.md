# Using the explorer

## Web UI

```bash
research-forest serve [--host 127.0.0.1] [--port 8000]
```

Open the URL it prints. The token in it is exchanged for a cookie, so later visits to
`http://127.0.0.1:8000/` just work.

| Area | What it does |
|---|---|
| **Lineage view** | A timeline with papers placed by year. Arrows are *builds-on* edges, labelled with the [kind of change](how-it-works.md#lineage-and-change-types). Faint dashed lines are plain citations. |
| **Landscape view** | Every node type together (papers, methods, datasets, limitation themes, open questions), laid out by force direction. |
| **Filters** | Node types, a year range and change types. Entities that lose all their visible papers are hidden too. |
| **Search** | Instant keyword search over every node, with highlighted snippets. Click a result to focus it. |
| **Details panel** | For a paper: its contribution, what it builds on and what changed, who builds on it, methods, datasets, limitations (marked *stated*, *citing* or *inferred*), future work, limitation themes and related open questions. For an entity: every connected paper, grouped by relation. |
| **Open questions** | The synthesised research questions. Click one to see its motivating themes and papers. |
| **Refine / Ask** | The command bar. See below. |
| **Progress card** | While a build or refinement runs: a stage stepper, an overall progress bar, the current step with counts (e.g. *Judging 275 citations… 24 / 39*) and elapsed time. It ends with a run summary. |
| **Progress log** | Every step in plain language: what was searched, how many papers were rated core or relevant, how many were added, how many needed full-text passages, and so on. |

The UI follows your system's light or dark theme and works offline; Cytoscape.js is bundled.

## Refining the graph

Type what you want in plain language. The LLM turns it into a plan made of these operations:

| Operation | Example command | Effect |
|---|---|---|
| `search` | *"add work on ring attention"* | New Semantic Scholar query. Hits are screened and the best ones added. |
| `expand` | *"expand around FlashAttention"* | Pulls in references and/or citations of the chosen papers. |
| `remove` | *"drop the Reformer paper"* | Removes specific papers. |
| `filter` | *"only work since 2021"*, *"at least 100 citations"* | Removes papers outside year or citation bounds. |
| `prune` | *"drop vision papers"* | Semantic filter: the LLM flags papers matching the description. |
| `deepen` | *"look deeper into Performer's limitations"* | Re-analyses papers using full-text passages. |
| `rescope` | *"focus on state-space models instead"* | Updates the scope, re-screens the graph and searches for the new focus. |
| `resynthesize` | *"refresh the open questions"* | Recomputes limitation themes and open questions. |

One command can combine several operations: *"expand around Mamba and drop everything before
2020"*. Only new work is computed. Each refinement takes a snapshot first, so **Undo** (or
`research-forest undo <project>`) restores the previous state, and repeated undos step further
back.

## Asking questions

Switch the command bar to **Ask**:

- *"Which efficient-attention methods were never evaluated on long-document benchmarks?"*
- *"What limitations do linear-attention papers share?"*
- *"Which papers address the memory cost of exact attention?"*

The answer comes from the part of the graph relevant to your question. Its references are
clickable and highlighted in the graph. When the graph doesn't contain enough to answer, the
answer says so.

Follow-ups work: *"…and which of those were evaluated on PG-19?"* re-uses the nodes the last
answer cited. Only that previous question is passed along, not its answer, so follow-ups stay
as cheap as a fresh question.

## CLI reference

```text
research-forest [--db FILE] [--backend openai|claude-code|codex] [--model NAME]
                [--base-url URL] [--no-fulltext] [-v] COMMAND
```

| Command | Description |
|---|---|
| `build PROMPT [-n PAPERS]` | Build a new landscape and print a summary |
| `list` | List landscapes with their id and status |
| `refine ID COMMAND` | Apply a natural-language refinement |
| `ask ID QUESTION` | Answer a question from the graph |
| `search ID QUERY` | Keyword search, showing kind, node id and title |
| `undo ID` | Undo the last refinement |
| `serve [--host H] [--port P]` | Run the web explorer |

All landscapes live in one SQLite file (`research.forest.db` by default), and LLM responses are
cached next to it in `research.forest.db.llmcache`.
