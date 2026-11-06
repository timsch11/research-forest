# How it works

```mermaid
flowchart LR
  P([Prompt]) --> S[Scope]
  S --> R[Search & screen]
  R --> B[Snowball]
  B --> X[Extract]
  X --> F[Follow-ups]
  F -.-> X
  X -. missing info .-> G[(Gap filling)]
  G -.-> X
  X --> L[Lineage]
  L --> N[Normalise]
  N --> Y[Synthesise]
  Y --> U{{Explore · refine · ask}}
  U -- refinement --> R
```

## Pipeline

1. **Scope.** The LLM turns your prompt into a title, boundaries (include and exclude criteria),
   4–8 keyword queries, a few landmark paper titles and an optional start year.
2. **Search and screen.** Queries and title matches run against Semantic Scholar. The LLM rates
   each candidate's relevance from 0 to 3, in batches of 25. Landmark papers it named itself
   start at 3.
3. **Select.** About 60% of the paper budget is filled from the ranked candidates. Ranking
   combines relevance, *citations per year since publication* (raw totals would favour old
   papers), *highly influential citations per year* (Semantic Scholar's label for citations by
   papers that use the work heavily; these weigh more than mentions in passing) and how often
   papers already in the graph cite the candidate.
4. **Snowball.** References of the graph are fetched together with their *citation contexts*
   (the sentences in which one paper cites another) and intents. References cited by two or more
   graph papers are screened and added. The rest of the budget is then topped up from the
   relevant candidates.
5. **Extract.** For every paper, the LLM extracts:
   - the contribution;
   - methods proposed and used;
   - datasets, with how they're used;
   - limitations, each marked *stated* (by the authors), *citing* (reported by later papers) or
     *inferred*;
   - future work.

   Its inputs are the abstract (or the TL;DR when there is none) and two kinds of citation
   sentences. Both come free with the citation lists, at no extra cost:
   - **The paper's own sentences** where it cites other work. Semantic Scholar takes them from
     the paper's full text, and the ones that read like evaluation statements ("we evaluate on
     GLUE [12]") are the best evidence for datasets.
   - **Sentences from other papers** that describe this one. These are a rich source of
     limitations.
6. **Gap filling.** Abstracts usually name only headline benchmarks, so by default the full
   text is searched for datasets for every paper (`always_fill_datasets`). This lifted dataset
   recall on the gold set from 68% to 82%. Limitations and methods are gap-filled when the model
   flags them as missing or returns them empty. Passages come from the paper's text (see
   [below](#gap-filling)). With `single_pass` (the default) the full text is read *before*
   extraction: dataset passages, table summaries and limitation sections go into the one
   extraction call. A small follow-up call fills only fields that are still empty and had no
   full-text evidence. Without it, extraction runs on the abstract first and a follow-up call
   fills the gaps from the passages.
7. **Baselines.** Extraction and gap filling also return the paper's *baselines*: the methods
   it compares against, mostly read from the main results tables. Each one is resolved to a
   paper without an LLM, trying these in order:
   - the table cell's citation ("Linformer [84]") → its bibliography entry → the reference with
     that arXiv id or title;
   - a graph paper that proposes a method of that name;
   - a reference titled "Name: …", or the only reference whose title contains the name
     ("Performer" → "Rethinking Attention with Performers").

   Names are also tried without qualifiers ("XLNet (K = 6)" → XLNet, "Megatron-LM
   implementation" → Megatron-LM) and with a parenthetical alias ("BiT (Big Transfer)").
   Semantic Scholar has no reference list for some papers, Mamba for example. For those, a
   baseline that matches an entry in the paper's own HTML bibliography is looked up by arXiv id
   or title (capped, and cached per project).

   On the gold set, 93% of gold baselines are extracted. About half of all extracted baselines
   resolve to a paper, and a random sample of resolutions was all correct. The rest are mostly
   generic names such as "LSTM" or "Previous best". Resolved baselines become `compares_with`
   edges. Papers that graph papers compare against
   but the graph lacks are strong discovery candidates: a build adds up to `baseline_papers` of
   them (after screening) and analyses them too. Being compared against also boosts a
   candidate's ranking.
8. **Lineage.** Each citation between two graph papers is judged once: does the citing paper
   *build on* the cited one? If so, the edge gets a change type, the component that changed and
   a one-sentence summary. The prompt notes when the cited paper is one of the citing paper's
   baselines; that alone doesn't decide it, since papers often compare against the work they
   extend. It also notes Semantic Scholar's *influential* flag, which is strong evidence.
   Influential citations are always judged, beyond the `max_refs_per_paper` cap. Citations
   that Semantic Scholar classified as background only (not methodology), that aren't
   influential and whose cited paper isn't a baseline are skipped without an LLM call. On a
   25-paper build this dropped 13% of the pairs and lost only two "builds on the Transformer"
   edges. Because pairs are batched around shared papers, it saves prompt length rather than
   calls. Pairs are batched (up to 15 per call) around a shared anchor paper.
   During a build that is usually the citing paper. After a refinement adds papers, it is usually
   the new papers, so older papers that cite them aren't re-sent one by one.
9. **Follow-ups.** Search and snowballing favour papers that are already cited. To find
   recent work that isn't visible yet, the citations of the graph's key papers (most built upon
   and cited within the graph; `followup_seeds`) are fetched newest first, one request each,
   restricted to the last `followup_years` years, and only the *influential* ones are kept. On
   FlashAttention, the newest 1,000 citations hold about 70 influential ones, almost all with
   no citations of their own. Papers that follow up several key papers rank first. They are
   screened against the scope (influential citers include other fields that merely reuse a
   method), ranked without citation counts, and up to `followup_papers` are added and analysed.
   The refinement op `followups` does the same for chosen papers.
10. **Normalise.** Duplicate methods and datasets are merged ("WT103" → "WikiText-103"). Exact
   duplicates are merged by key; aliases by one LLM call per entity type. After the first run,
   only *new* names are sent, together with existing names that share a word, a number or
   initials with them.
11. **Synthesise.** Limitations from all papers are grouped into 5–15 **themes**. Each theme is
   linked to the papers that have it and to the papers that address it. From these themes come
   5–12 **open research questions**, each linked to its motivating themes and relevant papers.

## Keeping a landscape current

**Check for new work** (`research-forest update ID`) asks Semantic Scholar only for papers
published since the last check, which is recorded at every build and check:

- the scope's search queries again, restricted by publication date;
- the influential citations of the key papers, as in [follow-ups](#pipeline), from that date.

Candidates are screened and ranked without citation counts, since new papers have none yet. Up
to `update_papers` are added and analysed, lineage and open questions are updated, and the new
papers are listed and outlined in the UI until the next check. A check takes a snapshot first,
so Undo reverts it, including the date of the last check.

## Lineage and change types

| Change type | Meaning |
|---|---|
| `architecture` | Changes model structure or components |
| `objective` | Changes the loss, the training objective or the problem formulation |
| `training` | Changes optimisation, schedule, fine-tuning or training procedure |
| `data` | Changes training data, augmentation or data collection |
| `efficiency` | Same goal, but cheaper, faster or smaller |
| `evaluation` | New benchmark, metric or evaluation protocol |
| `theory` | Theoretical analysis or explanation of the prior work |
| `application` | Applies the prior method to a new domain or task |
| `other` | None of the above |

Adopting and modifying, extending, re-implementing more efficiently, or replacing a component
all count as *builds on*. Background citations, comparison baselines and shared datasets do not.

## Graph model

| Node | Id | Created by |
|---|---|---|
| Paper | `p:<Semantic Scholar id>` | Search, snowball, refinement |
| Method, dataset | `e:<n>` | Extraction (then normalised) |
| Limitation theme, open question | `e:<n>` | Synthesis (rebuilt after each change) |

| Edge | From → to | Data |
|---|---|---|
| `cites` | paper → paper | citation contexts, intents, influential flag |
| `builds_on` | paper → paper | change type, component, summary |
| `compares_with` | paper → paper (or method, when unresolved) | baseline name as written, how it was resolved |
| `proposes` / `uses` | paper → method | – |
| `evaluates_on` | paper → dataset | usage (training, evaluation, both) |
| `has_limitation` / `addresses` | paper → limitation theme | – |
| `motivates` | limitation theme → question | – |
| `about` | question → paper | – |

Everything lives in SQLite: papers, entities, edges, an FTS5 search index and snapshots for
undo.

## Gap filling

When an abstract doesn't name the datasets or limitations, research-forest tries two sources
in order:

1. **Semantic Scholar snippet search.** Full-text snippets from S2's index, queried per missing
   field. Extra search terms suggested by the LLM are added to each query.
2. **Local BM25 search.** If S2 has no snippets, the paper's text is downloaded in a
   [sandboxed worker](security.md#sandboxed-full-text-processing), trying sources in order:
   - **arXiv's HTML rendering** (`arxiv.org/html/<id>`), which also exists for most older papers.
     Its tables come through as rows ("BigBird | 75.7 | 86.8"), and the bibliography, author
     blocks and math markup are dropped. That matters because datasets are usually named in
     tables and captions, which PDF text extraction scrambles.
   - the open-access PDF;
   - the arXiv PDF.

   Besides keyword-ranked passages, reading the document yields two structured inputs:
   - **Table summaries:** for datasets, each main-body table's caption, column headers and row
     labels, e.g. "Table 2: Dev results… — columns: Model | SST-2 | IMDB | QNLI | QQP". Appendix
     and supplementary tables are skipped. This is HTML only.
   - **Limitation sections:** the Limitations, Conclusion, Discussion and Future-work sections
     (`limitation_sections`). They fill limitations whenever the document is read. For PDFs
     they're found with a heading heuristic.

   Dataset passages come from two queries: one for evaluation benchmarks and one for training
   and pre-training corpora, which sit in setup sections rather than results.

   Pages without an HTML rendering fall through to the next source. The worker splits the
   text into overlapping windows of 120 words and ranks them with BM25, a standard
   keyword-ranking algorithm.

Both searches add the dataset and method names already used most in the graph to their
queries. If several papers evaluate on LRA or PG-19, passages naming those rank first.

Only the top passages are sent to the LLM, together with the title and the one-line
contribution, but not the abstract or citation contexts again. **Deepen** does this for every
field and reuses the existing extraction.

## Incremental processing and cost

Work is tracked so that refinements only pay for what is new:

| Step | Runs |
|---|---|
| Citation links | once per paper |
| Extraction | once per paper (again on `deepen`) |
| Builds-on judgement | once per citation pair |
| Normalisation | when new methods or datasets appeared; only new names plus likely aliases are sent |
| Synthesis | after a build, and after refinements that changed the papers (or asked for it) |

LLM responses are cached on disk, keyed by backend, model, prompt and schema, so re-running an
identical step costs nothing.

A 60-paper build makes roughly 60 extraction calls, up to 60 lineage calls, 5–15 screening
calls, plus a few scope, normalisation and synthesis calls. Papers with gaps take one extra
small gap-fill call.

### Keeping prompts lean

Prompt size is budgeted per item (see the constants at the top of `prompts.py`):
- Each abstract and set of citation contexts is sent once. The TL;DR is only used when there
  is no abstract, and repeated citation sentences are deduplicated.
- Lineage prompts carry two short citation contexts and a short contribution per pair.
- Q&A sends the relevant part of the graph, not the whole graph. Method and dataset links are
  inlined into paper lines, plain citations are left out, and open questions are only attached
  when the question is about gaps or the questions are linked to the hits.
- Follow-up questions re-use the nodes the previous answer cited, plus the previous question.
  The previous answer itself is never re-sent, so context doesn't pile up over a conversation.
- Schemas are sent as compact JSON, without descriptions that only repeat the prompt.

As a reference point, a synthetic 65-paper build plus a refinement and three questions dropped
from about 406k to 292k prompt tokens with these measures, roughly 28% less.

The CLI backends are agents with a fixed prompt of their own, sent with every call:

| Backend | Fixed overhead per call | What research-forest does about it |
|---|---|---|
| `claude-code` | About 3.4k tokens, mostly prompt-cache reads, re-read on each of its 2 turns (`--json-schema` returns the answer through a tool) | Skips your own setup (`--setting-sources ""`): a 2k-word `~/.claude/CLAUDE.md` alone otherwise added about 19k input tokens per call |
| `codex` | About 9k tokens of built-in tools, skills and agent text, plus your global `AGENTS.md` | Replaces Codex's 22k-character agent instructions with our system prompt, and skips `~/.codex/config.toml` (`--ignore-user-config`) |

Both use a small model by default (`haiku`, `gpt-6-luna`) and `effort = "low"`, because
thinking tokens otherwise outweigh the answer: about 1.2k thinking tokens per extraction call on
Haiku at default effort, against about 350 tokens of JSON. `research-forest llm-log` shows the
tokens each backend reported.

## Structured output

Every call requests a strict JSON schema, generated from Pydantic models with all fields
required and no extra properties. The output is validated, and invalid output is retried with
the error fed back to the model. Each backend gets the schema in the form it supports:

- `openai`: `response_format: json_schema` (strict). It falls back automatically to non-strict
  mode, then to `json_object`, then to a plain-text instruction for servers that don't support
  the stricter modes.
- `claude-code`: `claude -p --output-format json --json-schema …`
- `codex`: `codex exec --json --output-schema …`
