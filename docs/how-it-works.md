# How it works

```mermaid
flowchart LR
  P([Prompt]) --> S[Scope]
  S --> R[Search & screen]
  R --> B[Snowball]
  B --> X[Extract]
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
   combines relevance, citation count and how often papers already in the graph cite the
   candidate.
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

   Its inputs are the abstract, the TL;DR and the citation contexts in which *other* papers
   describe this one. Those contexts are a rich source of limitations.
6. **Gap filling.** If the extraction reports missing fields, research-forest looks for
   passages without reading the whole paper (see [below](#gap-filling)), then re-extracts.
7. **Lineage.** Each citation between two graph papers is judged once: does the citing paper
   *build on* the cited one? If so, the edge gets a change type, the component that changed and
   a one-sentence summary. Each citing paper takes a single LLM call, covering up to 15 of its
   references.
8. **Normalise.** Duplicate methods and datasets are merged ("WT103" → "WikiText-103"). Exact
   duplicates are merged by key; aliases by one LLM call per entity type.
9. **Synthesise.** Limitations from all papers are grouped into 5–15 **themes**. Each theme is
   linked to the papers that have it and to the papers that address it. From these themes come
   5–12 **open research questions**, each linked to its motivating themes and relevant papers.

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
2. **Local BM25 search.** If S2 has no snippets, the open-access PDF or HTML (or the arXiv PDF)
   is downloaded in a [sandboxed worker](security.md#sandboxed-full-text-processing). The worker
   splits the text into overlapping windows of 120 words and ranks them with BM25, a standard
   keyword-ranking algorithm.

Only the top passages are sent to the LLM. **Deepen** forces this for every field.

## Incremental processing and cost

Work is tracked so that refinements only pay for what is new:

| Step | Runs |
|---|---|
| Citation links | once per paper |
| Extraction | once per paper (again on `deepen`) |
| Builds-on judgement | once per citation pair |
| Normalisation | when new papers were extracted |
| Synthesis | after a build and after each refinement |

LLM responses are cached on disk, keyed by backend, model, prompt and schema, so re-running an
identical step costs nothing.

A 60-paper build makes roughly 60 extraction calls, up to 60 lineage calls, 10–15 screening
calls, plus a few scope, normalisation and synthesis calls. Some extractions take a second call
when gaps are filled.

## Structured output

Every call requests a strict JSON schema, generated from Pydantic models with all fields
required and no extra properties. The output is validated, and invalid output is retried with
the error fed back to the model. Each backend gets the schema in the form it supports:

- `openai`: `response_format: json_schema` (strict). It falls back automatically to non-strict
  mode, then to `json_object`, then to a plain-text instruction for servers that don't support
  the stricter modes.
- `claude-code`: `claude -p --output-format json --json-schema …`
- `codex`: `codex exec --output-schema …`
