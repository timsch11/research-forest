"""Domain objects and the structured-output schemas requested from the LLM.

Every LLM schema uses only required fields (nullable where optional) so it can
be sent as a *strict* JSON schema to OpenAI-compatible APIs and Codex.

LLM output is untrusted (it may be steered by injected text in papers), so all
strings are sanitised and length-capped, lists are count-capped and numbers are
clamped *after* validation — truncation instead of rejection keeps runs robust.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field

from .safety import sanitize


def _text(limit: int, keep_newlines: bool = False) -> Any:
    return Annotated[str, AfterValidator(lambda v: sanitize(v, limit, keep_newlines))]


def _capped(item: Any, count: int) -> Any:
    return Annotated[list[item], AfterValidator(lambda v: v[:count])]


def _clamped(lo: int, hi: int) -> Any:
    return Annotated[int, AfterValidator(lambda v: max(lo, min(hi, v)))]


Name = _text(120)
Short = _text(300)
Sentence = _text(500)
Paragraph = _text(800)

# ---------------------------------------------------------------- domain ----


class Paper(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    tldr: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = Field(default_factory=list)
    citation_count: int = 0
    influential_count: int = 0  # citations Semantic Scholar classifies as highly influential
    url: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None


class CitationLink(BaseModel):
    citing: str
    cited: str
    contexts: list[str] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    influential: bool = False


EntityKind = Literal["method", "dataset", "limitation", "question"]
ChangeType = Literal[
    "architecture", "objective", "training", "data", "efficiency",
    "evaluation", "theory", "application", "other",
]
CHANGE_TYPES: tuple[str, ...] = ChangeType.__args__  # type: ignore[attr-defined]

# ------------------------------------------------------------ LLM schemas ----


class Scope(BaseModel):
    title: Name = Field(description="Short name of the research area")
    summary: Paragraph = Field(description="2-3 sentence description of the area's boundaries")
    search_queries: _capped(Name, 10) = Field(description="4-8 diverse keyword queries for a paper search engine")
    seed_papers: _capped(Short, 10) = Field(description="Exact titles of up to 8 foundational or landmark papers")
    include_criteria: Sentence
    exclude_criteria: Sentence
    year_min: int | None = Field(description="Earliest publication year of interest, or null")


class ScreenItem(BaseModel):
    ref: int
    relevance: _clamped(0, 3) = Field(description="0 = off-topic, 1 = tangential, 2 = relevant, 3 = core")


class Screening(BaseModel):
    items: list[ScreenItem]


class MethodItem(BaseModel):
    name: Name = Field(description="Canonical short name, e.g. 'FlashAttention'")
    description: Sentence = Field(description="One sentence")


class DatasetItem(BaseModel):
    name: Name = Field(description="Canonical dataset/benchmark name, e.g. 'ImageNet-1k'")
    usage: Literal["training", "evaluation", "both"]


class LimitationItem(BaseModel):
    text: Sentence = Field(description="One concise sentence")
    source: Literal["stated", "citing", "inferred"]


MissingField = Literal["methods", "datasets", "limitations"]


class Extraction(BaseModel):
    contribution: Paragraph = Field(description="One sentence")
    methods_proposed: _capped(MethodItem, 8)
    methods_used: _capped(Name, 20)
    baselines: _capped(Name, 20) = Field(
        default_factory=list,  # absent in extractions stored before baselines existed
        description="Other methods/models compared against in the experiments (not the paper's own method, "
                    "its variants or ablations; not datasets or metrics)")
    datasets: _capped(DatasetItem, 20)
    limitations: _capped(LimitationItem, 10)
    future_work: _capped(Sentence, 8)
    missing: list[MissingField]
    search_terms: _capped(_text(60), 8)


class GapFill(BaseModel):
    """Only the fields gap filling can add; merged into an existing :class:`Extraction`."""

    methods_used: _capped(Name, 20)
    baselines: _capped(Name, 20) = Field(
        default_factory=list,  # absent in extractions stored before baselines existed
        description="Other methods/models compared against in the experiments (not the paper's own method, "
                    "its variants or ablations; not datasets or metrics)")
    datasets: _capped(DatasetItem, 20)
    limitations: _capped(LimitationItem, 10)
    future_work: _capped(Sentence, 8)


class Relation(BaseModel):
    ref: int
    builds_on: bool
    change_type: ChangeType | None
    component: _text(160) | None = Field(description="e.g. 'attention kernel'")
    summary: Sentence | None


class Relations(BaseModel):
    relations: list[Relation]


class Group(BaseModel):
    canonical: Name
    members: list[int]


class Grouping(BaseModel):
    groups: list[Group]


class Theme(BaseModel):
    name: Name = Field(description="Short label for a shared limitation")
    description: Sentence
    papers: list[int] = Field(description="Refs of papers exhibiting this limitation")
    addressed_by: list[int] = Field(description="Refs of papers that (partially) address it")


class OpenQuestion(BaseModel):
    question: Short
    rationale: Paragraph
    themes: list[int] = Field(description="Indices of the themes motivating this question")
    papers: list[int] = Field(description="Refs of the most relevant papers")


class Synthesis(BaseModel):
    themes: _capped(Theme, 20)
    questions: _capped(OpenQuestion, 15)


RefineAction = Literal[
    "search", "expand", "followups", "remove", "filter", "prune", "deepen", "rescope", "resynthesize"]
MAX_REFINE_LIMIT = 50


class RefineOp(BaseModel):
    action: RefineAction
    query: Short | None = Field(description="search: query; prune: criterion to REMOVE; rescope: new focus")
    papers: _capped(int, 100) = Field(description="Paper refs for expand/followups/remove/deepen")
    direction: Literal["references", "citations", "both"] | None
    limit: _clamped(1, MAX_REFINE_LIMIT) | None
    year_min: int | None
    year_max: int | None
    min_citations: _clamped(0, 10**7) | None


class RefinePlan(BaseModel):
    explanation: Paragraph
    ops: _capped(RefineOp, 8)


class Answer(BaseModel):
    answer: _text(6000, keep_newlines=True) = Field(description="Markdown answer that cites nodes inline as [n12]")
    nodes: _capped(_text(12), 40) = Field(description="Node handles (e.g. 'n12') supporting the answer")
