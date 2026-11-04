"""Domain objects and the structured-output schemas requested from the LLM.

Every LLM schema uses only required fields (nullable where optional) so it can
be sent as a *strict* JSON schema to OpenAI-compatible APIs and Codex.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    title: str = Field(description="Short name of the research area")
    summary: str = Field(description="2-3 sentence description of the area's boundaries")
    search_queries: list[str] = Field(description="4-8 diverse keyword queries for a paper search engine")
    seed_papers: list[str] = Field(description="Exact titles of up to 8 foundational or landmark papers")
    include_criteria: str
    exclude_criteria: str
    year_min: int | None = Field(description="Earliest publication year of interest, or null")


class ScreenItem(BaseModel):
    ref: int
    relevance: int = Field(description="0 = off-topic, 1 = tangential, 2 = relevant, 3 = core")


class Screening(BaseModel):
    items: list[ScreenItem]


class MethodItem(BaseModel):
    name: str = Field(description="Canonical short name, e.g. 'FlashAttention'")
    description: str = Field(description="One sentence")


class DatasetItem(BaseModel):
    name: str = Field(description="Canonical dataset/benchmark name, e.g. 'ImageNet-1k'")
    usage: Literal["training", "evaluation", "both"]


class LimitationItem(BaseModel):
    text: str = Field(description="One concise sentence")
    source: Literal["stated", "citing", "inferred"] = Field(
        description="stated by the authors, reported by citing papers, or inferred by you")


MissingField = Literal["methods", "datasets", "limitations"]


class Extraction(BaseModel):
    contribution: str = Field(description="One sentence: the paper's key contribution")
    methods_proposed: list[MethodItem]
    methods_used: list[str] = Field(description="Names of existing methods the paper builds on or compares to")
    datasets: list[DatasetItem]
    limitations: list[LimitationItem]
    future_work: list[str]
    missing: list[MissingField] = Field(
        description="Fields for which the provided text was insufficient to answer confidently")
    search_terms: list[str] = Field(
        description="Keywords likely to appear in the full-text passages that would fill the missing fields")


class Relation(BaseModel):
    ref: int
    builds_on: bool = Field(description="True only if the citing paper directly extends/modifies the cited work")
    change_type: ChangeType | None
    component: str | None = Field(description="Which part of the cited work is changed, e.g. 'attention kernel'")
    summary: str | None = Field(description="One sentence: what the citing paper changes relative to the cited one")


class Relations(BaseModel):
    relations: list[Relation]


class Group(BaseModel):
    canonical: str
    members: list[int]


class Grouping(BaseModel):
    groups: list[Group]


class Theme(BaseModel):
    name: str = Field(description="Short label for a shared limitation")
    description: str
    papers: list[int] = Field(description="Refs of papers exhibiting this limitation")
    addressed_by: list[int] = Field(description="Refs of papers that (partially) address it")


class OpenQuestion(BaseModel):
    question: str
    rationale: str
    themes: list[int] = Field(description="Indices of the themes motivating this question")
    papers: list[int] = Field(description="Refs of the most relevant papers")


class Synthesis(BaseModel):
    themes: list[Theme]
    questions: list[OpenQuestion]


RefineAction = Literal[
    "search", "expand", "remove", "filter", "prune", "deepen", "rescope", "resynthesize"]


class RefineOp(BaseModel):
    action: RefineAction
    query: str | None = Field(description="search: query; prune: criterion to REMOVE; rescope: new focus")
    papers: list[int] = Field(description="Paper refs for expand/remove/deepen")
    direction: Literal["references", "citations", "both"] | None
    limit: int | None
    year_min: int | None
    year_max: int | None
    min_citations: int | None


class RefinePlan(BaseModel):
    explanation: str
    ops: list[RefineOp]


class Answer(BaseModel):
    answer: str = Field(description="Markdown answer that cites nodes inline as [n12]")
    nodes: list[str] = Field(description="Node handles (e.g. 'n12') supporting the answer")
