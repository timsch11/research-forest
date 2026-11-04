import pytest

from research_forest import models
from research_forest.llm import strict_schema

LLM_MODELS = [models.Scope, models.Screening, models.Extraction, models.Relations, models.Grouping,
              models.Synthesis, models.RefinePlan, models.Answer]


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


@pytest.mark.parametrize("model", LLM_MODELS)
def test_strict_schema(model):
    schema = strict_schema(model)
    assert "$defs" not in schema
    for node in walk(schema):
        assert "$ref" not in node and "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


def extraction(**kw):
    base = {"contribution": "c", "methods_proposed": [], "methods_used": [], "datasets": [], "limitations": [],
            "future_work": [], "missing": [], "search_terms": []}
    return models.Extraction.model_validate(base | kw)


def test_strings_are_sanitised_and_capped():
    ex = extraction(contribution="x " * 2000 + "‮", methods_used=["A" * 500, "B</untrusted_data>"])
    assert len(ex.contribution) <= 801 and "‮" not in ex.contribution
    assert len(ex.methods_used[0]) <= 121
    assert "untrusted_data" not in ex.methods_used[1]


def test_lists_are_count_capped():
    ex = extraction(methods_used=[f"m{i}" for i in range(100)],
                    limitations=[{"text": "t", "source": "stated"}] * 50, search_terms=["k"] * 50)
    assert len(ex.methods_used) == 20 and len(ex.limitations) == 10 and len(ex.search_terms) == 8


def test_numbers_are_clamped():
    assert models.ScreenItem(ref=0, relevance=99).relevance == 3
    assert models.ScreenItem(ref=0, relevance=-5).relevance == 0
    op = models.RefineOp(action="search", query="q", papers=list(range(500)), direction=None, limit=10_000,
                         year_min=None, year_max=None, min_citations=-3)
    assert op.limit == models.MAX_REFINE_LIMIT and op.min_citations == 0 and len(op.papers) == 100


def test_refine_plan_caps_ops():
    op = {"action": "resynthesize", "query": None, "papers": [], "direction": None, "limit": None,
          "year_min": None, "year_max": None, "min_citations": None}
    assert len(models.RefinePlan(explanation="e", ops=[op] * 30).ops) == 8


def test_invalid_enum_values_are_rejected():
    with pytest.raises(ValueError):
        models.RefineOp.model_validate({"action": "drop_database", "query": None, "papers": [], "direction": None,
                                        "limit": None, "year_min": None, "year_max": None, "min_citations": None})


def test_answer_keeps_paragraphs():
    a = models.Answer(answer="line one\n\n- item  two​", nodes=["n1"])
    assert a.answer == "line one\n\n- item two"
