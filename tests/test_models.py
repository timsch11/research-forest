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


def _all_models(model, seen=None):
    """The model and every nested BaseModel reachable from its fields."""
    import typing
    from pydantic import BaseModel
    seen = seen if seen is not None else {}
    if model.__name__ not in seen:
        seen[model.__name__] = model
        for f in model.model_fields.values():
            stack = [f.annotation]
            while stack:
                t = stack.pop()
                if isinstance(t, type) and issubclass(t, BaseModel):
                    _all_models(t, seen)
                stack += list(typing.get_args(t))
    return seen


def _schema_objects(schema, path="$"):
    """Yield (path, object schema) for every object node, following properties and items."""
    if schema.get("type") == "object":
        yield path, schema
        for name, sub in schema["properties"].items():
            yield from _schema_objects(sub, f"{path}.{name}")
    for key in ("items",):
        if key in schema:
            yield from _schema_objects(schema[key], f"{path}[]")
    for sub in schema.get("anyOf", []):
        yield from _schema_objects(sub, path)


@pytest.mark.parametrize("model", LLM_MODELS)
def test_strict_schema_keeps_every_field(model):
    """Regression: a field named like a JSON-Schema keyword ("title") must not be stripped."""
    schema = strict_schema(model)
    assert set(schema["properties"]) == set(model.model_fields)
    nested = {name: m for name, m in _all_models(model).items()}
    for path, obj in _schema_objects(schema):
        matches = [m for m in nested.values() if set(m.model_fields) == set(obj["properties"])]
        assert matches, f"{model.__name__} {path}: properties {sorted(obj['properties'])} match no model"


@pytest.mark.parametrize("model", LLM_MODELS)
def test_strict_schema_output_round_trips(model):
    """A response that fills exactly the schema's required fields must validate."""
    def fake(schema):
        if "anyOf" in schema:
            return None
        t = schema.get("type")
        if "enum" in schema:
            return schema["enum"][0]
        if t == "object":
            return {k: fake(v) for k, v in schema["properties"].items()}
        return {"array": [], "string": "x", "integer": 1, "number": 1.0, "boolean": False}[t]
    model.model_validate(fake(strict_schema(model)))


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
