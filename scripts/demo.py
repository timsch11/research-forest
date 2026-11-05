"""Serve a hand-curated demo landscape — no API keys or network needed.

    python scripts/demo.py            # then open the printed URL

The data (efficient attention / long-context models) is illustrative: titles and years are
real, citation counts and lineage summaries are approximate. Build and Refine/Ask need a real
LLM backend; browsing, filtering and search work offline. Used for the README screenshots.
"""

import argparse
import os
import tempfile

import uvicorn

from research_forest.config import Settings
from research_forest.engine import Forest
from research_forest.models import DatasetItem, Extraction, LimitationItem, MethodItem, Paper, Scope
from research_forest.pipeline.analyze import apply_extraction
from research_forest.pipeline.context import Ctx
from research_forest.store import Store, pid
from research_forest.web import create_app


def seed(s: Store, settings: Settings) -> int:
    p = s.create_project("Efficient attention and long-context sequence models")
    s.set_scope(p, Scope(title="Efficient long-context sequence modelling",
                         summary="Methods that make attention or its replacements scale to long sequences.",
                         search_queries=["efficient attention", "linear attention", "state space models"],
                         seed_papers=[], include_criteria="efficient attention, long context, SSMs",
                         exclude_criteria="unrelated domains", year_min=None))
    P = {  # id: (title, year, citations, venue)
     "T": ("Attention Is All You Need", 2017, 130000, "NeurIPS"),
     "TXL": ("Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", 2019, 4000, "ACL"),
     "SP": ("Generating Long Sequences with Sparse Transformers", 2019, 2000, "arXiv"),
     "RF": ("Reformer: The Efficient Transformer", 2020, 2400, "ICLR"),
     "LF": ("Longformer: The Long-Document Transformer", 2020, 4000, "arXiv"),
     "LN": ("Linformer: Self-Attention with Linear Complexity", 2020, 1900, "arXiv"),
     "LT": ("Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", 2020, 1900, "ICML"),
     "BB": ("Big Bird: Transformers for Longer Sequences", 2020, 2200, "NeurIPS"),
     "PF": ("Rethinking Attention with Performers", 2020, 1800, "ICLR"),
     "LRA": ("Long Range Arena: A Benchmark for Efficient Transformers", 2020, 800, "ICLR"),
     "S4": ("Efficiently Modeling Long Sequences with Structured State Spaces", 2021, 1800, "ICLR"),
     "FA": ("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022, 2600, "NeurIPS"),
     "H3": ("Hungry Hungry Hippos: Towards Language Modeling with State Space Models", 2022, 500, "ICLR"),
     "HY": ("Hyena Hierarchy: Towards Larger Convolutional Language Models", 2023, 400, "ICML"),
     "FA2": ("FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023, 1200, "ICLR"),
     "RN": ("Retentive Network: A Successor to Transformer for Large Language Models", 2023, 500, "arXiv"),
     "MB": ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023, 3000, "COLM"),
     "RA": ("Ring Attention with Blockwise Transformers for Near-Infinite Context", 2023, 250, "ICLR"),
     "M2": ("Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", 2024, 500, "ICML"),
     "FA3": ("FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024, 200, "NeurIPS"),
    }
    s.add_papers(p, [Paper(id=k, title=t, year=y, citation_count=c, venue=v, abstract=f"{t}.", url=f"https://www.semanticscholar.org/search?q={t.split(':')[0]}") for k, (t, y, c, v) in P.items()])
    s.update_papers(p, list(P), status="included", relevance=3, refs_fetched=1)
    B = [  # citing, cited, change_type, component, summary
     ("TXL","T","architecture","context handling","Adds segment-level recurrence so attention reaches beyond a fixed window."),
     ("SP","T","efficiency","attention pattern","Factorises attention into sparse strided patterns, O(n√n)."),
     ("RF","T","efficiency","attention kernel","Replaces full attention with LSH bucketing and reversible layers."),
     ("LF","SP","efficiency","attention pattern","Sliding-window plus task-specific global attention."),
     ("LN","T","efficiency","attention matrix","Projects keys and values to a low-rank space for linear cost."),
     ("LT","T","efficiency","softmax","Kernel feature maps turn attention into a linear recurrence."),
     ("BB","LF","architecture","attention pattern","Adds random attention to window + global tokens; proves universality."),
     ("PF","LT","efficiency","softmax approximation","Unbiased random-feature approximation of softmax (FAVOR+)."),
     ("LRA","T","evaluation","benchmarking","Unified long-sequence benchmark for efficient Transformers."),
     ("S4","T","architecture","sequence mixer","Replaces attention with a structured state-space layer."),
     ("FA","T","efficiency","attention kernel","Exact attention, re-implemented with IO-aware tiling."),
     ("H3","S4","architecture","SSM layer","Adds shift + diagonal SSMs to recall tokens for language modelling."),
     ("HY","H3","architecture","sequence mixer","Long implicit convolutions with data-controlled gating."),
     ("FA2","FA","efficiency","work partitioning","Better parallelism and fewer non-matmul FLOPs."),
     ("RN","LT","architecture","recurrence","Retention with parallel, recurrent and chunked forms."),
     ("MB","S4","architecture","SSM parameters","Input-dependent (selective) SSM parameters with a hardware-aware scan."),
     ("MB","H3","architecture","block design","Merges H3 and MLP into a single simplified block."),
     ("RA","FA","efficiency","distribution","Blockwise attention distributed in a ring across devices."),
     ("M2","MB","theory","SSM–attention duality","Shows SSMs and attention are duals; derives a faster SSD layer."),
     ("FA3","FA2","efficiency","attention kernel","Asynchrony and FP8 on Hopper GPUs."),
    ]
    s.add_edges(p, [(pid(a), pid(b), "builds_on", {"change_type": c, "component": comp, "summary": sm}) for a, b, c, comp, sm in B])
    s.add_edges(p, [(pid(a), pid(b), "cites", {"contexts": [], "judged": True}) for a, b, *_ in B]
                + [(pid(x), pid("LRA"), "cites", {"judged": True}) for x in ("S4", "MB", "HY", "H3")])
    methods = {"T":"Transformer","TXL":"Transformer-XL","SP":"Sparse Transformer","RF":"Reformer","LF":"Longformer","LN":"Linformer",
               "LT":"Linear attention","BB":"BigBird","PF":"Performer","S4":"S4","FA":"FlashAttention","H3":"H3","HY":"Hyena",
               "FA2":"FlashAttention-2","RN":"RetNet","MB":"Mamba","RA":"Ring Attention","M2":"Mamba-2","FA3":"FlashAttention-3"}
    datasets = {"LRA":["T","LN","PF","RF","BB","S4","LT"], "WikiText-103":["TXL","S4","H3","HY","LT"], "enwik8":["TXL","SP","LF","RF"],
                "The Pile":["H3","HY","MB","RN","M2"], "PG-19":["RA","HY"], "GPT-2 training":["FA","FA2"]}
    limit_text = {
     "T":"Quadratic time and memory in sequence length", "LN":"Approximation hurts autoregressive decoding", "PF":"Random-feature variance degrades accuracy",
     "FA":"Requires custom CUDA kernels tied to specific GPUs", "S4":"Weaker in-context recall than attention", "MB":"Struggles on associative recall vs. attention",
     "LRA":"Synthetic tasks may not reflect real long-context use", "RF":"LSH adds overhead for moderate lengths"}
    for k in P:
        mp = [MethodItem(name=methods[k], description=f"{methods[k]} method")] if k in methods else []
        used = ["Transformer"] if k not in ("T",) else []
        ds = [DatasetItem(name=d, usage="evaluation") for d, ps in datasets.items() if k in ps]
        lim = [LimitationItem(text=limit_text.get(k, "Gains shrink at short sequence lengths"), source="stated")]
        ex = Extraction(contribution=f"{P[k][0].split(':')[0]}: " + next((b[4] for b in B if b[0] == k), "Introduces the Transformer architecture based solely on attention."),
                        methods_proposed=mp, methods_used=used, datasets=ds, limitations=lim, future_work=["Scale to longer contexts"], missing=[], search_terms=[])
        apply_extraction(Ctx(s, None, None, None, settings, p), k, ex)
    themes = [("Quadratic cost of exact attention", ["T","TXL","LRA"], ["SP","RF","LF","LN","LT","PF","BB","FA","S4","MB"]),
              ("Approximation gap vs. exact attention", ["LN","PF","RF","LT"], ["FA","FA2"]),
              ("Hardware-specific kernels", ["FA","FA2","FA3","MB"], []),
              ("Weak associative recall in recurrent models", ["S4","H3","HY","RN"], ["MB","M2"]),
              ("Benchmarks don't reflect real long-context use", ["LRA"], [])]
    qs = [("Can sub-quadratic models match attention on in-context recall?", [3], ["MB","M2","H3"]),
          ("How should long-context ability be benchmarked beyond LRA?", [4], ["LRA"]),
          ("Can IO-aware kernels be made portable across accelerators?", [2], ["FA3"])]
    tn = []
    for name, has, addr in themes:
        n = s.upsert_entity(p, "limitation", name, name)
        tn.append(n)
        s.add_edges(p, [(pid(x), n, "has_limitation", None) for x in has] + [(pid(x), n, "addresses", None) for x in addr])
    for q, ths, ps in qs:
        n = s.upsert_entity(p, "question", q, "Open question")
        s.add_edges(p, [(tn[i], n, "motivates", None) for i in ths] + [(n, pid(x), "about", None) for x in ps])
    s.reindex(p)
    s.set_status(p, "ready")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--db", default=os.path.join(tempfile.mkdtemp(prefix="rf-demo-"), "demo.forest.db"))
    args = ap.parse_args()
    settings = Settings(db_path=args.db, openai_api_key="demo-no-llm", cache_llm=False)
    forest = Forest(settings)
    if not forest.store.projects():
        seed(forest.store, settings)
    app = create_app(settings, forest, token="demo")
    print(f"demo explorer: http://127.0.0.1:{args.port}/?token=demo")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
