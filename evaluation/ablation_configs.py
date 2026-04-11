"""
ablation_configs.py
-------------------
Unified configuration for the combined ablation study.

Five architectures (A-E) are evaluated in BOTH retrieval and generation
experiments, making results directly comparable across both evaluation axes.

Mix-and-match design (not a simple progressive stack):
    A  Baseline               BM25-only — absolute floor
    B  Lightweight Precision  BM25 + Time Weighting + Reranker
    C  Semantic Stack         Filter + Hybrid + Reranker
    D  Graph without Precision Filter + Hybrid + GraphRAG
    E  Full System            Filter + Hybrid + Time Weighting + Reranker + GraphRAG
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    label: str
    use_metadata_filter: bool
    use_hybrid: bool          # False = BM25-only; True = BM25 + semantic
    use_time_weighting: bool
    use_reranker: bool
    use_graph_rag: bool


# ── Architecture registry (A–E) ───────────────────────────────────────────────

PIPELINE_CONFIGS: dict[str, AblationConfig] = {
    "A": AblationConfig(
        label               = "Baseline",
        use_metadata_filter = False,
        use_hybrid          = False,
        use_time_weighting  = False,
        use_reranker        = False,
        use_graph_rag       = False,
    ),
    "B": AblationConfig(
        label               = "Lightweight Precision",
        use_metadata_filter = False,
        use_hybrid          = False,
        use_time_weighting  = True,
        use_reranker        = True,
        use_graph_rag       = False,
    ),
    "C": AblationConfig(
        label               = "Semantic Stack",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = False,
        use_reranker        = True,
        use_graph_rag       = False,
    ),
    "D": AblationConfig(
        label               = "Graph w/o Precision",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = False,
        use_reranker        = False,
        use_graph_rag       = True,
    ),
    "E": AblationConfig(
        label               = "Full System",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = True,
        use_reranker        = True,
        use_graph_rag       = True,
    ),
}

# ── Generation strategies (Experiment 2) ─────────────────────────────────────

GENERATION_STRATEGIES: list[str] = [
    "zero_shot",   # Direct generation — floor
    "few_shot",    # One worked example prepended
    "citation",    # Mandatory inline citations (reused from Exp 1 cache)
    "reflection",  # Two-pass: draft → self-critique → revised
]

# Fixed strategy for Experiments 0 & 1
EXP1_FIXED_STRATEGY: str = "citation"

# All 50 QA pair IDs
SELECTED_QA_IDS: list[int] = list(range(1, 51))
