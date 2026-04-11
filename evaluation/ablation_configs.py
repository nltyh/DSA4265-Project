"""
ablation_configs.py
-------------------
Configuration definitions for the two-experiment RAG evaluation study.

Experiment 1 — Pipeline Ablation
    Measures contribution of each retrieval component.
    Fixed strategy: citation
    Configs A → E (5 variants, each adds one component group)

Experiment 2 — Generation Strategy Comparison
    Measures effect of prompt strategy on report quality.
    Fixed config: E (Full System)
    Strategies: zero_shot, few_shot, citation, reflection

Config consolidation rationale (vs original C0-C6):
    - Filter + Hybrid bundled into one step (B): they are always co-deployed;
      testing filter alone (original C1) without hybrid reveals nothing useful.
    - Duplicate full-system config (original C6) dropped.
    Result: 7 configs → 5 meaningful milestones.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    label: str               # Human-readable short name
    use_metadata_filter: bool
    use_hybrid: bool         # False → BM25-only; True → BM25 + semantic hybrid
    use_time_weighting: bool
    use_reranker: bool
    use_graph_rag: bool


# ── Experiment 1: Pipeline configs (A → E) ───────────────────────────────────

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
        label               = "Retrieval Stack",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = False,
        use_reranker        = False,
        use_graph_rag       = False,
    ),
    "C": AblationConfig(
        label               = "+ Temporal",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = True,
        use_reranker        = False,
        use_graph_rag       = False,
    ),
    "D": AblationConfig(
        label               = "+ Reranker",
        use_metadata_filter = True,
        use_hybrid          = True,
        use_time_weighting  = True,
        use_reranker        = True,
        use_graph_rag       = False,
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

# ── Experiment 2: Generation strategies ──────────────────────────────────────

GENERATION_STRATEGIES: list[str] = [
    "zero_shot",   # No examples, no special rules — generation baseline
    "few_shot",    # One worked example prepended — tests structure guidance
    "citation",    # Mandatory [Article N] inline citations — tests grounding discipline
    "reflection",  # Two-pass: draft → self-critique → revised — tests self-correction
]

# Fixed strategy for Experiment 1
EXP1_FIXED_STRATEGY: str = "citation"

# Fixed config for Experiment 2 (always Full System)
EXP2_FIXED_CONFIG: str = "E"

# ── QA pairs (shared across both experiments) ─────────────────────────────────
# 10 queries covering: 2022 Russia-Ukraine crisis, 2026 Hormuz crisis,
# low-risk baseline, Crude Oil / LNG / Diesel commodities,
# single-event and multi-event queries.

SELECTED_QA_IDS: list[int] = [1, 3, 6, 9, 11, 17, 19, 33, 38, 47]
