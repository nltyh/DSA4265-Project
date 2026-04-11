"""
ablation_runner.py
------------------
Two-experiment RAG evaluation study.

Experiment 1 — Pipeline Ablation
    Question : Do the advanced retrieval components improve report quality?
    Variable : Pipeline config (A → E), 5 variants
    Fixed    : Generation strategy = 'citation'
    Metric   : LLM Judge score (gpt-4o, 1–5)
    Runs     : 5 configs × 10 queries = 50

Experiment 2 — Generation Strategy Comparison
    Question : Does the prompt style affect report quality?
    Variable : Generation strategy (zero_shot, few_shot, citation, reflection)
    Fixed    : Pipeline = Full System (Config E)
    Metric   : LLM Judge score (gpt-4o, 1–5)
    Runs     : 4 strategies × 10 queries = 40

Total: 90 runs · ~25 min · ~$2.80

Usage
    cd RAG/
    python ablation_runner.py

Reports are cached to disk; re-running skips generation and re-evaluates only.

Output structure
    ablation_outputs/
        exp1_pipeline/
            raw_reports/{config}_citation_q{id}.txt
            results.json
            summary.csv
            summary_plot.png
        exp2_strategy/
            raw_reports/{strategy}_q{id}.txt
            results.json
            summary.csv
            summary_plot.png
"""

import os
import sys
import json
import time
import getpass
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI

# ── Path setup: allow imports from RAG/ and evaluation/
BASE_DIR = Path(__file__).parent
RAG_DIR  = BASE_DIR.parent / "RAG"
sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(BASE_DIR))

from ablation_configs import (
    PIPELINE_CONFIGS,
    GENERATION_STRATEGIES,
    EXP1_FIXED_STRATEGY,
    EXP2_FIXED_CONFIG,
    SELECTED_QA_IDS,
)
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from time_weighter import TimeWeighter
from graph_rag import GraphRAG
from metadata_filtering import QueryProcessor, MetadataFilter
from generation import Generator, build_context
from evaluate import evaluate_reports

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE  = BASE_DIR / ".." / "data" / "final_df.csv"
QA_FILE    = BASE_DIR / "qa_pairs.json"
OUTPUT_DIR = BASE_DIR / "ablation_outputs"

EXP1_DIR   = OUTPUT_DIR / "exp1_pipeline"
EXP2_DIR   = OUTPUT_DIR / "exp2_strategy"

# ── Model choices ─────────────────────────────────────────────────────────────
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL      = "gpt-4o-mini"

# ── Rate-limit delays ─────────────────────────────────────────────────────────
GENERATION_DELAY = 1.5   # seconds between generation calls
JUDGE_DELAY      = 1.0   # seconds between judge calls


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(file_path: Path):
    df = pd.read_csv(file_path)
    documents, metadata = [], []
    for _, row in df.iterrows():
        text = (
            f"\n        Title: {row['title']}\n"
            f"        Description: {row['description']}\n"
            f"        Commodity: {row['relevant_commodities']}\n"
            f"        Category: {row['news_category']}\n"
            f"        Risk: {row['risk_category']}\n"
            f"        "
        )
        documents.append(text)
        metadata.append({
            "date"     : row["date"],
            "commodity": row["relevant_commodities"],
            "risk"     : row["risk_category"],
            "severity" : row["risk_severity"],
        })
    doc_to_idx = {doc: i for i, doc in enumerate(documents)}
    return df, documents, metadata, doc_to_idx


def load_qa_pairs(file_path: Path, ids: list[int]) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        all_pairs = json.load(f)
    selected = [p for p in all_pairs if p["id"] in ids]
    if len(selected) != len(ids):
        found   = {p["id"] for p in selected}
        missing = set(ids) - found
        print(f"  ⚠  QA IDs not found: {missing}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Configurable pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    documents: list[str],
    metadata: list[dict],
    df: pd.DataFrame,
    doc_to_idx: dict,
    config,           # AblationConfig dataclass
    strategy: str,
    api_key: str,
) -> dict:
    """
    Run the RAG pipeline with components controlled by `config`.
    Returns {"report": str, "context": str}.
    """
    from rank_bm25 import BM25Okapi

    # ── 0. Query parsing (always on) ──────────────────────────────────────────
    qp         = QueryProcessor()
    parsed     = qp.parse_query(query)
    query_date = parsed.get("date")

    # ── 1. Metadata filtering ─────────────────────────────────────────────────
    if config.use_metadata_filter:
        mf = MetadataFilter(
            target_date = query_date,
            window_days = parsed["window_days"],
            commodity   = parsed["commodity"],
        )
        pool_docs, pool_meta = mf.apply(documents, metadata)
        if not pool_docs:           # fallback: never leave pool empty
            pool_docs, pool_meta = documents, metadata
    else:
        pool_docs, pool_meta = documents, metadata

    # ── 2. Retrieval ──────────────────────────────────────────────────────────
    if config.use_hybrid:
        retriever   = HybridRetriever(pool_docs)
        raw_results = retriever.hybrid_search(query, top_k=10, alpha=0.7)
    else:
        tokenized   = [d.lower().split() for d in pool_docs]
        bm25        = BM25Okapi(tokenized)
        scores      = bm25.get_scores(query.lower().split())
        top_idx     = np.argsort(scores)[::-1][:10]
        raw_results = [(pool_docs[i], float(scores[i])) for i in top_idx]

    # ── 3. Attach metadata ────────────────────────────────────────────────────
    results_with_meta = []
    for doc, score in raw_results:
        idx = doc_to_idx.get(doc)
        if idx is None:
            continue
        results_with_meta.append({
            "text"     : doc,
            "score"    : score,
            "date"     : metadata[idx]["date"],
            "commodity": metadata[idx]["commodity"],
        })

    # ── 4. Time weighting ─────────────────────────────────────────────────────
    if config.use_time_weighting:
        tw = TimeWeighter(decay_rate=0.03)
        results_with_meta = tw.apply(results_with_meta, reference_date=query_date)

    # ── 5. Reranking ──────────────────────────────────────────────────────────
    if config.use_reranker and results_with_meta:
        reranker = Reranker()
        texts    = [d["text"] for d in results_with_meta]
        pairs    = [(query, t) for t in texts]
        scores   = reranker.model.predict(pairs)
        reranked = []
        for i, sc in enumerate(scores):
            item = results_with_meta[i].copy()
            item["score"] = float(sc)
            reranked.append(item)
        reranked.sort(key=lambda x: x["score"], reverse=True)
    else:
        reranked = results_with_meta

    # ── 6. GraphRAG ───────────────────────────────────────────────────────────
    if config.use_graph_rag and reranked:
        graph = GraphRAG()
        docs_for_graph = []
        for d in reranked:
            idx = doc_to_idx.get(d["text"])
            if idx is None:
                continue
            entities = [
                metadata[idx]["commodity"],
                df.iloc[idx]["news_category"],
                df.iloc[idx]["risk_category"],
            ]
            docs_for_graph.append({"text": d["text"], "entities": entities})
        graph.build_graph(docs_for_graph)

        query_entities = []
        if parsed["commodity"]:
            query_entities.append(parsed["commodity"])
        query_entities.extend(query.lower().split())
        subgraph = graph.retrieve_subgraph(query_entities)
    else:
        subgraph = []

    # ── 7. Generation ─────────────────────────────────────────────────────────
    gen    = Generator(strategy=strategy, api_key=api_key, model=GENERATION_MODEL)
    report = gen.generate(query, reranked, graph_context=subgraph)
    ctx    = build_context(reranked[: gen.top_k_docs])

    return {"report": report, "context": ctx}


# ─────────────────────────────────────────────────────────────────────────────
# Core run loop (shared by both experiments)
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(
    runs: list[dict],          # each: {label, config, strategy, qa, cache_path}
    corpus: tuple,
    client: OpenAI,
    api_key: str,
    results_path: Path,
) -> dict:
    """
    Iterate over all (config, strategy, query) combos.
    Generates (or loads from cache) each report, then evaluates with LLM Judge.
    Checkpoints results.json after every query.
    Returns full results dict keyed by label → qid → metrics.
    """
    df, documents, metadata, doc_to_idx = corpus

    results: dict = {}
    total = len(runs)

    for i, run in enumerate(runs, 1):
        label      = run["label"]
        config     = run["config"]
        strategy   = run["strategy"]
        qa         = run["qa"]
        cache_path = run["cache_path"]

        qid   = qa["id"]
        query = qa["query"]

        results.setdefault(label, {})

        # ── Generate or load from cache ───────────────────────────────────────
        if cache_path.exists():
            report = cache_path.read_text(encoding="utf-8")
            print(f"  ✅ [{i:>3}/{total}] [cache] {label} | Q{qid}")
            # Still need context for the judge — re-run retrieval (no LLM call)
            try:
                out     = run_pipeline(query, documents, metadata, df, doc_to_idx,
                                       config, strategy, api_key)
                context = out["context"]
            except Exception as e:
                print(f"     ⚠  Context re-build failed: {e}. Using empty context.")
                context = ""
        else:
            print(f"\n  🔄 [{i:>3}/{total}] {label} | Q{qid}")
            print(f"     {textwrap.shorten(query, width=75)}")
            try:
                out     = run_pipeline(query, documents, metadata, df, doc_to_idx,
                                       config, strategy, api_key)
                report  = out["report"]
                context = out["context"]
                cache_path.write_text(report, encoding="utf-8")
                print(f"     💾 Cached → {cache_path.name}")
                time.sleep(GENERATION_DELAY)
            except Exception as e:
                print(f"     ❌ Generation failed: {e}")
                results[label][str(qid)] = {"llm_judge": None, "error": str(e)}
                _checkpoint(results, results_path)
                continue

        # ── LLM Judge evaluation ──────────────────────────────────────────────
        try:
            eval_out = evaluate_reports(
                query             = query,
                reports           = {strategy: report},
                context           = context,
                reference         = None,
                client            = client,
                #groundedness_model= "gpt-4o-mini",   # unused (run_groundedness=False)
                judge_model       = JUDGE_MODEL,
                run_groundedness  = False,
                run_llm_judge     = True,
            )
            results[label][str(qid)] = eval_out[strategy]
            score  = eval_out[strategy]["llm_judge"]["score"]
            reason = eval_out[strategy]["llm_judge"]["reason"]
            print(f"     ⚖  Judge: {score}/5 — {reason}")
            time.sleep(JUDGE_DELAY)
        except Exception as e:
            print(f"     ❌ Judge failed: {e}")
            results[label][str(qid)] = {"llm_judge": None, "error": str(e)}

        _checkpoint(results, results_path)

    return results


def _checkpoint(results: dict, path: Path):
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Pipeline Ablation
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1(corpus, qa_pairs, client, api_key):
    print("\n" + "═" * 60)
    print("  EXPERIMENT 1 — Pipeline Ablation")
    print(f"  Strategy fixed : {EXP1_FIXED_STRATEGY}")
    print(f"  Configs        : {list(PIPELINE_CONFIGS.keys())}")
    print(f"  Queries        : {len(qa_pairs)}")
    print(f"  Total runs     : {len(PIPELINE_CONFIGS) * len(qa_pairs)}")
    print("═" * 60)

    report_dir   = EXP1_DIR / "raw_reports"
    results_path = EXP1_DIR / "results.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for cfg_key, config in PIPELINE_CONFIGS.items():
        for qa in qa_pairs:
            runs.append({
                "label"     : cfg_key,
                "config"    : config,
                "strategy"  : EXP1_FIXED_STRATEGY,
                "qa"        : qa,
                "cache_path": report_dir / f"{cfg_key}_{EXP1_FIXED_STRATEGY}_q{qa['id']}.txt",
            })

    results = run_loop(runs, corpus, client, api_key, results_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Generation Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2(corpus, qa_pairs, client, api_key):
    full_config = PIPELINE_CONFIGS[EXP2_FIXED_CONFIG]

    print("\n" + "═" * 60)
    print("  EXPERIMENT 2 — Generation Strategy Comparison")
    print(f"  Pipeline fixed : Config {EXP2_FIXED_CONFIG} ({full_config.label})")
    print(f"  Strategies     : {GENERATION_STRATEGIES}")
    print(f"  Queries        : {len(qa_pairs)}")
    print(f"  Total runs     : {len(GENERATION_STRATEGIES) * len(qa_pairs)}")
    print("═" * 60)

    report_dir   = EXP2_DIR / "raw_reports"
    results_path = EXP2_DIR / "results.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for strategy in GENERATION_STRATEGIES:
        for qa in qa_pairs:
            runs.append({
                "label"     : strategy,
                "config"    : full_config,
                "strategy"  : strategy,
                "qa"        : qa,
                "cache_path": report_dir / f"{strategy}_q{qa['id']}.txt",
            })

    results = run_loop(runs, corpus, client, api_key, results_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation, CSV, and plots
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(results: dict, label_map: dict) -> pd.DataFrame:
    """
    Aggregate LLM Judge scores into a DataFrame.
    label_map: {key: display_label} — e.g. {"A": "A: Baseline"}
    """
    rows = []
    for key, qid_dict in results.items():
        scores = []
        for qid, metrics in qid_dict.items():
            lj = metrics.get("llm_judge")
            if lj and lj.get("score") is not None:
                scores.append(float(lj["score"]))
        rows.append({
            "key"        : key,
            "label"      : label_map.get(key, key),
            "mean_score" : float(np.mean(scores)) if scores else None,
            "std_score"  : float(np.std(scores))  if scores else None,
            "n"          : len(scores),
        })
    return pd.DataFrame(rows)


def save_summary(df: pd.DataFrame, out_dir: Path, filename: str = "summary.csv"):
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"  📄 Summary saved → {path}")


def plot_experiment(df: pd.DataFrame, out_dir: Path, title: str, x_label: str):
    """Single bar chart with error bars for one experiment."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    keys   = df["key"].tolist()
    labels = df["label"].tolist()
    means  = df["mean_score"].tolist()
    stds   = df["std_score"].fillna(0).tolist()

    colors = [
        "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
        "#64B5CD", "#777777",
    ]

    x = np.arange(len(keys))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=colors[: len(keys)], alpha=0.85, edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        if mean is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.05,
                f"{mean:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 6.0)
    ax.set_ylabel("Mean LLM Judge Score (1–5)", fontsize=11)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axhline(y=3, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = out_dir / "summary_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved → {path}")


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")
    print(f"  {'Key':<18} {'Mean Score':>11} {'Std':>7} {'N':>5}")
    print(f"{'─'*52}")
    for _, row in df.iterrows():
        mean = f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "   —  "
        std  = f"{row['std_score']:.3f}"  if row["std_score"]  is not None else "   —  "
        print(f"  {row['label']:<18} {mean:>11} {std:>7} {int(row['n']):>5}")
    print(f"{'─'*52}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Pre-flight summary ────────────────────────────────────────────────────
    n_exp1 = len(PIPELINE_CONFIGS) * len(SELECTED_QA_IDS)
    n_exp2 = len(GENERATION_STRATEGIES) * len(SELECTED_QA_IDS)

    print("=" * 60)
    print("  RAG Evaluation Runner — 2 Experiments")
    print("=" * 60)
    print(f"  Exp 1 (Pipeline Ablation)   : {n_exp1} runs  ~{n_exp1//4} min  ~${n_exp1*0.031:.2f}")
    print(f"  Exp 2 (Strategy Comparison) : {n_exp2} runs  ~{n_exp2//4} min  ~${n_exp2*0.031:.2f}")
    print(f"  Total                       : {n_exp1+n_exp2} runs  ~{(n_exp1+n_exp2)//4} min  ~${(n_exp1+n_exp2)*0.031:.2f}")
    print(f"  Generation model : {GENERATION_MODEL}")
    print(f"  Judge model      : {JUDGE_MODEL}")
    print(f"  Output dir       : {OUTPUT_DIR.resolve()}")
    print("=" * 60)
    print()

    # ── API key prompt (hidden input, never logged) ───────────────────────────
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ").strip()
    if not api_key:
        print("❌ No API key provided. Exiting.")
        sys.exit(1)
    print("✅ API key accepted.\n")

    client = OpenAI(api_key=api_key)

    # ── Load data (once, shared by both experiments) ──────────────────────────
    print("📂 Loading corpus...")
    corpus = load_corpus(DATA_FILE)
    print(f"   {len(corpus[1]):,} documents loaded.")

    print(f"📋 Loading QA pairs (IDs: {SELECTED_QA_IDS})...")
    qa_pairs = load_qa_pairs(QA_FILE, SELECTED_QA_IDS)
    print(f"   {len(qa_pairs)} pairs loaded.\n")

    t0 = time.time()

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    exp1_results = experiment_1(corpus, qa_pairs, client, api_key)

    exp1_label_map = {
        k: f"{k}: {v.label}" for k, v in PIPELINE_CONFIGS.items()
    }
    exp1_df = build_summary(exp1_results, exp1_label_map)
    save_summary(exp1_df, EXP1_DIR)
    plot_experiment(
        exp1_df, EXP1_DIR,
        title   = "Experiment 1 — Pipeline Ablation\n(LLM Judge score, higher = better)",
        x_label = "Pipeline Configuration",
    )
    print_table(exp1_df, "Experiment 1 — Pipeline Ablation Results")

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    exp2_results = experiment_2(corpus, qa_pairs, client, api_key)

    exp2_label_map = {s: s.replace("_", " ").title() for s in GENERATION_STRATEGIES}
    exp2_df = build_summary(exp2_results, exp2_label_map)
    save_summary(exp2_df, EXP2_DIR)
    plot_experiment(
        exp2_df, EXP2_DIR,
        title   = "Experiment 2 — Generation Strategy Comparison\n(LLM Judge score, higher = better)",
        x_label = "Generation Strategy",
    )
    print_table(exp2_df, "Experiment 2 — Generation Strategy Results")

    elapsed = time.time() - t0
    print(f"\n⏱  Total wall-clock time: {elapsed/60:.1f} min")
    print("🎉 All done! Results saved to ablation_outputs/")
