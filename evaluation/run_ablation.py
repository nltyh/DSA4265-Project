"""
run_ablation.py
---------------
Unified three-experiment RAG ablation study.

Experiment 0 — Retrieval Quality  (same 5 architectures A-E)
    Metrics : Precision@K, Recall@K, MRR, MAP
    Method  : LLM-judged relevance per retrieved doc (concurrent judging)
    Runs    : 5 configs × 50 queries × (up to 10 unique docs) ~ 2500 judge calls

Experiment 1 — Pipeline Generation Ablation
    Metric  : LLM Judge score (1-5)
    Fixed   : citation generation strategy
    Runs    : 5 configs × 50 queries = 250 generations + 250 judge calls

Experiment 2 — Generation Strategy Comparison
    Metric  : LLM Judge score (1-5)
    Fixed   : best config = argmax(mean retrieval score + mean generation score)
    Runs    : 4 strategies × 50 queries = 200 generations + 200 judge calls
              (citation reports for best config reused from Exp 1 cache)

Usage
    cd DSA4265-Project/
    python evaluation/run_ablation.py

Output
    evaluation/ablation_outputs/
        exp0_retrieval/    results.json  summary.csv  summary_plot.png
        exp1_pipeline/     results.json  summary.csv  summary_plot.png  raw_reports/
        exp2_strategy/     results.json  summary.csv  summary_plot.png  raw_reports/
        best_config_selection.json
"""

import sys
import json
import time
import getpass
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAG_DIR  = BASE_DIR.parent / "RAG"
sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(BASE_DIR))

from rank_bm25 import BM25Okapi
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from time_weighter import TimeWeighter
from graph_rag import GraphRAG
from metadata_filtering import QueryProcessor, MetadataFilter
from generation import Generator, build_context
from evaluate import evaluate_reports

from ablation_configs import (
    PIPELINE_CONFIGS,
    GENERATION_STRATEGIES,
    EXP1_FIXED_STRATEGY,
    SELECTED_QA_IDS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE  = BASE_DIR.parent / "data" / "final_df.csv"
QA_FILE    = BASE_DIR / "qa_pairs.json"
OUTPUT_DIR = BASE_DIR / "ablation_outputs"

EXP0_DIR = OUTPUT_DIR / "exp0_retrieval"
EXP1_DIR = OUTPUT_DIR / "exp1_pipeline"
EXP2_DIR = OUTPUT_DIR / "exp2_strategy"

# ── Model config ──────────────────────────────────────────────────────────────
RETRIEVAL_JUDGE_MODEL = "gpt-4o-mini"
GENERATION_MODEL      = "gpt-4o-mini"
JUDGE_MODEL           = "gpt-4o-mini"

# ── Concurrency ───────────────────────────────────────────────────────────────
RETRIEVAL_WORKERS  = 8    # parallel LLM relevance judges per query
JUDGE_WORKERS      = 5    # parallel LLM judge calls per config/strategy batch
TOP_K              = 10

# ── Rate limits ───────────────────────────────────────────────────────────────
GENERATION_DELAY   = 1.5  # seconds between generation calls


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
        missing = set(ids) - {p["id"] for p in selected}
        print(f"  Warning: QA IDs not found: {missing}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Shared: configurable retrieval (no generation)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_docs(
    query: str,
    documents: list[str],
    metadata: list[dict],
    doc_to_idx: dict,
    config,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run retrieval stages for a given config. Returns list of
    {text, score, date, commodity} dicts, sorted by score desc.
    """
    qp         = QueryProcessor()
    parsed     = qp.parse_query(query)
    query_date = parsed.get("date")

    # ── Filter ────────────────────────────────────────────────────────────────
    if config.use_metadata_filter:
        mf = MetadataFilter(
            target_date = query_date,
            window_days = parsed.get("window_days"),
            commodity   = parsed.get("commodity"),
            start_date  = parsed.get("start_date"),
            end_date    = parsed.get("end_date"),
        )
        pool_docs, pool_meta = mf.apply(documents, metadata)
        # Fall back to full corpus ONLY if the filter screens out everything.
        # If we have 1-9 highly relevant docs, we should keep them isolated
        # rather than diluting the pool with 10,000 unfiltered docs.
        if not pool_docs:
            pool_docs, pool_meta = documents, metadata
    else:
        pool_docs, pool_meta = documents, metadata

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if config.use_hybrid:
        retriever   = HybridRetriever(pool_docs)
        raw_results = retriever.hybrid_search(query, top_k=top_k, alpha=0.7)
        results     = [{"text": doc, "score": float(sc)} for doc, sc in raw_results]
    else:
        tokenized   = [d.lower().split() for d in pool_docs]
        bm25        = BM25Okapi(tokenized)
        scores      = bm25.get_scores(query.lower().split())
        top_idx     = np.argsort(scores)[::-1][:top_k]
        results     = [{"text": pool_docs[i], "score": float(scores[i])} for i in top_idx]

    # Attach metadata
    for r in results:
        idx = doc_to_idx.get(r["text"])
        if idx is not None:
            r["date"]      = metadata[idx]["date"]
            r["commodity"] = metadata[idx]["commodity"]

    # ── Time weighting ────────────────────────────────────────────────────────
    if config.use_time_weighting:
        tw      = TimeWeighter(decay_rate=0.03)
        results = tw.apply(results, reference_date=query_date)

    # ── Reranking ─────────────────────────────────────────────────────────────
    if config.use_reranker and results:
        reranker = Reranker()
        texts    = [d["text"] for d in results]
        pairs    = [(query, t) for t in texts]
        r_scores = reranker.model.predict(pairs)
        reranked = []
        for i, sc in enumerate(r_scores):
            item = results[i].copy()
            item["score"] = float(sc)
            reranked.append(item)
        reranked.sort(key=lambda x: x["score"], reverse=True)
        results = reranked

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 0 — Retrieval Quality
# ─────────────────────────────────────────────────────────────────────────────

RELEVANCE_SYSTEM = """\
You are a strict relevance assessor for a commodity news retrieval system.
Given a QUERY, an EXPECTED ANSWER summary, and a RETRIEVED DOCUMENT,
decide whether the document is relevant to answering the query.

A document is RELEVANT if it contains information that directly supports
or contributes to answering the query (matching commodity, time period, event).

Respond with ONLY a JSON object, nothing else:
{"relevant": true or false, "reason": "<one sentence>"}"""


def _judge_relevance_single(query: str, expected, doc_text: str, client: OpenAI) -> dict:
    if isinstance(expected, dict):
        exp_str = (
            f"Sentiment: {expected.get('sentiment', '')}\n"
            + "\n".join(
                f"Key Event: {kn.get('event','')} - {kn.get('impact','')}"
                for kn in expected.get("key_news", [])
            )
            + f"\nTakeaway: {expected.get('takeaway', '')}"
        )
    else:
        exp_str = str(expected)

    prompt = (
        f"QUERY:\n{query}\n\n"
        f"EXPECTED ANSWER SUMMARY:\n{exp_str}\n\n"
        f"RETRIEVED DOCUMENT:\n{doc_text.strip()[:800]}"
    )
    try:
        resp = client.chat.completions.create(
            model=RETRIEVAL_JUDGE_MODEL,
            max_tokens=80,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw     = resp.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(cleaned)
        return {"relevant": 1 if parsed.get("relevant") else 0, "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"relevant": 0, "reason": f"error: {e}"}


def _map_score(relevance: list[int]) -> float:
    """Mean Average Precision — area under precision-recall curve."""
    if not any(relevance):
        return 0.0
    hits, running_sum = 0, 0.0
    for i, r in enumerate(relevance, 1):
        if r == 1:
            hits += 1
            running_sum += hits / i
    total_relevant = sum(relevance)
    return running_sum / total_relevant if total_relevant > 0 else 0.0


def _mrr_score(relevance: list[int]) -> float:
    for i, r in enumerate(relevance, 1):
        if r == 1:
            return 1.0 / i
    return 0.0


def run_exp0_retrieval(corpus, qa_pairs, client) -> dict:
    """
    Experiment 0: evaluate retrieval quality of all 5 configs.
    Returns {config_key: {qid: {precision_at_k, recall_at_k, mrr, map}}}
    """
    df, documents, metadata, doc_to_idx = corpus
    results = {k: {} for k in PIPELINE_CONFIGS}

    EXP0_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EXP0_DIR / "results.json"

    print("\n" + "=" * 62)
    print("  EXPERIMENT 0 — Retrieval Quality Evaluation")
    print(f"  Architectures : {list(PIPELINE_CONFIGS.keys())}")
    print(f"  Queries       : {len(qa_pairs)}")
    print(f"  Top-K         : {TOP_K}")
    print(f"  Judge model   : {RETRIEVAL_JUDGE_MODEL}")
    print(f"  Workers       : {RETRIEVAL_WORKERS}")
    print("=" * 62)

    for qi, qa in enumerate(qa_pairs, 1):
        qid   = qa["id"]
        query = qa["query"]
        exp   = qa["expected_answer"]
        print(f"\n  Q{qid} ({qi}/{len(qa_pairs)}): {textwrap.shorten(query, 65)}")

        # Retrieve top-K per config
        config_docs = {}
        for cfg_key, config in PIPELINE_CONFIGS.items():
            config_docs[cfg_key] = retrieve_docs(query, documents, metadata, doc_to_idx, config)

        # Deduplicate docs across all configs for judging
        all_texts = list({d["text"] for docs in config_docs.values() for d in docs})
        print(f"  Judging {len(all_texts)} unique docs (concurrent)...")

        # Parallel LLM relevance judging
        relevance_cache: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=RETRIEVAL_WORKERS) as ex:
            futures = {
                ex.submit(_judge_relevance_single, query, exp, text, client): text
                for text in all_texts
            }
            for future in as_completed(futures):
                relevance_cache[futures[future]] = future.result()

        total_relevant = max(sum(r["relevant"] for r in relevance_cache.values()), 1)

        # Compute metrics per config
        for cfg_key, docs in config_docs.items():
            rel_list = [relevance_cache.get(d["text"], {"relevant": 0})["relevant"] for d in docs]
            k        = len(rel_list)
            p_at_k   = sum(rel_list) / k if k else 0.0
            r_at_k   = sum(rel_list) / total_relevant
            mrr      = _mrr_score(rel_list)
            map_s    = _map_score(rel_list)

            results[cfg_key][str(qid)] = {
                "precision_at_k": round(p_at_k, 4),
                "recall_at_k"   : round(r_at_k, 4),
                "mrr"           : round(mrr,    4),
                "map"           : round(map_s,  4),
                "relevance_list": rel_list,
            }
            print(f"    [{cfg_key}] P@{k}={p_at_k:.3f}  R@{k}={r_at_k:.3f}  MRR={mrr:.3f}  MAP={map_s:.3f}")

        # Checkpoint
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def _build_retrieval_summary(results: dict) -> pd.DataFrame:
    rows = []
    for cfg_key, qid_dict in results.items():
        vals = {"precision_at_k": [], "recall_at_k": [], "mrr": [], "map": []}
        for m in qid_dict.values():
            for k in vals:
                if m.get(k) is not None:
                    vals[k].append(m[k])
        composite = np.mean([
            np.mean(vals["precision_at_k"]),
            np.mean(vals["recall_at_k"]),
            np.mean(vals["mrr"]),
            np.mean(vals["map"]),
        ]) if all(vals.values()) else None
        rows.append({
            "config"        : cfg_key,
            "label"         : f"{cfg_key}: {PIPELINE_CONFIGS[cfg_key].label}",
            "mean_precision": round(float(np.mean(vals["precision_at_k"])), 4) if vals["precision_at_k"] else None,
            "mean_recall"   : round(float(np.mean(vals["recall_at_k"])),    4) if vals["recall_at_k"]    else None,
            "mean_mrr"      : round(float(np.mean(vals["mrr"])),            4) if vals["mrr"]            else None,
            "mean_map"      : round(float(np.mean(vals["map"])),            4) if vals["map"]            else None,
            "composite"     : round(float(composite), 4) if composite is not None else None,
            "n"             : len(vals["precision_at_k"]),
        })
    return pd.DataFrame(rows)


def _plot_retrieval(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("mean_precision", f"P@{TOP_K}", "#4C72B0"),
        ("mean_recall",    f"R@{TOP_K}", "#55A868"),
        ("mean_mrr",       "MRR",        "#C44E52"),
        ("mean_map",       "MAP",        "#8172B2"),
    ]
    labels  = df["label"].tolist()
    x       = np.arange(len(labels))
    width   = 0.20

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, label, color) in enumerate(metrics):
        vals = df[col].tolist()
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Experiment 0 — Retrieval Quality by Architecture\n"
        f"(LLM-judged relevance, {len(df)} configs, {TOP_K} docs/query)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "summary_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Pipeline Generation Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report(query, documents, metadata, df, doc_to_idx, config, strategy, api_key) -> dict:
    """Run full pipeline (retrieval + generation). Returns {report, context, retrieved_docs}."""
    qp         = QueryProcessor()
    parsed     = qp.parse_query(query)
    query_date = parsed.get("date")

    retrieved = retrieve_docs(query, documents, metadata, doc_to_idx, config)

    # GraphRAG
    if config.use_graph_rag and retrieved:
        graph = GraphRAG()
        docs_for_graph = []
        for d in retrieved:
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

        q_entities = []
        if parsed["commodity"]:
            q_entities.append(parsed["commodity"])
        q_entities.extend(query.lower().split())
        subgraph = graph.retrieve_subgraph(q_entities)
    else:
        subgraph = []

    gen    = Generator(strategy=strategy, api_key=api_key, model=GENERATION_MODEL)
    report = gen.generate(query, retrieved, graph_context=subgraph)
    ctx    = build_context(retrieved[: gen.top_k_docs])
    return {"report": report, "context": ctx}


def _llm_judge_single(query: str, report: str, context: str, client: OpenAI) -> dict:
    """Single LLM Judge call — used in thread pool."""
    try:
        result = evaluate_reports(
            query             = query,
            reports           = {"_": report},
            context           = context,
            reference         = None,
            client            = client,
            groundedness_model= "gpt-4o-mini",
            judge_model       = JUDGE_MODEL,
            run_groundedness  = False,
            run_llm_judge     = True,
        )
        return result["_"]["llm_judge"]
    except Exception as e:
        return {"score": None, "reason": str(e)}


def _run_generation_loop(
    runs: list[dict],
    corpus: tuple,
    client: OpenAI,
    api_key: str,
    report_dir: Path,
    results_path: Path,
    exp_label: str,
) -> dict:
    """
    Generic generation + evaluation loop shared by Exp 1 and Exp 2.
    runs: list of {label, config, strategy, qa, cache_path}
    """
    df, documents, metadata, doc_to_idx = corpus
    results: dict = {}
    total = len(runs)

    # ── Phase 1: Generate all reports (sequential, cached) ───────────────────
    print(f"\n  [{exp_label}] Phase 1/2 — Generating reports ({total} total)...")
    contexts: dict = {}   # key: (label, qid) -> context string

    for i, run in enumerate(runs, 1):
        label      = run["label"]
        config     = run["config"]
        strategy   = run["strategy"]
        qa         = run["qa"]
        cache_path = run["cache_path"]
        qid        = qa["id"]
        query      = qa["query"]
        key        = (label, qid)

        results.setdefault(label, {})

        if cache_path.exists() and cache_path.stat().st_size > 0:
            print(f"  [cache] [{i:>3}/{total}] {label} | Q{qid}")
            # Rebuild context (no LLM call, just retrieval)
            try:
                out = _generate_report(query, documents, metadata, df, doc_to_idx,
                                       config, strategy, api_key)
                contexts[key] = out["context"]
                results[label][str(qid)] = {"_report": cache_path.read_text(encoding="utf-8")}
            except Exception as e:
                contexts[key] = ""
                results[label][str(qid)] = {"_report": "", "_error": str(e)}
        else:
            print(f"  [ gen ] [{i:>3}/{total}] {label} | Q{qid}  {textwrap.shorten(query, 50)}")
            try:
                out = _generate_report(query, documents, metadata, df, doc_to_idx,
                                       config, strategy, api_key)
                cache_path.write_text(out["report"], encoding="utf-8")
                contexts[key]                = out["context"]
                results[label][str(qid)]     = {"_report": out["report"]}
                time.sleep(GENERATION_DELAY)
            except Exception as e:
                print(f"    ERROR generating: {e}")
                contexts[key]                = ""
                results[label][str(qid)]     = {"_report": "", "_error": str(e)}

        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Phase 2: Evaluate all reports with LLM Judge (concurrent per batch) ──
    print(f"\n  [{exp_label}] Phase 2/2 — LLM Judge scoring ({total} total, {JUDGE_WORKERS} workers)...")

    def _judge_run(run):
        label  = run["label"]
        qa     = run["qa"]
        qid    = qa["id"]
        query  = qa["query"]
        key    = (label, qid)
        report = results.get(label, {}).get(str(qid), {}).get("_report", "")
        ctx    = contexts.get(key, "")
        if not report:
            return label, str(qid), {"score": None, "reason": "empty report"}
        judge = _llm_judge_single(query, report, ctx, client)
        return label, str(qid), judge

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        futures = {ex.submit(_judge_run, run): run for run in runs}
        done = 0
        for future in as_completed(futures):
            label, qid_str, judge = future.result()
            results[label][qid_str]["llm_judge"] = judge
            done += 1
            score = judge.get("score")
            print(f"  [judge] {done:>3}/{total}  {label} | Q{qid_str}  score={score}")
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Remove internal _report key from final output to keep results.json clean
    for label_dict in results.values():
        for qid_metrics in label_dict.values():
            qid_metrics.pop("_report", None)
            qid_metrics.pop("_error", None)

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp1_pipeline(corpus, qa_pairs, client, api_key) -> dict:
    EXP1_DIR.mkdir(parents=True, exist_ok=True)
    report_dir   = EXP1_DIR / "raw_reports"
    results_path = EXP1_DIR / "results.json"
    report_dir.mkdir(exist_ok=True)

    n_total = len(PIPELINE_CONFIGS) * len(qa_pairs)
    print("\n" + "=" * 62)
    print("  EXPERIMENT 1 — Pipeline Architecture Ablation")
    print(f"  Configs   : {list(PIPELINE_CONFIGS.keys())}")
    print(f"  Strategy  : {EXP1_FIXED_STRATEGY} (fixed)")
    print(f"  Queries   : {len(qa_pairs)}")
    print(f"  Total gen : {n_total}")
    print("=" * 62)

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

    return _run_generation_loop(runs, corpus, client, api_key, report_dir, results_path, "Exp1")


# ─────────────────────────────────────────────────────────────────────────────
# Best config selection
# ─────────────────────────────────────────────────────────────────────────────

def select_best_config(ret_results: dict, gen_results: dict) -> str:
    """
    Compute composite score per config:
        retrieval_score = mean(P@K, R@K, MRR, MAP)  → already in [0,1]
        generation_score = (mean LLM Judge - 1) / 4  → normalise to [0,1]
        combined = (retrieval_score + generation_score) / 2

    Returns the config key with highest combined score.
    """
    scores = {}
    for cfg_key in PIPELINE_CONFIGS:
        # Retrieval score: MAP only
        # MAP accounts for the rank of every relevant doc (not just the first),
        # making it the most comprehensive single retrieval metric.
        map_vals = [
            qid_m["map"]
            for qid_m in ret_results.get(cfg_key, {}).values()
            if qid_m.get("map") is not None
        ]
        ret_score = float(np.mean(map_vals)) if map_vals else 0.0

        # Generation composite (normalise 1-5 → 0-1)
        g_scores = []
        for qid_m in gen_results.get(cfg_key, {}).values():
            lj = qid_m.get("llm_judge")
            if lj and lj.get("score") is not None:
                g_scores.append((float(lj["score"]) - 1.0) / 4.0)
        gen_score = float(np.mean(g_scores)) if g_scores else 0.0

        combined = (ret_score + gen_score) / 2.0
        scores[cfg_key] = {
            "retrieval_score" : round(ret_score, 4),
            "generation_score": round(gen_score, 4),
            "combined_score"  : round(combined,  4),
        }

    # Print selection table
    print("\n" + "-" * 62)
    print("  Best Architecture Selection")
    print(f"  {'Config':<8} {'MAP':>10} {'Generation':>12} {'Combined':>10}")
    print("-" * 62)
    best_key = max(scores, key=lambda k: scores[k]["combined_score"])
    for k, v in scores.items():
        marker = " <-- SELECTED" if k == best_key else ""
        print(f"  {k:<8} {v['retrieval_score']:>10.4f} {v['generation_score']:>12.4f} {v['combined_score']:>10.4f}{marker}")
    print("-" * 62)
    print(f"  Best config: {best_key} ({PIPELINE_CONFIGS[best_key].label})")

    # Save selection
    selection_path = OUTPUT_DIR / "best_config_selection.json"
    selection_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"  Saved -> {selection_path}")

    return best_key


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Generation Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2_strategy(corpus, qa_pairs, client, api_key, best_config_key: str) -> dict:
    best_config = PIPELINE_CONFIGS[best_config_key]
    EXP2_DIR.mkdir(parents=True, exist_ok=True)
    report_dir   = EXP2_DIR / "raw_reports"
    results_path = EXP2_DIR / "results.json"
    report_dir.mkdir(exist_ok=True)

    n_total = len(GENERATION_STRATEGIES) * len(qa_pairs)
    print("\n" + "=" * 62)
    print("  EXPERIMENT 2 — Generation Strategy Comparison")
    print(f"  Best config : {best_config_key} ({best_config.label})")
    print(f"  Strategies  : {GENERATION_STRATEGIES}")
    print(f"  Queries     : {len(qa_pairs)}")
    print(f"  Total gen   : {n_total}  (citation reuses Exp1 cache)")
    print("=" * 62)

    # Symlink/copy citation cache from Exp 1 to avoid re-generating
    exp1_report_dir = EXP1_DIR / "raw_reports"

    runs = []
    for strategy in GENERATION_STRATEGIES:
        for qa in qa_pairs:
            if strategy == EXP1_FIXED_STRATEGY:
                # Reuse cached citation reports from Exp 1
                src = exp1_report_dir / f"{best_config_key}_{strategy}_q{qa['id']}.txt"
                dst = report_dir / f"{strategy}_q{qa['id']}.txt"
                if src.exists() and src.stat().st_size > 0 and not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            runs.append({
                "label"     : strategy,
                "config"    : best_config,
                "strategy"  : strategy,
                "qa"        : qa,
                "cache_path": report_dir / f"{strategy}_q{qa['id']}.txt",
            })

    return _run_generation_loop(runs, corpus, client, api_key, report_dir, results_path, "Exp2")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: aggregation and plotting
# ─────────────────────────────────────────────────────────────────────────────

def build_generation_summary(results: dict, label_map: dict) -> pd.DataFrame:
    rows = []
    for key, qid_dict in results.items():
        scores = []
        for m in qid_dict.values():
            lj = m.get("llm_judge")
            if lj and lj.get("score") is not None:
                scores.append(float(lj["score"]))
        rows.append({
            "key"       : key,
            "label"     : label_map.get(key, key),
            "mean_score": round(float(np.mean(scores)), 4) if scores else None,
            "std_score" : round(float(np.std(scores)),  4) if scores else None,
            "n"         : len(scores),
        })
    return pd.DataFrame(rows)


def plot_generation(df: pd.DataFrame, out_dir: Path, title: str, x_label: str):
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    keys   = df["key"].tolist()
    means  = df["mean_score"].tolist()
    stds   = df["std_score"].fillna(0).tolist()
    x      = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors[:len(keys)], alpha=0.85, edgecolor="white",
                  error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    for bar, mean, std in zip(bars, means, stds):
        if mean is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.05,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 6.0)
    ax.set_ylabel("Mean LLM Judge Score (1-5)", fontsize=11)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axhline(y=3, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "summary_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {path}")


def print_generation_table(df: pd.DataFrame, title: str):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")
    print(f"  {'Label':<22} {'Mean':>8} {'Std':>7} {'N':>5}")
    print(f"{'─'*58}")
    for _, row in df.iterrows():
        mean = f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "  —  "
        std  = f"{row['std_score']:.3f}"  if row["std_score"]  is not None else "  —  "
        print(f"  {row['label']:<22} {mean:>8} {std:>7} {int(row['n']):>5}")
    print(f"{'─'*58}")


def print_retrieval_table(df: pd.DataFrame):
    print(f"\n{'─'*72}")
    print(f"  Experiment 0 — Retrieval Quality Results")
    print(f"{'─'*72}")
    print(f"  {'Label':<30} {'P@K':>7} {'R@K':>7} {'MRR':>7} {'MAP':>7} {'Comp':>7}")
    print(f"{'─'*72}")
    for _, row in df.iterrows():
        print(
            f"  {row['label']:<30} "
            f"{row['mean_precision']:>7.4f} "
            f"{row['mean_recall']:>7.4f} "
            f"{row['mean_mrr']:>7.4f} "
            f"{row['mean_map']:>7.4f} "
            f"{row['composite']:>7.4f}"
        )
    print(f"{'─'*72}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_queries  = len(SELECTED_QA_IDS)
    n_configs  = len(PIPELINE_CONFIGS)
    n_strats   = len(GENERATION_STRATEGIES)
    n_exp0_est = n_configs * n_queries * TOP_K   # upper bound judge calls
    n_exp1     = n_configs * n_queries
    n_exp2     = n_strats  * n_queries

    print("=" * 62)
    print("  RAG Ablation Study — 3 Experiments")
    print("=" * 62)
    print(f"  Architectures     : {list(PIPELINE_CONFIGS.keys())}")
    print(f"  Queries           : {n_queries}")
    print(f"  Exp 0 (Retrieval) : ~{n_exp0_est} relevance judge calls (concurrent)")
    print(f"  Exp 1 (Pipeline)  : {n_exp1} generations + {n_exp1} judge calls")
    print(f"  Exp 2 (Strategy)  : {n_exp2} generations + {n_exp2} judge calls (citation cached)")
    print(f"  Generation model  : {GENERATION_MODEL}")
    print(f"  Judge model       : {JUDGE_MODEL}")
    print(f"  Output dir        : {OUTPUT_DIR.resolve()}")
    print("=" * 62)

    api_key = getpass.getpass("\nEnter your OpenAI API key (input hidden): ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        raise SystemExit(1)
    print("API key accepted.\n")

    client = OpenAI(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    corpus = load_corpus(DATA_FILE)
    print(f"  {len(corpus[1]):,} documents loaded.")

    print(f"Loading QA pairs (all {n_queries})...")
    qa_pairs = load_qa_pairs(QA_FILE, SELECTED_QA_IDS)
    print(f"  {len(qa_pairs)} pairs loaded.")

    t0 = time.time()

    # ── Experiment 0 ─────────────────────────────────────────────────────────
    exp0_results = run_exp0_retrieval(corpus, qa_pairs, client)
    exp0_df      = _build_retrieval_summary(exp0_results)
    exp0_df.to_csv(EXP0_DIR / "summary.csv", index=False)
    print_retrieval_table(exp0_df)
    _plot_retrieval(exp0_df, EXP0_DIR)

    # ── Experiment 1 ─────────────────────────────────────────────────────────
    exp1_results = run_exp1_pipeline(corpus, qa_pairs, client, api_key)
    exp1_label   = {k: f"{k}: {v.label}" for k, v in PIPELINE_CONFIGS.items()}
    exp1_df      = build_generation_summary(exp1_results, exp1_label)
    exp1_df.to_csv(EXP1_DIR / "summary.csv", index=False)
    print_generation_table(exp1_df, "Experiment 1 — Pipeline Ablation Results")
    plot_generation(
        exp1_df, EXP1_DIR,
        title   = "Experiment 1 — Pipeline Ablation\n(LLM Judge score, higher = better)",
        x_label = "Pipeline Architecture",
    )

    # ── Best config selection ─────────────────────────────────────────────────
    best_key = select_best_config(exp0_results, exp1_results)

    # ── Experiment 2 ─────────────────────────────────────────────────────────
    exp2_results = run_exp2_strategy(corpus, qa_pairs, client, api_key, best_key)
    exp2_label   = {s: s.replace("_", " ").title() for s in GENERATION_STRATEGIES}
    exp2_df      = build_generation_summary(exp2_results, exp2_label)
    exp2_df.to_csv(EXP2_DIR / "summary.csv", index=False)
    print_generation_table(exp2_df, "Experiment 2 — Strategy Comparison Results")
    plot_generation(
        exp2_df, EXP2_DIR,
        title   = f"Experiment 2 — Generation Strategy Comparison\n"
                  f"(Best config: {best_key} — {PIPELINE_CONFIGS[best_key].label})",
        x_label = "Generation Strategy",
    )

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock time: {elapsed/60:.1f} min")
    print("Done! Results saved to ablation_outputs/")
