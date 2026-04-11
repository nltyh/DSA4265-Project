"""
evaluate_retrieval.py
---------------------
Evaluates retrieval quality across 5 variants:
    A: BM25-only
    B: Semantic-only
    C: Hybrid (BM25 + Semantic)
    D: Hybrid + Time Weighting
    E: Hybrid + Time Weighting + Reranker

For each query in qa_pairs.json, retrieves top-K documents per variant,
then uses an LLM judge to score each document's relevance (0 or 1).

Metrics computed per variant:
    - Precision@K  : fraction of top-K docs that are relevant
    - Recall@K     : estimated recall (relevant retrieved / total relevant estimated)
    - MRR          : Mean Reciprocal Rank (rank of first relevant doc)

Usage:
    cd DSA4265-Project/
    python evaluation/evaluate_retrieval.py

Output:
    evaluation/retrieval_outputs/
        results.json
        summary.csv
        summary_plot.png
"""

import os
import sys
import json
import getpass
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

from rank_bm25 import BM25Okapi
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from time_weighter import TimeWeighter
from metadata_filtering import QueryProcessor, MetadataFilter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE   = BASE_DIR.parent / "data" / "final_df.csv"
QA_FILE     = BASE_DIR / "qa_pairs.json"
OUTPUT_DIR  = BASE_DIR / "retrieval_outputs"

TOP_K           = 10
JUDGE_MODEL     = "gpt-4o-mini"
SELECTED_QA_IDS = [1, 3, 6, 9, 11, 17, 19, 33, 38, 47]

VARIANTS = ["bm25", "semantic", "hybrid", "hybrid_time", "hybrid_time_rerank"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus(file_path: Path):
    import pandas as pd
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


def load_qa_pairs(file_path: Path, ids: list) -> list:
    with open(file_path, encoding="utf-8") as f:
        all_pairs = json.load(f)
    return [p for p in all_pairs if p["id"] in ids]


# ── Retrieval variants ────────────────────────────────────────────────────────

def retrieve(variant: str, query: str, documents: list, metadata: list,
             doc_to_idx: dict, top_k: int) -> list[dict]:
    """
    Returns list of dicts: [{text, score, date, commodity}, ...]
    """
    qp     = QueryProcessor()
    parsed = qp.parse_query(query)

    mf           = MetadataFilter(
        target_date = parsed["date"],
        window_days = parsed["window_days"],
        commodity   = parsed["commodity"],
    )
    pool_docs, pool_meta = mf.apply(documents, metadata)
    if not pool_docs:
        pool_docs, pool_meta = documents, metadata

    if variant == "bm25":
        tokenized = [d.lower().split() for d in pool_docs]
        bm25      = BM25Okapi(tokenized)
        scores    = bm25.get_scores(query.lower().split())
        top_idx   = np.argsort(scores)[::-1][:top_k]
        results   = [{"text": pool_docs[i], "score": float(scores[i])} for i in top_idx]

    elif variant == "semantic":
        retriever = HybridRetriever(pool_docs)
        raw       = retriever.semantic_search(query)
        top_idx   = np.argsort(raw)[::-1][:top_k]
        results   = [{"text": pool_docs[i], "score": float(raw[i])} for i in top_idx]

    elif variant in ("hybrid", "hybrid_time", "hybrid_time_rerank"):
        retriever = HybridRetriever(pool_docs)
        raw       = retriever.hybrid_search(query, top_k=top_k, alpha=0.7)
        results   = [{"text": doc, "score": float(sc)} for doc, sc in raw]

        # attach date early so time weighter can use it
        for r in results:
            idx = doc_to_idx.get(r["text"])
            if idx is not None:
                r["date"]      = metadata[idx]["date"]
                r["commodity"] = metadata[idx]["commodity"]

        if variant in ("hybrid_time", "hybrid_time_rerank"):
            tw      = TimeWeighter(decay_rate=0.03)
            results = tw.apply(results, reference_date=parsed["date"])

        if variant == "hybrid_time_rerank" and results:
            reranker = Reranker()
            texts    = [d["text"] for d in results]
            pairs    = [(query, t) for t in texts]
            scores   = reranker.model.predict(pairs)
            reranked = []
            for i, sc in enumerate(scores):
                item = results[i].copy()
                item["score"] = float(sc)
                reranked.append(item)
            reranked.sort(key=lambda x: x["score"], reverse=True)
            results = reranked

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Attach metadata (for variants that haven't done it yet)
    for r in results:
        if "date" not in r:
            idx = doc_to_idx.get(r["text"])
            if idx is not None:
                r["date"]      = metadata[idx]["date"]
                r["commodity"] = metadata[idx]["commodity"]

    return results


# ── LLM relevance judge ───────────────────────────────────────────────────────

RELEVANCE_SYSTEM = """\
You are a strict relevance assessor for a commodity news retrieval system.
Given a QUERY, an EXPECTED ANSWER summary, and a RETRIEVED DOCUMENT,
decide whether the document is relevant to answering the query.

A document is RELEVANT if it contains information that directly supports
or contributes to answering the query (matching commodity, time period, event).

Respond with ONLY a JSON object, nothing else:
{"relevant": true or false, "reason": "<one sentence explaining your decision>"}"""


def judge_relevance(query: str, expected_answer, doc_text: str,
                    client: OpenAI, model: str = JUDGE_MODEL) -> dict:
    """Returns {"relevant": int (0 or 1), "reason": str}."""
    if isinstance(expected_answer, dict):
        expected_str = (
            f"Sentiment: {expected_answer.get('sentiment', '')}\n"
            + "\n".join(
                f"Key Event: {kn.get('event', '')} — {kn.get('impact', '')}"
                for kn in expected_answer.get("key_news", [])
            )
            + f"\nTakeaway: {expected_answer.get('takeaway', '')}"
        )
    else:
        expected_str = str(expected_answer)
    prompt = (
        f"QUERY:\n{query}\n\n"
        f"EXPECTED ANSWER SUMMARY:\n{expected_str}\n\n"
        f"RETRIEVED DOCUMENT:\n{doc_text.strip()[:800]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=80,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw     = resp.choices[0].message.content.strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(cleaned)
        return {
            "relevant": 1 if parsed.get("relevant") else 0,
            "reason"  : parsed.get("reason", ""),
        }
    except Exception as e:
        print(f"    relevance judge error: {e}")
        return {"relevant": 0, "reason": f"error: {e}"}


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(relevance: list[int]) -> float:
    return sum(relevance) / len(relevance) if relevance else 0.0


def recall_at_k(relevance: list[int], total_relevant: int) -> float:
    """Estimated recall — total_relevant estimated as sum across all variants."""
    if total_relevant == 0:
        return 0.0
    return sum(relevance) / total_relevant


def mean_reciprocal_rank(relevance: list[int]) -> float:
    for i, r in enumerate(relevance, 1):
        if r == 1:
            return 1.0 / i
    return 0.0


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(corpus, qa_pairs: list, client: OpenAI) -> dict:
    """
    Returns nested dict:
        {variant: {qid: {precision_at_k, recall_at_k, mrr, relevance_list}}}
    """
    df, documents, metadata, doc_to_idx = corpus
    results = {v: {} for v in VARIANTS}

    for qa in qa_pairs:
        qid   = qa["id"]
        query = qa["query"]
        exp   = qa["expected_answer"]

        print(f"\n── Q{qid}: {query[:70]}...")

        # Collect top-K docs per variant
        variant_docs = {}
        for variant in VARIANTS:
            print(f"   Retrieving [{variant}]...")
            docs = retrieve(variant, query, documents, metadata, doc_to_idx, TOP_K)
            variant_docs[variant] = docs

        # Estimate total relevant = unique relevant docs across all variants
        all_texts = list({d["text"] for v_docs in variant_docs.values() for d in v_docs})
        print(f"   Judging {len(all_texts)} unique docs for relevance...")

        relevance_cache = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_text = {
                executor.submit(judge_relevance, query, exp, text, client): text
                for text in all_texts
            }
            for future in as_completed(future_to_text):
                text = future_to_text[future]
                relevance_cache[text] = future.result()

        total_relevant = max(sum(r["relevant"] for r in relevance_cache.values()), 1)

        # Compute metrics per variant
        for variant in VARIANTS:
            rel_list    = [relevance_cache.get(d["text"], {"relevant": 0})["relevant"] for d in variant_docs[variant]]
            reason_list = [relevance_cache.get(d["text"], {"reason": ""})["reason"]   for d in variant_docs[variant]]
            results[variant][str(qid)] = {
                "precision_at_k": round(precision_at_k(rel_list), 4),
                "recall_at_k"   : round(recall_at_k(rel_list, total_relevant), 4),
                "mrr"           : round(mean_reciprocal_rank(rel_list), 4),
                "relevance_list": rel_list,
                "reasons"       : reason_list,
            }
            print(
                f"   [{variant}] P@{TOP_K}={results[variant][str(qid)]['precision_at_k']:.3f} "
                f"R@{TOP_K}={results[variant][str(qid)]['recall_at_k']:.3f} "
                f"MRR={results[variant][str(qid)]['mrr']:.3f}"
            )

    return results


# ── Aggregation & output ──────────────────────────────────────────────────────

def build_summary(results: dict) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        qid_dict = results[variant]
        p_scores  = [m["precision_at_k"] for m in qid_dict.values()]
        r_scores  = [m["recall_at_k"]    for m in qid_dict.values()]
        mrr_scores = [m["mrr"]           for m in qid_dict.values()]
        rows.append({
            "variant"       : variant,
            "mean_precision": round(float(np.mean(p_scores)),   4) if p_scores  else None,
            "mean_recall"   : round(float(np.mean(r_scores)),   4) if r_scores  else None,
            "mean_mrr"      : round(float(np.mean(mrr_scores)), 4) if mrr_scores else None,
            "n"             : len(p_scores),
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    print(f"\n{'─'*62}")
    print(f"  {'Variant':<18} {'P@K':>8} {'R@K':>8} {'MRR':>8} {'N':>5}")
    print(f"{'─'*62}")
    for _, row in df.iterrows():
        print(
            f"  {row['variant']:<18} "
            f"{row['mean_precision']:>8.4f} "
            f"{row['mean_recall']:>8.4f} "
            f"{row['mean_mrr']:>8.4f} "
            f"{int(row['n']):>5}"
        )
    print(f"{'─'*62}")


def plot_summary(df: pd.DataFrame, out_dir: Path):
    metrics = ["mean_precision", "mean_recall", "mean_mrr"]
    labels  = [f"P@{TOP_K}", f"R@{TOP_K}", "MRR"]
    x       = np.arange(len(VARIANTS))
    width   = 0.25
    colors  = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = df[metric].tolist()
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(VARIANTS, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Retrieval Variant", fontsize=11)
    ax.set_title(
        f"Retrieval Evaluation — P@{TOP_K}, R@{TOP_K}, MRR\n"
        f"(LLM-judged relevance, {len(SELECTED_QA_IDS)} queries)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = out_dir / "summary_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Retrieval Evaluation")
    print(f"  Variants : {VARIANTS}")
    print("    bm25               : BM25-only")
    print("    semantic           : Semantic-only")
    print("    hybrid             : BM25 + Semantic")
    print("    hybrid_time        : Hybrid + Time Weighting")
    print("    hybrid_time_rerank : Hybrid + Time Weighting + Reranker")
    print(f"  Top-K    : {TOP_K}")
    print(f"  Queries  : {SELECTED_QA_IDS}")
    print(f"  Judge    : {JUDGE_MODEL}")
    print("=" * 60)

    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ").strip()
    if not api_key:
        print("❌ No API key provided. Exiting.")
        raise SystemExit(1)
    print("✅ API key accepted.\n")

    client = OpenAI(api_key=api_key)

    print("📂 Loading corpus...")
    corpus = load_corpus(DATA_FILE)
    print(f"   {len(corpus[1]):,} documents loaded.")

    print(f"📋 Loading QA pairs (IDs: {SELECTED_QA_IDS})...")
    qa_pairs = load_qa_pairs(QA_FILE, SELECTED_QA_IDS)
    print(f"   {len(qa_pairs)} pairs loaded.\n")

    results = evaluate(corpus, qa_pairs, client)

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Results saved → {results_path}")

    df = build_summary(results)
    csv_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Summary saved → {csv_path}")

    print_summary(df)
    plot_summary(df, OUTPUT_DIR)

    print("\n🎉 Done!")
