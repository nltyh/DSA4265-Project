import os
import json
import time
import re
import numpy as np
from datetime import datetime
from anthropic import Anthropic

# ── Optional heavy deps (graceful fallback) ─────────────────────────
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠  rouge-score not installed – ROUGE-L will be skipped. pip install rouge-score")

try:
    from bert_score import score as bert_score_fn
    HAS_BERT = True
except ImportError:
    HAS_BERT = False
    print("⚠  bert-score not installed – BERTScore will be skipped. pip install bert-score")


# ────────────────────────────────────────────────────────────────────
# 1.  ROUGE-L
# ────────────────────────────────────────────────────────────────────
def compute_rouge_l(hypothesis: str, reference: str) -> float:
    if not HAS_ROUGE:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, hypothesis)
    return result["rougeL"].fmeasure


# ────────────────────────────────────────────────────────────────────
# 2.  BERTScore
# ────────────────────────────────────────────────────────────────────
def compute_bertscore(hypothesis: str, reference: str) -> float:
    if not HAS_BERT:
        return None
    P, R, F1 = bert_score_fn([hypothesis], [reference], lang="en", verbose=False)
    return F1[0].item()


# ────────────────────────────────────────────────────────────────────
# 3.  Groundedness  (reference-free)
#     Fraction of generated sentences that are entailed by the
#     retrieved context, judged sentence-by-sentence by Claude.
# ────────────────────────────────────────────────────────────────────
GROUNDEDNESS_SYSTEM = """\
You are a strict fact-checking assistant.
You will be given a CONTEXT (retrieved news articles) and a SENTENCE from a generated report.
Your only job: decide whether the SENTENCE is fully supported by the CONTEXT.

Respond with exactly one word: SUPPORTED or UNSUPPORTED.
Do NOT explain. Do NOT add punctuation."""

def compute_groundedness(
    report: str,
    context: str,
    client: Anthropic,
    model: str = "claude-sonnet-4-20250514",
    delay: float = 0.3,
) -> float:
    """
    Split report into sentences and ask Claude whether each is supported
    by the retrieved context.  Returns fraction that are SUPPORTED.
    """
    # Simple sentence split (handles most cases in structured reports)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report) if len(s.strip()) > 20]

    if not sentences:
        return 0.0

    supported = 0
    for sent in sentences:
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=5,
                system=GROUNDEDNESS_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nSENTENCE:\n{sent}"
                }]
            )
            verdict = resp.content[0].text.strip().upper()
            if "SUPPORTED" in verdict:
                supported += 1
            time.sleep(delay)
        except Exception as e:
            print(f"  groundedness error: {e}")

    return supported / len(sentences)


# ────────────────────────────────────────────────────────────────────
# 4.  LLM Judge  (1–5 holistic score)
# ────────────────────────────────────────────────────────────────────
LLM_JUDGE_SYSTEM = """\
You are an expert evaluator of commodity risk reports.
Score the following report on a scale of 1 to 5 using these criteria:

5 – Excellent: fully grounded, no hallucinations, clear structure, actionable insight
4 – Good: mostly grounded, minor gaps, clear and useful
3 – Adequate: some unsupported claims or vague sections, still usable
2 – Poor: multiple hallucinations or missing sections, limited usefulness
1 – Very poor: mostly fabricated, incoherent, or completely off-topic

Consider:
- Factual grounding (only facts from the provided articles)
- Coverage of key risk signals
- Clarity and structure
- Actionability of the bottom line / watch list

Respond with ONLY a JSON object, nothing else:
{"score": <integer 1-5>, "reason": "<one sentence>"}"""

def compute_llm_judge(
    query: str,
    report: str,
    context: str,
    client: Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Returns {"score": int, "reason": str}"""
    prompt = (
        f"## Query:\n{query}\n\n"
        f"## Retrieved Context (articles):\n{context}\n\n"
        f"## Generated Report:\n{report}"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=LLM_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  LLM judge error: {e}")
        return {"score": None, "reason": str(e)}


# ────────────────────────────────────────────────────────────────────
# 5.  Full Evaluation Runner
# ────────────────────────────────────────────────────────────────────
def evaluate_reports(
    query: str,
    reports: dict[str, str],        # {strategy_name: report_text}
    context: str,                   # the retrieved articles passed to Generator
    reference: str | None = None,   # optional gold-standard answer
    client: Anthropic | None = None,
    model: str = "claude-sonnet-4-20250514",
    run_groundedness: bool = True,
    run_llm_judge: bool = True,
    groundedness_delay: float = 0.3,
) -> dict[str, dict]:
    """
    Evaluate all strategy reports and return a results dict.

    Returns
    -------
    {
        "zero_shot":  {"rouge_l": 0.22, "bertscore": 0.85, "groundedness": 0.68, "llm_judge": {"score": 4, "reason": "..."}},
        "citation":   {...},
        ...
    }
    """
    if client is None:
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = {}

    for strategy, report in reports.items():
        print(f"\n── Evaluating: {strategy} ──────────────────────")
        entry = {}

        # ROUGE-L
        if reference:
            rl = compute_rouge_l(report, reference)
            bs = compute_bertscore(report, reference)
            entry["rouge_l"] = round(rl, 4) if rl is not None else None
            entry["bertscore"] = round(bs, 4) if bs is not None else None
            print(f"  ROUGE-L:    {entry['rouge_l']}")
            print(f"  BERTScore:  {entry['bertscore']}")
        else:
            entry["rouge_l"] = None
            entry["bertscore"] = None
            print("  ROUGE-L / BERTScore: skipped (no reference provided)")

        # Groundedness
        if run_groundedness:
            print("  Computing groundedness (sentence-by-sentence)...")
            g = compute_groundedness(report, context, client, model, groundedness_delay)
            entry["groundedness"] = round(g, 4)
            print(f"  Groundedness: {entry['groundedness']}")
        else:
            entry["groundedness"] = None

        # LLM Judge
        if run_llm_judge:
            print("  Running LLM judge...")
            j = compute_llm_judge(query, report, context, client, model)
            entry["llm_judge"] = j
            print(f"  LLM Judge score: {j.get('score')} — {j.get('reason')}")
            time.sleep(1.0)   # rate-limit courtesy between strategies
        else:
            entry["llm_judge"] = None

        results[strategy] = entry

    return results


# ────────────────────────────────────────────────────────────────────
# 6.  Pretty printer / summary table
# ────────────────────────────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "="*70)
    print(f"{'STRATEGY':<18} {'ROUGE-L':>9} {'BERTScore':>10} {'Grounded':>10} {'LLM Judge':>10}")
    print("="*70)
    for strat, m in results.items():
        rl  = f"{m['rouge_l']:.4f}"  if m["rouge_l"]     is not None else "  —  "
        bs  = f"{m['bertscore']:.4f}" if m["bertscore"]   is not None else "  —  "
        gr  = f"{m['groundedness']:.4f}" if m["groundedness"] is not None else "  —  "
        lj  = str(m["llm_judge"]["score"]) if (m["llm_judge"] and m["llm_judge"].get("score")) else "  —  "
        print(f"{strat:<18} {rl:>9} {bs:>10} {gr:>10} {lj:>10}")
    print("="*70)


# ────────────────────────────────────────────────────────────────────
# 7.  Demo / quick test  (uses mock data so you can run without CSV)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Import your actual pipeline ──────────────────────────────────
    # Uncomment the block below to run against real data:
    #
    # from main import load_data, run_pipeline
    # from generation import Generator
    #
    # df, documents, metadata, doc_to_idx = load_data("data/merged_df_v2.csv")
    # query = "After Russia invaded Ukraine, provide crude oil market snapshot"
    #
    # gen = Generator(strategy="citation")
    # # run_pipeline returns the report; we need the reranked docs too —
    # # modify run_pipeline to also return reranked + context if needed.
    # reports = gen.generate_all_strategies(query, reranked_docs, graph_context)

    # ── Mock data for standalone testing ────────────────────────────
    from generation import Generator, build_context

    MOCK_DOCS = [
        {
            "text": (
                "Title: Russia launches full-scale invasion of Ukraine\n"
                "Description: Russian forces crossed into Ukraine on multiple fronts. "
                "Brent crude surged past $100/barrel. NATO condemned the attack.\n"
                "Commodity: crude oil\nCategory: Geopolitics & Policy\nRisk: Geopolitical Conflict"
            ),
            "score": 0.92, "date": "2022-02-24", "commodity": "crude oil",
        },
        {
            "text": (
                "Title: IEA members release 60 million barrels from strategic reserves\n"
                "Description: Emergency coordinated release aimed to calm energy markets. "
                "Oil prices pulled back slightly from intraday highs.\n"
                "Commodity: crude oil\nCategory: Inventory & Storage\nRisk: Inventory Shock"
            ),
            "score": 0.78, "date": "2022-02-25", "commodity": "crude oil",
        },
        {
            "text": (
                "Title: SWIFT sanctions cut Russian banks from global payments\n"
                "Description: Western allies agreed to remove selected Russian banks from SWIFT. "
                "Analysts warn Russian oil exports could be disrupted.\n"
                "Commodity: crude oil\nCategory: Financial Flows/Positioning\nRisk: Supply Chain Blockage"
            ),
            "score": 0.71, "date": "2022-02-26", "commodity": "crude oil",
        },
    ]

    MOCK_GRAPH = ["crude oil", "geopolitical conflict", "supply chain blockage", "Ukraine"]
    QUERY = "After Russia invaded Ukraine, provide crude oil market snapshot"

    # Optional: a reference answer for ROUGE / BERTScore
    # Set to None if you don't have one
    REFERENCE = None

    # Build context string (same as what Generator sees)
    CONTEXT = build_context(MOCK_DOCS)

    print("Generating reports for all 4 strategies...")
    gen = Generator(strategy="citation")
    reports = gen.generate_all_strategies(QUERY, MOCK_DOCS, MOCK_GRAPH, delay=1.5)

    # Save reports to disk for inspection
    os.makedirs("eval_outputs", exist_ok=True)
    for strat, rpt in reports.items():
        with open(f"eval_outputs/{strat}_report.txt", "w") as f:
            f.write(rpt)
    print("Reports saved to eval_outputs/")

    # Run evaluation
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = evaluate_reports(
        query=QUERY,
        reports=reports,
        context=CONTEXT,
        reference=REFERENCE,
        client=client,
        run_groundedness=True,
        run_llm_judge=True,
        groundedness_delay=0.3,
    )

    # Print summary table
    print_summary(results)

    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval_outputs/eval_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")