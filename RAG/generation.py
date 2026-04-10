import os
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ───────────────────────────────────────────────
# System Prompt  (shared across all strategies)
# ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior commodity risk analyst. Your job is to read retrieved \
news articles and produce a structured commodity risk report that helps traders and risk \
managers make informed decisions.

Each retrieved article comes with the following structured metadata:
- date: publication date
- title: headline
- description: article body
- relevant_commodities: which commodity this affects
- news_category: one of [Geopolitics & Policy, Supply (Production/Upstream),
  Refining (Downstream), Inventory & Storage, Demand/Macro Activity, LNG & Natural Gas,
  Weather, Shipping & Logistics, Financial Flows/Positioning, Currency & Interest Rates,
  Accidents/Disruptions, Energy Transition/Structural]
- risk_category: one of [Supply Chain Blockage, Production Shortfall, Refining Outage,
  Geopolitical Conflict, Macro-Economic Cooling, Inventory Shock, Infrastructure Damage,
  Weather Extremity, Regulatory Constraint, No Significant Risk]
- risk_severity: High / Medium / Low
- market_sentiment: Bullish / Bearish / Neutral

Rules you must follow:
1. Only use facts from the retrieved articles. Never hallucinate.
2. If articles contradict each other, explicitly flag the contradiction.
3. If data is insufficient for a section, write "Insufficient data from current sources."
4. Always assign an overall sentiment — if mixed, explain why.
5. Keep tone analytical and concise. No alarmism.
6. If the user asks about multiple commodities, generate one report block per commodity."""


# ───────────────────────────────────────────────
# Report template  (injected into every prompt)
# ───────────────────────────────────────────────
REPORT_TEMPLATE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## COMMODITY RISK REPORT
**Commodity:** {commodity}
**Report Date:** {date}
**Articles Analysed:** {n} articles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1. OVERALL SENTIMENT
**[Bullish / Bearish / Neutral / Mixed]**
> One sentence explaining the dominant signal and why.
> If Mixed: explain which forces are pulling in opposite directions.

---
### 2. KEY RISK SIGNALS
For each distinct risk found across the articles, output one block:

**[news_category] | [risk_category] | Severity: [High/Medium/Low]**
- **Headline:** <title>
- **What happened:** 1–2 sentence factual summary
- **Market impact:** How does this affect supply, demand, or price?
- **Sentiment contribution:** Bullish / Bearish / Neutral — and why

*(Repeat for each significant risk signal.)*

---
### 3. SIGNAL SYNTHESIS
2–4 sentences:
- Are the risk signals aligned or contradictory?
- Are risks sentiment-driven or confirmed by fundamentals?
- What is the net effect on commodity risk exposure right now?

---
### 4. WATCH LIST
3–5 specific triggers to monitor:
- [ ] <trigger 1> → what it would mean if it happens
- [ ] <trigger 2> → what it would mean if it happens
- [ ] <trigger 3> → what it would mean if it happens

---
### 5. BOTTOM LINE
**Risk Level: [High / Medium / Low]**
1–2 sentences. Most important thing a risk manager should know right now.
Flag explicitly if the signal is fragile, unconfirmed, or contradicted by other data.
"""


# ───────────────────────────────────────────────
# Context builder  (reranked docs → article block)
# ───────────────────────────────────────────────
def build_context(reranked_docs: list[dict]) -> str:
    """
    reranked_docs: list of dicts from run_pipeline, each containing:
        text, score, date, commodity
    The 'text' field already has Title/Description/Commodity/Category/Risk
    formatted by load_data() in main.py.
    """
    blocks = []
    for i, doc in enumerate(reranked_docs, 1):
        blocks.append(
            f"[Article {i}]\n"
            f"Date: {doc.get('date', 'unknown')}\n"
            f"Rerank Score: {doc['score']:.4f}\n"
            f"{doc['text'].strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def extract_commodity(reranked_docs: list[dict]) -> str:
    """Best-effort commodity label from the top retrieved doc."""
    if reranked_docs:
        return reranked_docs[0].get("commodity", "commodity")
    return "commodity"


# ───────────────────────────────────────────────
# Prompt builders  (one per strategy)
# ───────────────────────────────────────────────
def _prompt_zero_shot(context: str, query: str, commodity: str,
                      date: str, n: int, graph_context: list) -> str:
    graph_note = (
        f"\n\n## Related Entities (Graph Context):\n{', '.join(graph_context)}"
        if graph_context else ""
    )
    report_shell = REPORT_TEMPLATE.format(commodity=commodity, date=date, n=n)
    return (
        f"## Retrieved News Articles:\n{context}"
        f"{graph_note}\n\n"
        f"## User Question:\n{query}\n\n"
        "---\n"
        "Using ONLY the articles above, generate a commodity risk report "
        f"in this exact structure:\n{report_shell}"
    )


def _prompt_few_shot(context: str, query: str, commodity: str,
                     date: str, n: int, graph_context: list) -> str:
    example = """## Example — well-structured KEY RISK SIGNAL block:

**[Geopolitics & Policy] | [Geopolitical Conflict] | Severity: High**
- **Headline:** Russia launches full-scale invasion of Ukraine
- **What happened:** Russian forces invaded Ukraine on multiple fronts, triggering \
martial law and international sanctions. Brent crude surged above $100/barrel on \
the day of the attack.
- **Market impact:** Immediate upward price shock from supply disruption fears; \
Russian crude exports (~10 mb/d, ~10% of global supply) at risk of sanction-driven \
dislocation. Replacement barrels unavailable at short notice.
- **Sentiment contribution:** Bullish — hard supply shock with no near-term substitute \
source; confirmed by physical price action, not just headlines.

---
"""
    graph_note = (
        f"\n\n## Related Entities (Graph Context):\n{', '.join(graph_context)}"
        if graph_context else ""
    )
    report_shell = REPORT_TEMPLATE.format(commodity=commodity, date=date, n=n)
    return (
        f"{example}"
        f"## Retrieved News Articles:\n{context}"
        f"{graph_note}\n\n"
        f"## User Question:\n{query}\n\n"
        "---\n"
        "Now generate a full commodity risk report following the same structure "
        f"and analytical depth as the example above:\n{report_shell}"
    )


def _prompt_citation(context: str, query: str, commodity: str,
                     date: str, n: int, graph_context: list) -> str:
    citation_rules = """## CITATION RULES (mandatory — no exceptions):
- Every factual claim must end with [Article N] citing its source.
- If a claim draws on multiple articles: [Article 1, 2].
- If no article supports a claim, do NOT include it.
- Never infer facts not explicitly stated in the articles.
- Sentiment labels (Bullish/Bearish) must also be cited.

"""
    graph_note = (
        f"\n\n## Related Entities (Graph Context):\n{', '.join(graph_context)}"
        if graph_context else ""
    )
    report_shell = REPORT_TEMPLATE.format(commodity=commodity, date=date, n=n)
    return (
        f"{citation_rules}"
        f"## Retrieved News Articles:\n{context}"
        f"{graph_note}\n\n"
        f"## User Question:\n{query}\n\n"
        "---\n"
        "Generate a commodity risk report with inline [Article N] citations "
        f"on every factual claim:\n{report_shell}"
    )


def _prompt_reflection(context: str, query: str, commodity: str,
                       date: str, n: int, graph_context: list) -> str:
    graph_note = (
        f"\n\n## Related Entities (Graph Context):\n{', '.join(graph_context)}"
        if graph_context else ""
    )
    report_shell = REPORT_TEMPLATE.format(commodity=commodity, date=date, n=n)
    return (
        f"## Retrieved News Articles:\n{context}"
        f"{graph_note}\n\n"
        f"## User Question:\n{query}\n\n"
        "---\n"
        "You will work in TWO passes.\n\n"
        "**PASS 1 — Draft Report:**\n"
        f"Write a draft commodity risk report using this structure:\n{report_shell}\n\n"
        "---\n"
        "**PASS 2 — Reflection & Revision:**\n"
        "Review your draft against these checks:\n"
        "1. Hallucination check: Did I include any facts NOT in the articles?\n"
        "2. Contradiction check: Did I fail to flag any articles that disagree with each other?\n"
        "3. Coverage check: Are there important signals in the articles I missed?\n"
        "4. Sentiment check: Is my overall sentiment conclusion well-supported by the evidence?\n\n"
        "Output your reflection as:\n\n"
        "[REFLECTION]\n"
        "List 2–4 specific issues found, or write 'No issues found.' if the draft is solid.\n\n"
        "[FINAL REPORT]\n"
        "The fully revised report incorporating your reflection.\n"
        "The final report must follow the same 5-section structure above."
    )


PROMPT_BUILDERS = {
    "zero_shot":  _prompt_zero_shot,
    "few_shot":   _prompt_few_shot,
    "citation":   _prompt_citation,
    "reflection": _prompt_reflection,
}


def _get_response_text(response) -> str:
    return response.choices[0].message.content.strip()


# ───────────────────────────────────────────────
# Generator class
# ───────────────────────────────────────────────
class Generator:
    """
    Parameters
    ----------
    strategy : str
        One of: "zero_shot", "few_shot", "citation", "reflection"
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    model : str
        OpenAI model to use.
    max_tokens : int
        Max tokens for the response.
    top_k_docs : int
        How many reranked docs to pass as context (default 5).
    """

    VALID_STRATEGIES = ("zero_shot", "few_shot", "citation", "reflection")

    def __init__(
        self,
        strategy: str = "citation",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        top_k_docs: int = 5,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.model = model
        self.max_tokens = max_tokens
        self.top_k_docs = top_k_docs

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "No API key provided. Set OPENAI_API_KEY or pass api_key= to Generator()."
            )
        self.client = OpenAI(api_key=key)

    def generate(
        self,
        query: str,
        reranked_docs: list[dict],
        graph_context: list[str] | None = None,
    ) -> str:
        """
        Parameters
        ----------
        query : str
            The original user query.
        reranked_docs : list[dict]
            Output of the reranking step from run_pipeline().
            Each dict must have: text, score, date, commodity.
        graph_context : list[str], optional
            Entity list from GraphRAG.retrieve_subgraph().

        Returns
        -------
        str
            The generated commodity risk report.
        """
        docs = reranked_docs[: self.top_k_docs]
        context = build_context(docs)
        commodity = extract_commodity(docs)
        date = datetime.now().strftime("%Y-%m-%d")
        n = len(docs)
        graph_context = graph_context or []

        prompt = PROMPT_BUILDERS[self.strategy](
            context, query, commodity, date, n, graph_context
        )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        return _get_response_text(response)

    def generate_all_strategies(
        self,
        query: str,
        reranked_docs: list[dict],
        graph_context: list[str] | None = None,
        delay: float = 1.0,
    ) -> dict[str, str]:
        """
        Run all 4 strategies and return a dict of {strategy: report}.
        Useful for evaluation / comparison.

        delay : float
            Seconds to wait between API calls (rate-limit courtesy).
        """
        results = {}
        original_strategy = self.strategy

        for strat in self.VALID_STRATEGIES:
            print(f"  Running strategy: {strat}...")
            self.strategy = strat
            results[strat] = self.generate(query, reranked_docs, graph_context)
            time.sleep(delay)

        self.strategy = original_strategy
        return results


# ───────────────────────────────────────────────
# Quick test  (run directly: python generation.py)
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal mock data — replace with real pipeline output
    mock_docs = [
        {
            "text": (
                "Title: Russia launches full-scale invasion of Ukraine\n"
                "Description: Russian forces crossed into Ukraine on multiple fronts. "
                "Brent crude surged past $100/barrel. NATO condemned the attack.\n"
                "Commodity: crude oil\n"
                "Category: Geopolitics & Policy\n"
                "Risk: Geopolitical Conflict"
            ),
            "score": 0.92,
            "date": "2022-02-24",
            "commodity": "crude oil",
        },
        {
            "text": (
                "Title: IEA members to release 60 million barrels from strategic reserves\n"
                "Description: Emergency release coordinated to calm energy markets. "
                "Oil prices pulled back slightly from intraday highs.\n"
                "Commodity: crude oil\n"
                "Category: Inventory & Storage\n"
                "Risk: Inventory Shock"
            ),
            "score": 0.78,
            "date": "2022-02-25",
            "commodity": "crude oil",
        },
        {
            "text": (
                "Title: SWIFT sanctions cut Russian banks from global payments\n"
                "Description: Western allies agreed to remove selected Russian banks "
                "from SWIFT. Analysts warn Russian oil exports could be disrupted.\n"
                "Commodity: crude oil\n"
                "Category: Financial Flows/Positioning\n"
                "Risk: Supply Chain Blockage"
            ),
            "score": 0.71,
            "date": "2022-02-26",
            "commodity": "crude oil",
        },
    ]

    mock_graph = ["crude oil", "geopolitical conflict", "supply chain blockage", "Ukraine"]
    query = "After Russia invaded Ukraine, provide crude oil market snapshot"

    # Test a single strategy
    strategy = os.environ.get("GENERATION_STRATEGY", "citation")
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*60}\n")

    gen = Generator(strategy=strategy)
    report = gen.generate(query, mock_docs, mock_graph)
    print(report)
