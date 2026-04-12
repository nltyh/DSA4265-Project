import os
import sys
import time
import re
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_SOURCE = "csv"          # "api" | "csv"
FETCH_ALPHAVANTAGE = True
FETCH_NEWSAPI = True
LABEL_WITH_LLM = False

ALPHA_KEY   = os.getenv("ALPHAVANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DATA_PATH   = "final_df.csv"

AV_CONFIG = {
    "start_date": "20000101T0000",
    "end_date":   "20150101T0000",
    "tickers":    ["CL"],  # ["CL", "USO", "NG", "CRAK", "LNG"]
    "limit":      10000,
    "topics":     ["energy_transportation"],  # ["energy_transportation", "economy_macro", "economy_monetary", "economy_fiscal"]
}

NEWSAPI_CONFIG = {
    "query": '"crude oil" OR "diesel" OR "LNG" OR "liquefied natural gas"',
    "days_back": 30,
    "page_size": 100,
    "domains": "reuters.com,bloomberg.com,ft.com,oilprice.com,wsj.com",
}

# ── Mappings ──────────────────────────────────────────────────────────────────

TICKER_MAPPING = {
    "CL":   "crude oil",
    "USO":  "crude oil",
    "NG":   "LNG",
    "CRAK": "diesel",
    "LNG":  "LNG"
}

COMMODITY_KEYWORDS = {
    "crude oil": ["crude oil", "oil prices", "oil market", "brent", "wti",
                  "west texas intermediate", "opec", "opec+", "oil production",
                  "oil supply", "oil demand", "barrel", "bpd", "petroleum", "oil futures"],
    "diesel":    ["diesel", "gasoil", "ultra low sulfur diesel", "ulsd", "heating oil",
                  "distillate", "middle distillate", "diesel prices", "diesel demand",
                  "diesel supply", "refining margins", "crack spread"],
    "LNG":       ["lng", "liquefied natural gas", "natural gas", "gas prices", "ttf",
                  "henry hub", "jkm", "gas supply", "gas demand", "gas storage",
                  "regasification", "lng cargo", "lng export", "lng import"],
}

COMMODITY_PATTERNS = {
    k: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\b", re.IGNORECASE)
    for k, v in COMMODITY_KEYWORDS.items()
}

SYSTEM_INSTRUCTION_EXPAND = """
You are an expert Energy Market & Supply Chain Analyst.
Your task is to analyze news title and description to categorize energy news for a high-precision Risk Management RAG system.

Assign each item:
1. news_category
2. risk_category
3. risk_severity
4. market_sentiment

--- PREDEFINED NEWS CATEGORIES ---
1. Geopolitics & policy
2. Supply (production / upstream)
3. Refining (downstream)
4. Inventory & storage
5. Demand / macro activity
6. LNG & natural gas
7. Weather
8. Shipping & logistics
9. Financial flows / positioning
10. Currency & interest rates
11. Accidents / disruptions
12. Energy transition / structural

--- PREDEFINED RISK CATEGORIES ---
- Supply Chain Blockage
- Production Shortfall
- Refining Outage
- Geopolitical Conflict
- Macro-Economic Cooling
- Inventory Shock
- Infrastructure Damage
- Weather Extremity
- Regulatory Constraint
- No Significant Risk

--- PREDEFINED RISK SEVERITY ---
- Low
- Medium
- High

--- PREDEFINED MARKET SENTIMENT ---
- Bullish
- Bearish
- Neutral

Rules:
- Choose exactly one news_category from the predefined list.
- Choose exactly one risk_category from the predefined list.
- Choose exactly one risk_severity from: Low, Medium, High.
- Choose exactly one market_sentiment from: Bullish, Bearish, Neutral.
- Base the label on the title + description only.
- If the article is mainly about company earnings, stock movement, analyst ratings, or financing, prefer:
  news_category = Financial flows / positioning
- If the article is mainly about LNG, gas supply, pipelines, or gas exports/imports, prefer:
  news_category = LNG & natural gas
- If no material risk is present, use:
  risk_category = No Significant Risk
  risk_severity = Low
- market_sentiment should reflect likely commodity-market impact, not just company-level stock tone.

--- OUTPUT FORMAT ---
Return a JSON list.
Each object must contain exactly:
{
  "id": <input id>,
  "news_category": "<one predefined news category>",
  "risk_category": "<one predefined risk category>",
  "risk_severity": "<Low|Medium|High>",
  "market_sentiment": "<Bullish|Bearish|Neutral>"
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_date(date_str: str) -> str:
    """Parse AlphaVantage YYYYMMDDTHHMMSS into YYYY-MM-DD."""
    return datetime.strptime(date_str, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")

def detect_commodities(text: str) -> str:
    return ",".join(
        commodity for commodity, pattern in COMMODITY_PATTERNS.items()
        if pattern.search(str(text))
    )

def build_batch_prompt(batch_df: pd.DataFrame) -> str:
    prompt_input = ""
    for idx, row in batch_df.iterrows():
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        commodities = str(row.get("relevant_commodities", "")).strip()

        prompt_input += (
            f"ID: {idx} | "
            f"Title: {title} | "
            f"Description: {description} | "
            f"Relevant Commodities: {commodities}\n"
        )

    return SYSTEM_INSTRUCTION_EXPAND + "\n\nAnalyze these items:\n" + prompt_input

def _extract_json_list(text: str) -> list[dict]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(cleaned[start:end + 1])

def expand_categories_with_llm(
    df: pd.DataFrame,
    client=None,
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
    sleep_seconds: float = 0.5
) -> pd.DataFrame:
    working_df = df.copy()
    label_cols = [
        "news_category",
        "risk_category",
        "risk_severity",
        "market_sentiment",
    ]

    if "commodity" in working_df.columns and "relevant_commodities" not in working_df.columns:
        working_df = working_df.rename(columns={"commodity": "relevant_commodities"})

    required_cols = ["date", "title", "description", "relevant_commodities"]
    missing = [c for c in required_cols if c not in working_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    existing_label_cols = [col for col in label_cols if col in working_df.columns]
    if existing_label_cols:
        working_df = working_df.drop(columns=existing_label_cols)

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for LLM labeling.") from exc

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when label_with_llm=True.")

        client = OpenAI(api_key=openai_api_key)

    all_results = []

    for start in range(0, len(working_df), batch_size):
        batch = working_df.iloc[start:start + batch_size]
        prompt = build_batch_prompt(batch)

        response = client.responses.create(
            model=model,
            input=prompt,
        )

        try:
            batch_results = _extract_json_list(response.output_text)
        except json.JSONDecodeError as exc:
            print(f"JSON parse error for batch starting at row {start}: {exc}")
            print(response.output_text[:1000])
            continue

        all_results.extend(batch_results)
        print(f"Processed rows {start} to {start + len(batch) - 1}")
        time.sleep(sleep_seconds)

    labels_df = pd.DataFrame(all_results)

    if labels_df.empty:
        raise ValueError("No labels returned from LLM.")

    if "id" not in labels_df.columns:
        raise ValueError("LLM output must contain 'id'.")

    labels_df["id"] = labels_df["id"].astype(int)

    merged = (
        working_df.reset_index()
        .merge(labels_df, left_on="index", right_on="id", how="left")
        .drop(columns=["index", "id"])
    )

    final_cols = [
        "date",
        "title",
        "description",
        "relevant_commodities",
        *label_cols,
    ]
    return merged[final_cols]

# ── Source fetchers ───────────────────────────────────────────────────────────

def fetch_alphavantage(cfg: dict) -> pd.DataFrame:
    print("Fetching AlphaVantage news...")
    rows = []

    for tick in cfg["tickers"]:
        for topic in cfg["topics"]:
            time.sleep(5)
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&topics={topic}"
                f"&tickers={tick}"
                f"&limit={cfg['limit']}"
                f"&time_from={cfg['start_date']}"
                f"&time_to={cfg['end_date']}"
                f"&apikey={ALPHA_KEY}"
            )
            data = requests.get(url).json()

            for article in data.get("feed", []):
                rows.append({
                    "date":        normalize_date(article["time_published"]),
                    "title":       article["title"],
                    "description": article["summary"],
                    "commodity":   TICKER_MAPPING.get(tick, tick),
                })

    print(f"  AlphaVantage: {len(rows)} articles fetched.")
    return pd.DataFrame(rows, columns=["date", "title", "description", "commodity"])

def fetch_newsapi(cfg: dict) -> pd.DataFrame:
    print("Fetching NewsAPI articles...")
    client    = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.today() - timedelta(days=cfg["days_back"])).strftime("%Y-%m-%d")
    to_date   = datetime.today().strftime("%Y-%m-%d")

    response = client.get_everything(
        q=cfg["query"],
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="publishedAt",
        page_size=cfg["page_size"],
        domains=cfg["domains"],
    )

    df = pd.DataFrame(response["articles"])
    df["source"]    = df["source"].apply(lambda x: x.get("name") if isinstance(x, dict) else x)
    df["date"]      = pd.to_datetime(df["publishedAt"]).dt.date.astype(str)
    df["commodity"] = df["description"].apply(detect_commodities)

    print(f"  NewsAPI: {len(df)} articles fetched.")
    return df[["date", "title", "description", "commodity"]].copy()

def load_csv(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: Could not find required data file at {path}")
        sys.exit(1)

# ── Main loader ───────────────────────────────────────────────────────────────

def load_data(
    source: str             = DATA_SOURCE,
    alphavantage: bool      = FETCH_ALPHAVANTAGE,
    newsapi: bool           = FETCH_NEWSAPI,
    data_path: str | Path   = DATA_PATH,
    label_with_llm: bool    = LABEL_WITH_LLM,
    openai_client           = None,
    llm_model: str          = "gpt-4o-mini",
    llm_batch_size: int     = 20,
    llm_sleep_seconds: float = 0.5
) -> pd.DataFrame:

    if source == "csv":
        df = load_csv(data_path)
        if label_with_llm:
            return expand_categories_with_llm(
                df=df,
                client=openai_client,
                model=llm_model,
                batch_size=llm_batch_size,
                sleep_seconds=llm_sleep_seconds,
            )
        return df

    if source == "api":
        if not alphavantage and not newsapi:
            raise ValueError("At least one of FETCH_ALPHAVANTAGE or FETCH_NEWSAPI must be True.")

        frames = []
        if alphavantage:
            frames.append(fetch_alphavantage(AV_CONFIG))
        if newsapi:
            frames.append(fetch_newsapi(NEWSAPI_CONFIG))

        merged = pd.concat(frames, axis=0, ignore_index=True)
        print(f"Total articles loaded: {len(merged)}")
        if label_with_llm:
            return expand_categories_with_llm(
                df=merged,
                client=openai_client,
                model=llm_model,
                batch_size=llm_batch_size,
                sleep_seconds=llm_sleep_seconds,
            )
        return merged

    raise ValueError(f"Unknown DATA_SOURCE '{source}'. Use 'api' or 'csv'.")
