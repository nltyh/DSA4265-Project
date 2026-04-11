import os
import sys
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_SOURCE = "csv"          # "api" | "csv"
FETCH_ALPHAVANTAGE = True
FETCH_NEWSAPI = True

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_date(date_str: str) -> str:
    """Parse AlphaVantage YYYYMMDDTHHMMSS into YYYY-MM-DD."""
    return datetime.strptime(date_str, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")

def detect_commodities(text: str) -> str:
    return ",".join(
        commodity for commodity, pattern in COMMODITY_PATTERNS.items()
        if pattern.search(str(text))
    )

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
    data_path: str | Path   = DATA_PATH
) -> pd.DataFrame:

    if source == "csv":
        return load_csv(data_path)

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
        return merged

    raise ValueError(f"Unknown DATA_SOURCE '{source}'. Use 'api' or 'csv'.")
