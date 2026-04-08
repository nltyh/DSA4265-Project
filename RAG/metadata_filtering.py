import re
import dateparser
from datetime import datetime


class QueryProcessor:
    def __init__(self):
        self.commodities = [
            "crude oil", "oil", "wheat", "copper",
            "soybean", "gas", "coal"
        ]

    # ---- Extract Commodity ----
    def extract_commodity(self, query):
        query_lower = query.lower()
        for c in self.commodities:
            if c in query_lower:
                return c
        return None

    # ---- Extract Date ----
    def extract_date(self, query):
        pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            date = dateparser.parse(match.group(0))
            if date:
                return date.strftime("%Y-%m-%d")
        return None

    # ---- Parse Query ----
    def parse_query(self, query):
        query_lower = query.lower()

        commodity = self.extract_commodity(query)
        date = self.extract_date(query)

        # Default: small window
        window_days = 3

        # ---- Intent-based logic ----
        if "recent" in query_lower:
            date = datetime.now().strftime("%Y-%m-%d")
            window_days = 7

        elif "on" in query_lower:
            # exact date match
            window_days = None

        elif "after" in query_lower:
            window_days = 7

        elif "before" in query_lower:
            window_days = 7

        elif "during" in query_lower:
            window_days = 5

        return {
            "commodity": commodity,
            "date": date,
            "window_days": window_days
        }


class MetadataFilter:
    def __init__(self, target_date=None, window_days=None, commodity=None):
        self.target_date = target_date
        self.window_days = window_days
        self.commodity = commodity

    def apply(self, documents, metadata):
        filtered_docs = []
        filtered_meta = []

        # Prepare target date
        if self.target_date:
            target = datetime.strptime(self.target_date, "%Y-%m-%d")

        for doc, meta in zip(documents, metadata):
            keep = True

            # ---- Date Filtering ----
            if self.target_date:
                try:
                    doc_date = datetime.strptime(meta["date"], "%Y-%m-%d")

                    if self.window_days is not None:
                        # ✅ Window filtering
                        if abs((doc_date - target).days) > self.window_days:
                            keep = False
                    else:
                        # ✅ Exact date filtering
                        if doc_date != target:
                            keep = False

                except:
                    # fallback: keep doc if date parsing fails
                    pass

            # ---- Commodity Filtering ----
            if self.commodity:
                doc_commodity = meta.get("commodity") or ""
                if self.commodity.lower() not in str(doc_commodity).lower():
                    keep = False

            if keep:
                filtered_docs.append(doc)
                filtered_meta.append(meta)

        return filtered_docs, filtered_meta