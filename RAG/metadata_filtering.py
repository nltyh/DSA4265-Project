import re
import logging
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

    # ---- Extract Date and Range ----
    def extract_dates(self, query):
        import calendar
        query_lower = query.lower()
        months_re = r'january|february|march|april|may|june|july|august|september|october|november|december'

        # 1. Quarters: Q[1-4] YYYY
        qm = re.search(r'\bq([1-4])\s+(\d{4})\b', query, re.IGNORECASE)
        if qm:
            q, y = int(qm.group(1)), int(qm.group(2))
            start_month = (q - 1) * 3 + 1
            end_month = q * 3
            _, last_day = calendar.monthrange(y, end_month)
            return {'date': f'{y}-{start_month:02d}-15', 'start_date': f'{y}-{start_month:02d}-01', 'end_date': f'{y}-{end_month:02d}-{last_day:02d}'}

        # 2. Month-Month Year: August-September 2023
        mm_re = fr'\b({months_re})\s*(?:-|to|and)\s*({months_re})\s+(\d{{4}})\b'
        mm_match = re.search(mm_re, query, re.IGNORECASE)
        if mm_match:
            m1, m2, y = mm_match.groups()
            d1 = dateparser.parse(f'1 {m1} {y}')
            d2 = dateparser.parse(f'1 {m2} {y}')
            if d1 and d2:
                _, last_day = calendar.monthrange(int(y), d2.month)
                return {'date': d1.strftime('%Y-%m-%d'), 'start_date': d1.strftime('%Y-%m-%d'), 'end_date': f'{y}-{d2.month:02d}-{last_day:02d}'}

        # 3. Final week of month year
        fw_re = fr'\b(?:final|last) week of ({months_re})\s+(\d{{4}})\b'
        fw_match = re.search(fw_re, query, re.IGNORECASE)
        if fw_match:
            m, y = fw_match.groups()
            d = dateparser.parse(f'1 {m} {y}')
            if d:
                _, last_day = calendar.monthrange(int(y), d.month)
                start_day = max(1, last_day - 6)
                return {'date': f'{y}-{d.month:02d}-{start_day:02d}', 'start_date': f'{y}-{d.month:02d}-{start_day:02d}', 'end_date': f'{y}-{d.month:02d}-{last_day:02d}'}

        # 4. Semantic early/mid/late month year
        semantic_pattern = fr'\b(early|mid|late)\s+({months_re})(?:,?\s*(\d{{4}}))?\b'
        sem_match = re.search(semantic_pattern, query, re.IGNORECASE)
        if sem_match:
            modifier = sem_match.group(1).lower()
            month_str = sem_match.group(2)
            year_str = sem_match.group(3) or str(datetime.now().year)
            d = dateparser.parse(f'1 {month_str} {year_str}')
            if d:
                _, last_day = calendar.monthrange(d.year, d.month)
                if modifier == 'early':
                    start_date = f'{d.year}-{d.month:02d}-01'
                    end_date   = f'{d.year}-{d.month:02d}-15'
                elif modifier == 'mid':
                    start_date = f'{d.year}-{d.month:02d}-10'
                    end_date   = f'{d.year}-{d.month:02d}-20'
                else: # late
                    start_date = f'{d.year}-{d.month:02d}-16'
                    end_date   = f'{d.year}-{d.month:02d}-{last_day:02d}'
                return {'date': start_date, 'start_date': start_date, 'end_date': end_date}

        # 5. Day-Day Month Year: 23-24 March 2026 or 25 March 2026
        dd_re = fr'\b(\d{{1,2}})(?:\s*(?:-|to|and)\s*(\d{{1,2}}))?\s+({months_re})\s+(\d{{4}})\b'
        dd_match = re.search(dd_re, query, re.IGNORECASE)
        if dd_match:
            d1, d2, m, y = dd_match.groups()
            dt1 = dateparser.parse(f'{d1} {m} {y}')
            if dt1:
                if d2:
                    dt2 = dateparser.parse(f'{d2} {m} {y}')
                    if dt2:
                        return {'date': dt1.strftime('%Y-%m-%d'), 'start_date': dt1.strftime('%Y-%m-%d'), 'end_date': dt2.strftime('%Y-%m-%d')}
                return {'date': dt1.strftime('%Y-%m-%d'), 'start_date': None, 'end_date': None}

        # 6. Month Year: February 2024 (without explicit day, meaning entire month)
        my_re = fr'\b({months_re})\s+(\d{{4}})\b'
        my_match = re.search(my_re, query, re.IGNORECASE)
        if my_match:
            m, y = my_match.groups()
            d = dateparser.parse(f'1 {m} {y}')
            if d:
                _, last_day = calendar.monthrange(d.year, d.month)
                return {'date': f'{d.year}-{d.month:02d}-15', 'start_date': f'{d.year}-{d.month:02d}-01', 'end_date': f'{d.year}-{d.month:02d}-{last_day:02d}'}

        # 7. Fallback generic dates (YYYY-MM-DD or DD/MM/YYYY)
        gen_re = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
        gen_match = re.search(gen_re, query, re.IGNORECASE)
        if gen_match:
            d = dateparser.parse(gen_match.group(0))
            if d:
                return {"date": d.strftime("%Y-%m-%d"), "start_date": None, "end_date": None}

        return {"date": None, "start_date": None, "end_date": None}

    # ---- Parse Query ----
    def parse_query(self, query):
        query_lower = query.lower()

        commodity = self.extract_commodity(query)
        dates = self.extract_dates(query)
        date = dates["date"]
        start_date = dates["start_date"]
        end_date = dates["end_date"]

        window_days = 3
        if start_date and end_date:
            window_days = None
        else:
            if "recent" in query_lower:
                date = datetime.now().strftime("%Y-%m-%d")
                window_days = 7
            elif "on" in query_lower:
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
            "window_days": window_days,
            "start_date": start_date,
            "end_date": end_date,
        }


def _parse_date(date_str: str) -> datetime:
    """Parse date string supporting both YYYY-MM-DD and DD/MM/YYYY formats."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return datetime.strptime(date_str, "%d/%m/%Y")


class MetadataFilter:
    def __init__(self, target_date=None, window_days=None, commodity=None, start_date=None, end_date=None):
        self.target_date = target_date
        self.window_days = window_days
        self.commodity = commodity
        self.start_date = start_date
        self.end_date = end_date

    def apply(self, documents, metadata):
        filtered_docs = []
        filtered_meta = []

        if self.start_date and self.end_date:
            range_start = _parse_date(self.start_date).date()
            range_end = _parse_date(self.end_date).date()
        elif self.target_date:
            target = _parse_date(self.target_date).date()
            range_start = None
            range_end = None
        else:
            target = None

        for doc, meta in zip(documents, metadata):
            keep = True

            # ---- Date Filtering ----
            if self.start_date and self.end_date:
                try:
                    doc_date = _parse_date(meta["date"]).date()
                    if not (range_start <= doc_date <= range_end):
                        keep = False
                except (ValueError, KeyError) as e:
                    logging.warning("Date parsing failed: %s", e)

            elif self.target_date:
                try:
                    doc_date = _parse_date(meta["date"]).date()
                    if self.window_days is not None:
                        if abs((doc_date - target).days) > self.window_days:
                            keep = False
                    else:
                        if doc_date != target:
                            keep = False
                except (ValueError, KeyError) as e:
                    logging.warning("Date parsing failed: %s", e)

            # ---- Commodity Filtering ----
            if self.commodity:
                doc_commodity = meta.get("commodity") or ""
                if self.commodity.lower() not in str(doc_commodity).lower():
                    keep = False

            if keep:
                filtered_docs.append(doc)
                filtered_meta.append(meta)

        return filtered_docs, filtered_meta