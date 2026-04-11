from datetime import datetime
import numpy as np


def _parse_date(date_str: str) -> datetime:
    """Parse date string supporting both YYYY-MM-DD and DD/MM/YYYY formats."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return datetime.strptime(date_str, "%d/%m/%Y")

class TimeWeighter:
    def __init__(self, decay_rate=0.1):
        """
        decay_rate: higher → faster decay
        """
        self.decay_rate = decay_rate

    def apply(self, documents, reference_date=None):
        """
        documents: list of dicts with arbitrary metadata
        {
            "text": str,
            "score": float,
            "date": "YYYY-MM-DD",
            ...
        }
        reference_date: str | None
            If provided ("YYYY-MM-DD"), decay is anchored to this date (e.g. the
            query date) instead of wallclock time.  Documents closer to this date
            receive higher weights.  Falls back to datetime.now() when None,
            preserving the original live-query behaviour.
        """
        # Anchor: query date if given, wallclock otherwise
        if reference_date:
            try:
                anchor = _parse_date(reference_date)
            except ValueError:
                anchor = datetime.now()
        else:
            anchor = datetime.now()

        weighted_docs = []

        for doc in documents:
            # ---- Handle missing or bad date safely ----
            try:
                doc_date = _parse_date(doc["date"])
                # abs() → symmetric: penalise articles far from the query date
                # in either direction, not just older ones
                days_diff = abs((anchor - doc_date).days)
                time_weight = np.exp(-self.decay_rate * days_diff)
            except Exception:
                # fallback: no time decay if date invalid
                time_weight = 1.0

            new_score = doc["score"] * time_weight

            # Preserve all metadata
            new_doc = doc.copy()
            new_doc["score"] = new_score

            weighted_docs.append(new_doc)

        # sort again
        weighted_docs.sort(key=lambda x: x["score"], reverse=True)
        return weighted_docs