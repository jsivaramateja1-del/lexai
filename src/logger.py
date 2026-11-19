"""Query logging. Every question becomes one row of analytics/queries.csv.

That CSV is the dataset for the Power BI dashboard, so log from the very first
working pipeline run - not in Week 7 when you suddenly need data.
"""

from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone

from src.config import ANALYTICS_DIR, QUERY_LOG

FIELDS = [
    "query_id",
    "timestamp_ist",
    "question",
    "answer_mode",       # llm | extractive | no_answer | error
    "num_sources",
    "sources",           # "Patents_Act_1970.pdf p.14 | ..."
    "top_distance",
    "retrieval_ms",
    "llm_ms",
    "total_ms",
    "feedback",          # up | down | empty
    "error",
]


def _ensure_file() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    if not QUERY_LOG.exists():
        with QUERY_LOG.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writeheader()


def new_query_id() -> str:
    return uuid.uuid4().hex[:10]


def log_query(record: dict) -> str:
    """Append one row. Missing keys are written empty instead of crashing."""
    _ensure_file()
    query_id = record.get("query_id") or new_query_id()
    row = {field: "" for field in FIELDS}
    row.update({k: v for k, v in record.items() if k in FIELDS})
    row["query_id"] = query_id
    row["timestamp_ist"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with QUERY_LOG.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
    return query_id


def set_feedback(query_id: str, feedback: str) -> bool:
    """Rewrite the CSV with one row's feedback changed. True if found."""
    if not QUERY_LOG.exists():
        return False
    with QUERY_LOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    found = False
    for row in rows:
        if row.get("query_id") == query_id:
            row["feedback"] = feedback
            found = True
            break
    if not found:
        return False

    with QUERY_LOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return True
