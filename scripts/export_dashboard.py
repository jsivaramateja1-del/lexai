"""Prepare dashboard-ready data from the query log and the evaluation results.

    python -m scripts.export_dashboard

Writes analytics/dashboard_data.csv: one row per query, with the derived columns
Power BI needs so the report can be built from a single clean table instead of
repeating the same DAX in five places.

Derived columns
---------------
answered       1 when the assistant produced an answer, 0 when it refused
refused        1 when nothing passed the relevance guard
latency_band   <100 ms, 100-300 ms, 300-1000 ms, >1000 ms
hour_of_day    0-23 in local time, for the time-of-day chart
source_doc     the document most cited for that question, or empty
top_page       the first cited page, as an integer

Why a script and not manual cleanup: the log is appended by the app on every
question, so the dashboard must be rebuildable in one command before a review.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERY_LOG = ROOT / "analytics" / "queries.csv"
EVAL_RESULTS = ROOT / "eval" / "results.csv"
OUT = ROOT / "analytics" / "dashboard_data.csv"

FIELDS = [
    "timestamp", "date", "hour_of_day", "question", "answer_mode",
    "answered", "refused", "num_sources", "source_doc", "top_page",
    "top_distance", "attempts", "retrieval_ms", "llm_ms", "total_ms",
    "latency_band", "feedback",
]


def latency_band(ms: float) -> str:
    if ms < 100:
        return "<100 ms"
    if ms < 300:
        return "100-300 ms"
    if ms < 1000:
        return "300-1000 ms"
    return ">1000 ms"


def parse_time(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def most_cited_document(sources_field: str) -> tuple[str, int | None]:
    """'a.pdf p.10 | a.pdf p.11 | b.pdf p.3' -> ('a.pdf', 10)."""
    parts = [part.strip() for part in (sources_field or "").split("|") if part.strip()]
    if not parts:
        return "", None
    documents = Counter(part.rsplit(" p.", 1)[0] for part in parts)
    document = documents.most_common(1)[0][0]
    for part in parts:
        if part.startswith(document) and " p." in part:
            try:
                return document, int(part.rsplit(" p.", 1)[1])
            except ValueError:
                break
    return document, None


def main() -> None:
    if not QUERY_LOG.exists():
        sys.exit(
            f"No query log at {QUERY_LOG.relative_to(ROOT)}. Ask a few questions in the app first, "
            "or run: python -m scripts.run_eval"
        )

    with QUERY_LOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit("The query log is empty. Ask a few questions in the app first.")

    out_rows = []
    for row in rows:
        stamp = parse_time(row.get("timestamp_ist", ""))
        mode = row.get("answer_mode", "")
        refused = mode == "no_answer"
        answered = mode in {"llm", "extractive", "llm_error"}
        document, page = most_cited_document(row.get("sources", ""))
        try:
            total_ms = float(row.get("total_ms") or 0)
        except ValueError:
            total_ms = 0.0
        attempts = row.get("agent_trace", "").count("->") or 0

        out_rows.append(
            {
                "timestamp": stamp.isoformat(timespec="seconds") if stamp else "",
                "date": stamp.date().isoformat() if stamp else "",
                "hour_of_day": stamp.hour if stamp else "",
                "question": row.get("question", ""),
                "answer_mode": mode,
                "answered": int(answered),
                "refused": int(refused),
                "num_sources": row.get("num_sources", ""),
                "source_doc": document,
                "top_page": page if page is not None else "",
                "top_distance": row.get("top_distance", ""),
                "attempts": attempts,
                "retrieval_ms": row.get("retrieval_ms", ""),
                "llm_ms": row.get("llm_ms", ""),
                "total_ms": row.get("total_ms", ""),
                "latency_band": latency_band(total_ms),
                "feedback": row.get("feedback", ""),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    answered = sum(row["answered"] for row in out_rows)
    refused = sum(row["refused"] for row in out_rows)
    documents = Counter(row["source_doc"] for row in out_rows if row["source_doc"])
    print(f"wrote {len(out_rows)} rows to {OUT.relative_to(ROOT)}")
    print(f"  answered {answered} · refused {refused} · "
          f"refusal rate {100 * refused / len(out_rows):.1f}%")
    print(f"  documents cited: {', '.join(f'{k} ({v})' for k, v in documents.most_common()) or 'none'}")
    if EVAL_RESULTS.exists():
        print(f"  evaluation results also available at {EVAL_RESULTS.relative_to(ROOT)} "
              "(second table for the report)")


if __name__ == "__main__":
    main()
