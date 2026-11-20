"""Evaluate retrieval and refusal quality on a fixed question set.

    python -m scripts.run_eval                 # quote mode: free, no API calls
    python -m scripts.run_eval --llm           # real answers (~2 paise per question)
    python -m scripts.run_eval --top-k 3       # stricter retrieval test

What it measures, and why these four numbers
--------------------------------------------
retrieval hit@k   did any expected page appear among the retrieved passages?
top-1             was the expected page the first one returned?
refusal accuracy  did unanswerable questions get refused, and answerable ones get answered?
citation rate     of the answered questions, how many carried a page reference?

These are deliberately separate. A system can retrieve the right page and still write a wrong
answer, or answer well from the wrong page. Collapsing them into one "accuracy" number hides
exactly the failure you need to see.

Ground truth lives in eval/questions.json. Expected pages were derived by searching the PDF for
the section wording, not chosen from what the system happened to return.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evaluation set")
    parser.add_argument("--llm", action="store_true", help="use the model (costs credits)")
    parser.add_argument("--top-k", type=int, default=None, help="passages to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--threshold", type=float, default=None,
                        help="override MAX_DISTANCE for this run")
    parser.add_argument("--sweep", default="",
                        help="comma-separated thresholds, e.g. 0.30,0.35,0.40: reports each")
    return parser.parse_args()


def sweep(questions: list[dict], thresholds: list[float], top_k: int, config) -> None:
    """Re-run the whole set at several thresholds and report the trade-off.

    The refusal threshold is a policy choice, not a fact: lowering it answers more
    questions and invents more answers. This prints both sides so the choice is
    made from the evaluation set instead of from taste.
    """
    from src.pipeline import answer_question

    print(f"{'thresh':<8}{'hit@k':<8}{'top-1':<8}{'answered':<10}{'refused ok':<12}{'false answers':<15}")
    print("-" * 62)
    for threshold in thresholds:
        config.MAX_DISTANCE = threshold
        config.STRONG_DISTANCE = max(0.05, threshold - 0.07)

        hits = top1 = answered = refused_ok = false_answers = 0
        answerable = [q for q in questions if q["answerable"]]
        unanswerable = [q for q in questions if not q["answerable"]]

        for item in answerable:
            result = answer_question(item["question"], top_k=top_k)
            pages = [hit["page"] for hit in result["passages"]]
            if set(pages) & set(item["expected_pages"]):
                hits += 1
            if pages and pages[0] in item["expected_pages"]:
                top1 += 1
            if result["mode"] != "no_answer":
                answered += 1
        for item in unanswerable:
            result = answer_question(item["question"], top_k=top_k)
            if result["mode"] == "no_answer":
                refused_ok += 1
            else:
                false_answers += 1

        print(f"{threshold:<8.2f}{100 * hits / max(1, len(answerable)):<8.1f}"
              f"{100 * top1 / max(1, len(answerable)):<8.1f}"
              f"{answered}/{len(answerable):<8}"
              f"{refused_ok}/{len(unanswerable):<10}"
              f"{false_answers:<15}")
    print("\nPick the threshold where 'answered' is high and 'false answers' is still 0.")


def main() -> None:
    args = parse_args()

    # Set before importing config: quote mode is the default so an evaluation run
    # never spends credits unless asked to.
    os.environ["LLM_ENABLED"] = "true" if args.llm else "false"

    from src.config import TOP_K
    from src.pipeline import answer_question
    from src import retriever

    import src.config as config

    if args.threshold is not None:
        config.MAX_DISTANCE = args.threshold
        config.STRONG_DISTANCE = max(0.05, args.threshold - 0.07)

    top_k = args.top_k or TOP_K
    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    try:
        indexed = retriever.stats()["chunks"]
    except Exception:                                            # noqa: BLE001
        indexed = 0
    if indexed == 0:
        sys.exit("The vector store is empty. Run: python -m scripts.ingest --reset")

    if args.sweep:
        sweep(questions, [float(t) for t in args.sweep.split(",") if t.strip()], top_k, config)
        return

    print(f"Mode: {'LLM answers' if args.llm else 'quote mode'} · top_k={top_k} · "
          f"{len(questions)} questions · {indexed} chunks indexed\n")
    print(f"{'id':<5}{'pages':<16}{'hit':<5}{'top1':<6}{'refused':<9}{'ms':<7}question")
    print("-" * 92)

    rows = []
    for item in questions:
        started = time.perf_counter()
        result = answer_question(item["question"], top_k=top_k)
        elapsed = int((time.perf_counter() - started) * 1000)

        pages = [hit["page"] for hit in result["passages"]]
        expected = item.get("expected_pages", [])
        hit = bool(set(pages) & set(expected)) if expected else False
        top1 = bool(pages) and pages[0] in expected
        refused = result["mode"] == "no_answer"
        cited = any(f"p.{page}" in result["answer"] for page in pages) or "p." in result["answer"]

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "source": item.get("source", ""),
                "answerable": item["answerable"],
                "expected_pages": " ".join(str(p) for p in expected),
                "retrieved_pages": " ".join(str(p) for p in pages),
                "retrieval_hit": int(hit),
                "top1_correct": int(top1),
                "refused": int(refused),
                "mode": result["mode"],
                "cited": int(cited),
                "best_distance": round(result["passages"][0]["distance"], 3) if result["passages"] else "",
                "best_score": round(result["passages"][0]["score"], 3) if result["passages"] else "",
                "strong": int(bool(result.get("strong"))),
                "attempts": len(result.get("trace", [])),
                "latency_ms": elapsed,
            }
        )

        print(f"{item['id']:<5}{' '.join(str(p) for p in expected)[:14]:<16}"
              f"{'yes' if hit else '-':<5}{'yes' if top1 else '-':<6}"
              f"{'yes' if refused else '-':<9}{elapsed:<7}{item['question'][:44]}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------ summary
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    hit_rate = 100 * sum(r["retrieval_hit"] for r in answerable) / max(1, len(answerable))
    top1_rate = 100 * sum(r["top1_correct"] for r in answerable) / max(1, len(answerable))
    answered_ok = sum(1 for r in answerable if not r["refused"])
    refused_ok = sum(1 for r in unanswerable if r["refused"])
    refusal_accuracy = 100 * (answered_ok + refused_ok) / max(1, len(rows))
    citation_rate = 100 * sum(r["cited"] for r in rows if not r["refused"]) / max(
        1, sum(1 for r in rows if not r["refused"])
    )
    latencies = [r["latency_ms"] for r in rows]

    print("\n" + "=" * 52)
    print(f"retrieval hit@{top_k}        {hit_rate:5.1f}%   ({len(answerable)} answerable questions)")
    print(f"top-1 correct          {top1_rate:5.1f}%")
    print(f"refusal accuracy       {refusal_accuracy:5.1f}%   "
          f"({answered_ok}/{len(answerable)} answered, {refused_ok}/{len(unanswerable)} refused)")
    print(f"citations present      {citation_rate:5.1f}%")
    print(f"median latency         {statistics.median(latencies):5.0f} ms")
    print(f"mean attempts per question {statistics.mean(r['attempts'] for r in rows):4.1f}")
    print("=" * 52)
    print(f"\nPer-question detail written to {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
