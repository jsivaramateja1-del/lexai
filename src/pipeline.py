"""The pipeline: question in, grounded answer + citations out.

Read this file top to bottom and you have the whole RAG idea:
    plan queries -> retrieve -> check relevance -> generate -> log

The retrieval step is delegated to src/agent.py, which may try more than one
search phrasing before giving up. Everything else is unchanged.
"""

from __future__ import annotations

import time

from src import agent, config, generator
from src.config import DISCLAIMER, TOP_K
from src.logger import log_query, new_query_id

NO_ANSWER = (
    "The provided documents do not cover this. I can only answer from the "
    "ingested IP documents."
)


def answer_question(
    question: str,
    history: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """Run the full RAG loop and return a dict the UI can render directly."""
    question = (question or "").strip()
    query_id = new_query_id()
    if not question:
        return _result(query_id, "Please type a question.", [], "error", "empty question")

    total_started = time.perf_counter()

    # 1. Retrieve - the agent may try several phrasings and records each attempt
    retrieve_started = time.perf_counter()
    evidence = agent.collect_evidence(question, top_k=top_k)
    hits = evidence["hits"]
    trace = evidence["trace"]
    retrieval_ms = int((time.perf_counter() - retrieve_started) * 1000)

    # 2. Relevance guard: refuse instead of guessing
    best_distance = hits[0]["distance"] if hits else 1.0
    # The agent already decided, using cosine distance AND lexical evidence.
    # Re-testing distance here would silently undo the lexical rule.
    if not evidence["accepted"] or not hits:
        result = _result(
            query_id, NO_ANSWER, hits, "no_answer", "",
            retrieval_ms=retrieval_ms, total_ms=_elapsed(total_started),
        )
        result["trace"] = trace
        log_query(_log_row(query_id, question, result, best_distance))
        return result

    # 3. Generate (LLM if configured, otherwise verbatim quotes)
    llm_ms = 0
    if generator.llm_available():
        messages = generator.build_messages(question, hits, history)
        reply = generator.call_llm(messages)
        llm_ms = int(reply["latency_s"] * 1000)
        text = (reply["text"] or "").strip()
        refused = generator.REFUSAL_MARKER in text.lower()[:120]
        if not reply["error"] and refused and evidence.get("strong"):
            # Strong evidence but a refusal: the model under-read the context.
            # One corrective attempt, because a strong lexical match means the
            # answer is present. Bounded, so this cannot loop.
            nudge_messages = messages + [
                {"role": "user", "content": generator.NUDGE}
            ]
            second = generator.call_llm(nudge_messages)
            llm_ms += int(second["latency_s"] * 1000)
            if not second["error"] and second["text"] and \
                    generator.REFUSAL_MARKER not in second["text"].lower()[:120]:
                reply = second

        if reply["error"]:
            # Distinct mode, so the UI and the log can tell "the model was never
            # called" apart from "the model was called and failed".
            answer = generator.extractive_answer(question, hits)
            mode, error = "llm_error", reply["error"]
            answer = (
                f"**Model unavailable — showing source passages instead.**\n\n"
                f"`{reply['error'][:200]}`\n\n{answer}"
            )
        else:
            answer, mode, error = reply["text"], "llm", ""
    else:
        # Quote mode: no key configured, or LLM_ENABLED is off. Cannot hallucinate.
        answer, mode, error = generator.extractive_answer(question, hits), "extractive", ""

    result = _result(
        query_id, answer, hits, mode, error,
        retrieval_ms=retrieval_ms, llm_ms=llm_ms, total_ms=_elapsed(total_started),
    )
    result["trace"] = trace
    result["trace_text"] = agent.describe_trace(trace)
    log_query(_log_row(query_id, question, result, best_distance))
    return result


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _result(
    query_id: str,
    answer: str,
    hits: list[dict],
    mode: str,
    error: str,
    retrieval_ms: int = 0,
    llm_ms: int = 0,
    total_ms: int = 0,
) -> dict:
    return {
        "query_id": query_id,
        "answer": answer,
        "mode": mode,
        "error": error,
        "sources": [
            {"source": h["source"], "page": h["page"], "score": h["score"]}
            for h in hits
        ],
        "passages": hits,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
        "disclaimer": DISCLAIMER,
    }


def _log_row(query_id: str, question: str, result: dict, best_distance: float) -> dict:
    return {
        "query_id": query_id,
        "question": question,
        "answer_mode": result["mode"],
        "num_sources": len(result["sources"]),
        "sources": " | ".join(f"{s['source']} p.{s['page']}" for s in result["sources"]),
        "top_distance": round(best_distance, 4),
        "agent_trace": agent.describe_trace(result.get("trace", []))[:400],
        "retrieval_ms": result["retrieval_ms"],
        "llm_ms": result["llm_ms"],
        "total_ms": result["total_ms"],
        "feedback": "",
        "error": result["error"][:300],
    }
