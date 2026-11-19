"""The agentic step: plan, act, observe, decide.

A plain RAG pipeline does exactly one search and then answers with whatever came
back. This module gives the system a small decision loop instead:

  1. PLAN     - propose several search phrasings for the same question
  2. ACT      - retrieve with the first phrasing
  3. OBSERVE  - is the best passage close enough to trust?
  4. RE-ACT   - if not, try the next phrasing; stop as soon as one is good enough

The loop is bounded (a fixed number of attempts) and every decision is recorded
in a trace, so the UI and the query log can show what the system did rather than
hiding it. Bounded and auditable is the difference between an agent and a retry
storm that burns API credits.

Nothing here calls the LLM unless a rewrite is genuinely needed and a key exists,
so the loop runs free during development.
"""

from __future__ import annotations

import re

from src import config, generator, retriever

# Words that carry no retrieval value on their own.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in",
    "on", "at", "to", "for", "from", "by", "with", "about", "into", "over", "under",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall", "may",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "my", "our", "your",
    "there", "here", "any", "some", "all", "not", "no", "yes", "please", "tell",
    "explain", "mean", "means", "meaning", "say", "says", "according", "act",
}

REWRITE_PROMPT = """You rewrite search queries for a legal document search engine.

Given a user question, produce up to 2 alternative search phrasings that would
find the relevant passages in an Indian statute. Use the vocabulary a statute
would use. Reply with one phrasing per line, no numbering, no explanation.

QUESTION: {question}"""


def key_terms(question: str, limit: int = 6) -> list[str]:
    """Rule-based fallback: the content words of the question, longest first.

    Works with no API key, no network, and no cost. This is what runs in
    development and in tests.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z-]+", question.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    # de-duplicate while keeping the longest, most specific words
    seen: list[str] = []
    for word in sorted(content, key=len, reverse=True):
        if word not in seen:
            seen.append(word)
    return seen[:limit]


def llm_rewrites(question: str, max_variants: int = 2) -> list[str]:
    """Ask the model for alternative phrasings. Empty list if unavailable."""
    if not generator.llm_available():
        return []
    reply = generator.call_llm(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}]
    )
    if reply["error"] or not reply["text"]:
        return []
    lines = [line.strip(" -•\t") for line in reply["text"].splitlines()]
    cleaned = [line for line in lines if 3 <= len(line) <= 200]
    return cleaned[:max_variants]


def plan_queries(question: str, use_llm: bool = True) -> list[dict]:
    """Propose search phrasings, cheapest and most faithful first.

    The original question always goes first: rewriting is a fallback, not a
    replacement, because a rewrite can drift away from what the user meant.
    """
    plan: list[dict] = [{"query": question, "source": "original"}]

    terms = key_terms(question)
    if terms:
        plan.append({"query": " ".join(terms), "source": "key terms"})

    if use_llm:
        for variant in llm_rewrites(question):
            plan.append({"query": variant, "source": "llm rewrite"})

    # keep order, drop duplicates, respect the attempt budget
    unique: list[dict] = []
    seen: set[str] = set()
    for item in plan:
        key = item["query"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:config.MAX_ATTEMPTS]


def collect_evidence(
    question: str,
    top_k: int | None = None,
    use_llm: bool = True,
) -> dict:
    """Run the loop and return the best passages plus a trace of what happened.

    Returns {"hits": [...], "trace": [...], "accepted": bool}.
    An empty hits list with accepted False means: refuse, do not answer.
    """
    top_k = top_k or config.TOP_K
    if not config.AGENT_ENABLED:
        hits = retriever.search(question, top_k=top_k)
        return {
            "hits": hits,
            "accepted": bool(hits) and hits[0]["distance"] <= config.MAX_DISTANCE,
            "trace": [{"step": 1, "query": question, "source": "original",
                       "top_score": hits[0]["score"] if hits else None,
                       "top_page": hits[0]["page"] if hits else None,
                       "accepted": bool(hits) and hits[0]["distance"] <= config.MAX_DISTANCE}],
        }

    plan = plan_queries(question, use_llm=use_llm)
    trace: list[dict] = []
    best: dict | None = None

    for step, item in enumerate(plan, start=1):
        hits = retriever.search(item["query"], top_k=top_k)
        accepted = bool(hits) and hits[0]["distance"] <= config.MAX_DISTANCE
        trace.append(
            {
                "step": step,
                "query": item["query"],
                "source": item["source"],
                "top_score": hits[0]["score"] if hits else None,
                "top_page": hits[0]["page"] if hits else None,
                "accepted": accepted,
            }
        )
        if accepted:
            best = hits
            break
        # keep the closest result seen so far in case nothing is accepted
        if hits and (best is None or hits[0]["score"] > best[0]["score"]):
            best = hits

    return {
        "hits": best or [],
        "accepted": any(step["accepted"] for step in trace),
        "trace": trace,
    }


def describe_trace(trace: list[dict]) -> str:
    """One-line, human-readable summary for the UI and the CSV log."""
    parts = []
    for step in trace:
        score = f"{step['top_score']:.3f}" if step["top_score"] is not None else "n/a"
        mark = "accepted" if step["accepted"] else "rejected"
        parts.append(f"{step['source']} '{step['query'][:60]}' -> {score} {mark}")
    return " | ".join(parts)
