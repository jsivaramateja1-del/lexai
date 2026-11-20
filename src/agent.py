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

Why the glossary matters: a statute says "computer programme per se" where a
user says "software". Dense retrieval often bridges that gap and often does not.
Adding the statutory phrasing as one extra search costs nothing and closes a
whole class of misses.
"""

from __future__ import annotations

import re

from src import config, generator, hybrid, retriever

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
    "india", "indian", "legal", "law",
}

# User vocabulary -> the words an Indian IP statute actually uses.
# Keys are matched as whole words, case-insensitively.
GLOSSARY: dict[str, tuple[str, ...]] = {
    "software": ("computer programme per se",),
    "app": ("computer programme per se",),
    "apps": ("computer programme per se",),
    "code": ("computer programme per se",),
    "coding": ("computer programme per se",),
    "algorithm": ("computer programme per se algorithms",),
    "algorithms": ("computer programme per se algorithms",),
    "ai": ("computer programme per se",),
    "ml": ("computer programme per se",),
    "business": ("business method",),
    "formula": ("mathematical method",),
    "maths": ("mathematical method",),
    "medicine": ("pharmaceutical substance",),
    "drug": ("pharmaceutical substance",),
    "drugs": ("pharmaceutical substance",),
    "plant": ("plants and animals essentially biological processes",),
    "plants": ("plants and animals essentially biological processes",),
    "seed": ("plants and animals seeds varieties species",),
    "seeds": ("plants and animals seeds varieties species",),
    "brand": ("trade mark",),
    "brandname": ("trade mark",),
    "logo": ("trade mark",),
    "trademark": ("trade mark",),
    "trademarks": ("trade mark",),
    "copying": ("infringement",),
    "piracy": ("infringement",),
    "book": ("literary work",),
    "song": ("musical work",),
    "songs": ("musical work",),
    "movie": ("cinematograph film",),
    "film": ("cinematograph film",),
    "photo": ("artistic work",),
    "duration": ("term of copyright",),
    "expire": ("term of copyright",),
    "expiry": ("term of copyright",),
    "cost": ("fee payable",),
    "costs": ("fee payable",),
    "price": ("fee payable",),
    "renew": ("renewal fee",),
    "renewal": ("renewal fee",),
    "draft": ("provisional specification complete specification",),
    "filing": ("application for patent",),
    "file": ("application for patent",),
    "examination": ("examination of application",),
    "examiner": ("examination of application",),
    "challenge": ("opposition",),
    "dispute": ("opposition revocation",),
    "licence": ("compulsory licence",),
    "license": ("compulsory licence",),
    "generic": ("compulsory licence",),
    "novel": ("novelty inventive step",),
    "new": ("novelty inventive step",),
    "obvious": ("inventive step",),
    "unique": ("novelty",),
}

REWRITE_PROMPT = """You rewrite search queries for a legal document search engine.

Given a user question, produce up to 2 alternative search phrasings that would
find the relevant passages in an Indian statute. Use the vocabulary a statute
would use. Reply with one phrasing per line, no numbering, no explanation.

QUESTION: {question}"""


def key_terms(question: str, limit: int = 6) -> list[str]:
    """Rule-based fallback: the content words of the question, longest first.

    Works with no API key, no network, and no cost.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z-]+", question.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    seen: list[str] = []
    for word in sorted(content, key=len, reverse=True):
        if word not in seen:
            seen.append(word)
    return seen[:limit]


def statutory_terms(question: str) -> list[str]:
    """Statutory phrasings for the words the user actually used."""
    words = set(re.findall(r"[a-zA-Z][a-zA-Z-]+", question.lower()))
    phrases: list[str] = []
    for word in words:
        for phrase in GLOSSARY.get(word, ()):
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def llm_rewrites(question: str, max_variants: int = 2) -> list[str]:
    """Ask the model for alternative phrasings. Empty list if unavailable."""
    if not generator.llm_available():
        return []
    reply = generator.call_llm(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}]
    )
    if reply["error"] or not reply["text"]:
        return []
    lines = [line.strip(" -\u2022\t") for line in reply["text"].splitlines()]
    cleaned = [line for line in lines if 3 <= len(line) <= 200]
    return cleaned[:max_variants]


def plan_queries(question: str, use_llm: bool = True) -> list[dict]:
    """Propose search phrasings, cheapest and most faithful first.

    Order matters. The original question always goes first, because a rewrite can
    drift from what the user meant. The statutory phrasing comes next, since it
    costs one local search and fixes vocabulary mismatch. Key terms and model
    rewrites follow.
    """
    plan: list[dict] = [{"query": question, "source": "original"}]

    phrases = statutory_terms(question)
    if phrases:
        plan.append({"query": " ".join(phrases), "source": "statutory terms"})

    terms = key_terms(question)
    if terms:
        plan.append({"query": " ".join(terms), "source": "key terms"})

    if use_llm:
        for variant in llm_rewrites(question):
            plan.append({"query": variant, "source": "llm rewrite"})

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

    Returns {"hits", "trace", "accepted", "strong"}. `accepted` False means: refuse,
    do not answer. `strong` False with `accepted` True means the answer rests on a
    provisional match - usable, but worth showing as such.
    """
    top_k = top_k or config.TOP_K
    phrases = statutory_terms(question)

    if not config.AGENT_ENABLED:
        hits = retriever.search(question, top_k=top_k)
        level = hybrid.accept_level(hits[0], question, phrases) if hits else 0
        return {
            "hits": hits,
            "accepted": level >= 1,
            "strong": level >= 2,
            "trace": [
                {
                    "step": 1,
                    "query": question,
                    "source": "original",
                    "top_score": hits[0]["score"] if hits else None,
                    "top_page": hits[0]["page"] if hits else None,
                    "matched_by": hits[0].get("matched_by", "vector") if hits else "",
                    "level": level,
                    "accepted": level >= 1,
                }
            ],
        }

    plan = plan_queries(question, use_llm=use_llm)
    trace: list[dict] = []
    best: list[dict] = []
    best_level = 0

    for step, item in enumerate(plan, start=1):
        hits = retriever.search(item["query"], top_k=top_k)
        # Any phrasing may certify a hit by verbatim containment, including a model
        # rewrite: "how long does a patent last" is answered only because the model
        # rewrote it as "TERM OF PATENT", a section heading. The cost is that a
        # rewrite can also certify a merely on-topic passage; that trade is visible
        # in the evaluation, and precision is protected by the distance threshold.
        level = hybrid.accept_level(
            hits[0], question, phrases, attempt_query=item["query"]
        ) if hits else 0
        trace.append(
            {
                "step": step,
                "query": item["query"],
                "source": item["source"],
                "top_score": hits[0]["score"] if hits else None,
                "top_page": hits[0]["page"] if hits else None,
                "matched_by": hits[0].get("matched_by", "vector") if hits else "",
                "level": level,
                "accepted": level >= 1,
            }
        )

        if hits:
            better = level > best_level or (
                level == best_level and (not best or hits[0]["score"] > best[0]["score"])
            )
            if better:
                best, best_level = hits, level

        # Only a strong match ends the search. A provisional match is kept as a
        # fallback and the loop moves on: that is how the statutory phrasing gets
        # a chance even when the first attempt barely cleared the threshold.
        if level >= 2:
            break

    return {
        "hits": best,
        "accepted": best_level >= 1,
        "strong": best_level >= 2,
        "trace": trace,
    }


def describe_trace(trace: list[dict]) -> str:
    """One-line, human-readable summary for the UI and the CSV log."""
    parts = []
    for step in trace:
        score = f"{step['top_score']:.3f}" if step["top_score"] is not None else "n/a"
        mark = {2: "strong", 1: "provisional", 0: "rejected"}.get(step.get("level", 0), "rejected")
        via = step.get("matched_by") or "vector"
        parts.append(f"{step['source']} '{step['query'][:60]}' -> {score} via {via} {mark}")
    return " | ".join(parts)
