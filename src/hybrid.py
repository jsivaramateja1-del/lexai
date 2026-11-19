"""Hybrid retrieval: keyword evidence beside vector similarity.

Why this exists
---------------
Dense vectors are good at meaning and bad at exactness. Ask "can I patent
software" and the vector search returns pages about patent applications, because
the *topic* is patents. Ask for the exact statutory phrase "computer programme
per se" and it still returns the wrong page, because a 900-character chunk full
of a dozen other clauses has a diluted vector.

A keyword index has the opposite bias: it is blind to meaning, precise about
words. The phrase "computer programme per se" appears on exactly one page of the
Act, and a term-frequency index finds it instantly.

So this module does two things:

1. `keyword_search` - a TF-IDF index over the stored chunks, rebuilt whenever the
   collection size changes.
2. `fuse` - Reciprocal Rank Fusion, which merges two ranked lists without
   pretending their scores are comparable. RRF uses ranks, not scores, so a
   cosine distance and a TF-IDF weight never have to be put on one scale.

Plus `lexical_evidence`, a guard: if a retrieved passage literally contains the
statutory wording the user's question maps to, that is stronger evidence than a
mediocre cosine score, and the pipeline is allowed to accept it.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src import config

_TOKEN = re.compile(r"[a-z][a-z-]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "about", "and", "or", "but", "if", "then",
    "than", "that", "this", "these", "those", "what", "which", "who", "when",
    "where", "why", "how", "do", "does", "did", "can", "could", "should",
    "would", "will", "shall", "may", "i", "we", "you", "it", "they", "my",
    "our", "your", "there", "here", "any", "some", "all", "not", "no", "yes",
    "please", "tell", "explain", "mean", "means", "according", "india", "indian",
}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


# --------------------------------------------------------------------- tf-idf
@lru_cache(maxsize=4)
def _build_index(signature: tuple[int, str]):
    """Fit a TF-IDF index over the corpus. Cached; rebuilt when the corpus changes.

    `signature` is (chunk count, first chunk id) - cheap to compute and enough to
    notice that the collection was rebuilt.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    from src import retriever

    collection = retriever.get_collection()
    data = collection.get(include=["documents", "metadatas"])
    texts = data["documents"]
    ids = data["ids"]
    metas = data["metadatas"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),      # trigrams matter: "computer programme per se"
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, ids, texts, metas


def _signature() -> tuple[int, str] | None:
    from src import retriever

    collection = retriever.get_collection()
    count = collection.count()
    if count == 0:
        return None
    first = collection.get(limit=1, include=[])["ids"]
    return (count, first[0] if first else "")


def keyword_search(question: str, top_n: int = 12) -> list[dict]:
    """Rank chunks by TF-IDF similarity to the question. Empty if no corpus."""
    signature = _signature()
    if signature is None:
        return []

    vectorizer, matrix, ids, texts, metas = _build_index(signature)
    query = vectorizer.transform([question])
    scores = (matrix @ query.T).toarray().ravel()

    ranked = scores.argsort()[::-1][:top_n]
    hits = []
    for position in ranked:
        if scores[position] <= 0:
            continue
        hits.append(
            {
                "chunk_id": ids[position],
                "text": texts[position],
                "source": metas[position].get("source", "unknown"),
                "page": int(metas[position].get("page", 0)),
                "keyword_score": float(scores[position]),
            }
        )
    return hits


# ----------------------------------------------------------------------- fusion
def fuse(vector_hits: list[dict], keyword_hits: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked lists.

    RRF adds 1/(k + rank) per list, so an item ranked first in both lists wins,
    and an item that appears in only one list still surfaces. Scores from the two
    systems are never compared, which is the whole point.
    """
    merged: dict[str, dict] = {}
    for hits, label in ((vector_hits, "vector"), (keyword_hits, "keyword")):
        for rank, hit in enumerate(hits, start=1):
            entry = merged.setdefault(hit["chunk_id"], dict(hit, sources=set(), rrf=0.0))
            entry["rrf"] += 1.0 / (k + rank)
            entry["sources"].add(label)
            # keep the richer fields whichever list supplied them
            for field in ("text", "source", "page", "distance", "score", "keyword_score"):
                if field in hit and field not in entry:
                    entry[field] = hit[field]

    fused = sorted(merged.values(), key=lambda item: item["rrf"], reverse=True)
    for item in fused:
        item["matched_by"] = "+".join(sorted(item.pop("sources")))
        item["rrf"] = round(item["rrf"], 5)
    return fused


# ------------------------------------------------------------- lexical evidence
def lexical_evidence(text: str, phrases: list[str], question: str,
                     min_coverage: float = 0.75) -> float:
    """How strongly a passage matches the question lexically, 0.0 to 1.0.

    Two signals:
      - a statutory phrase the question maps to appears verbatim (strongest)
      - a large share of the question's content words appear in the passage

    Returns 1.0 for a verbatim statutory phrase, otherwise the coverage ratio,
    capped at 0.9 so a verbatim match always outranks partial coverage.
    """
    haystack = text.lower()
    for phrase in phrases:
        if phrase and phrase.lower() in haystack:
            return 1.0

    words = tokens(question)
    if not words:
        return 0.0
    present = sum(1 for word in words if word in haystack)
    return min(0.9, present / len(words))


def should_accept(hit: dict, question: str, phrases: list[str]) -> bool:
    """The relevance guard, with lexical evidence allowed to rescue a weak score.

    Semantic similarity is the default test. A verbatim statutory phrase is
    accepted as well, because exact wording in a legal document is stronger
    evidence than a mediocre cosine score - and this is precisely the case that
    pure vector search gets wrong.
    """
    if hit is None:
        return False
    distance = hit.get("distance")
    if distance is not None and distance <= config.MAX_DISTANCE:
        return True
    if not config.LEXICAL_ACCEPT:
        return False
    return lexical_evidence(hit.get("text", ""), phrases, question) >= 0.999
