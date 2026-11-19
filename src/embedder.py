"""Stage 2: text -> vectors, using a free local model.

One model is used for BOTH documents and questions. If you embedded documents
with model A and questions with model B, the two would live in different
"coordinate systems" and every similarity score would be meaningless.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    """Load the model once per process. Loading takes seconds; keep it cached."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed many chunks at once (used during ingestion)."""
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit length => cosine == dot product
    )
    return [vector.tolist() for vector in vectors]


def embed_query(question: str) -> list[float]:
    """Embed a single user question (used at search time)."""
    return embed_texts([question])[0]


def embedding_dimension() -> int:
    model = get_model()
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())
