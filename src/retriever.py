"""Stage 3: store vectors and search them, with ChromaDB.

ChromaDB is a local vector database: a folder on disk that answers
"give me the k vectors closest to this one" quickly.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import CHROMA_DIR, COLLECTION_NAME, HYBRID_ENABLED, TOP_K
from src.embedder import embed_query, embed_texts
from src.ingestor import Chunk


@lru_cache(maxsize=1)
def get_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """The collection is created on first use with cosine distance."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """Embed and store chunks. Re-ingesting the same PDF replaces its chunks."""
    collection = get_collection()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed_texts([c.text for c in batch]),
            metadatas=[
                {"doc_id": c.doc_id, "source": c.source, "page": c.page}
                for c in batch
            ],
        )
    return len(chunks)


def search(question: str, top_k: int = TOP_K, hybrid: bool | None = None) -> list[dict]:
    """Return the closest chunks as dicts, nearest first.

    'distance' is cosine distance: 0.0 = identical meaning, 1.0 = unrelated.

    With hybrid retrieval on (the default) the vector ranking is fused with a
    keyword ranking, because exact statutory wording is invisible to vectors.
    Every returned item carries 'matched_by' so you can see which system found it.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = embed_query(question)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(top_k * 3, top_k), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "chunk_id": result["ids"][0][len(hits)],
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": int(meta.get("page", 0)),
                "distance": float(distance),
                "score": round(1.0 - float(distance), 3),
                "matched_by": "vector",
            }
        )

    use_hybrid = HYBRID_ENABLED if hybrid is None else hybrid
    if not use_hybrid:
        return hits[:top_k]

    from src import hybrid as hybrid_mod

    keyword_hits = hybrid_mod.keyword_search(question, top_n=max(top_k * 3, 12))
    fused = hybrid_mod.fuse(hits, keyword_hits)
    fused = hybrid_mod.prefer_exact_phrase(fused, question)[:top_k]

    # Keyword-only hits have no cosine distance yet; compute it so the guard and
    # the UI keep one comparable number.
    for item in fused:
        if item.get("distance") is None:
            vector = embed_texts([item["text"]])[0]
            similarity = sum(a * b for a, b in zip(query_vector, vector))
            item["distance"] = float(1.0 - similarity)
            item["score"] = round(similarity, 3)
    return fused


def stats() -> dict:
    collection = get_collection()
    return {"chunks": collection.count(), "collection": COLLECTION_NAME}


def reset() -> None:
    """Delete every stored chunk. Used when you rebuild the knowledge base."""
    get_client().delete_collection(COLLECTION_NAME)
    get_collection.cache_clear() if hasattr(get_collection, "cache_clear") else None
