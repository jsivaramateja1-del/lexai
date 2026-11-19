"""Stage 3: store vectors and search them, with ChromaDB.

ChromaDB is a local vector database: a folder on disk that answers
"give me the k vectors closest to this one" quickly.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
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


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the closest chunks as dicts, nearest first.

    'distance' is cosine distance: 0.0 = identical meaning, 1.0 = unrelated.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
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
            }
        )
    return hits


def stats() -> dict:
    collection = get_collection()
    return {"chunks": collection.count(), "collection": COLLECTION_NAME}


def reset() -> None:
    """Delete every stored chunk. Used when you rebuild the knowledge base."""
    get_client().delete_collection(COLLECTION_NAME)
    get_collection.cache_clear() if hasattr(get_collection, "cache_clear") else None
