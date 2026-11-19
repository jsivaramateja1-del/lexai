"""End-to-end smoke test with a fake embedder (no model download, no API key).

Proves the wiring works: PDF -> chunks -> vector store -> search -> pipeline.
Run:  python tests/smoke_test.py
"""

import hashlib
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

# --- 1. a tiny fake embedding model: bag-of-words hashed into 64 dimensions ---
import src.embedder as embedder  # noqa: E402

DIM = 64


def fake_embed(text: str) -> list[float]:
    vector = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


embedder.embed_texts = lambda texts, batch_size=32: [fake_embed(t) for t in texts]
embedder.embed_query = lambda q: fake_embed(q)

import src.retriever as retriever  # noqa: E402
import src.config as config  # noqa: E402

retriever.embed_texts = embedder.embed_texts
retriever.embed_query = embedder.embed_query
config.CHROMA_DIR = Path(tempfile.mkdtemp()) / "chroma"
retriever.CHROMA_DIR = config.CHROMA_DIR
retriever.get_client.cache_clear()

from src.ingestor import ingest_pdf  # noqa: E402
import src.pipeline as pipeline  # noqa: E402

# --- 2. build a two-page PDF that looks like a statute ---
pdf_path = Path(tempfile.mkdtemp()) / "Patents_Act_sample.pdf"
doc = pymupdf.open()
page1 = doc.new_page()
page1.insert_text(
    (60, 80),
    "CHAPTER I\n\n1. Short title. This Act may be called the Patents Act, 1970.\n\n"
    "2. Definitions. In this Act, invention means a new product or process "
    "involving an inventive step and capable of industrial application.",
    fontsize=11,
)
page2 = doc.new_page()
page2.insert_text(
    (60, 80),
    "3. What are not inventions. An invention which is frivolous, or which claims "
    "anything contrary to well established natural laws, is not patentable.\n\n"
    "6. Application for patent. An application for a patent may be made by any "
    "person claiming to be the true and first inventor.",
    fontsize=11,
)
doc.save(pdf_path)
doc.close()

chunks = ingest_pdf(pdf_path)
print(f"chunks: {len(chunks)}  pages: {sorted({c.page for c in chunks})}")
assert len(chunks) >= 2, "expected at least one chunk per page"
assert all(c.source == pdf_path.name for c in chunks)

stored = retriever.add_chunks(chunks)
print("stored:", stored, retriever.stats())

# --- 3. search ---
hits = retriever.search("what is not patentable", top_k=2)
print("top hit:", hits[0]["source"], "p.", hits[0]["page"], "score", hits[0]["score"])
assert hits and "frivolous" in hits[0]["text"].lower(), "retrieval did not find the right page"

# --- 4. full pipeline in quote mode (no LLM) ---
pipeline.retriever = retriever
result = pipeline.answer_question("what is not patentable")
print("mode:", result["mode"], "| sources:", result["sources"])
assert result["mode"] == "extractive"
assert result["sources"] and result["sources"][0]["page"] == 2

# --- 5. refusal path: a question with no match in the corpus ---
pipeline.MAX_DISTANCE = -1.0     # force every distance to look too far
refused = pipeline.answer_question("what is the capital of France")
print("refusal mode:", refused["mode"])
assert refused["mode"] == "no_answer"

# --- 6. the log file ---
log = Path(config.QUERY_LOG)
print("log rows:", max(0, len(log.read_text().splitlines()) - 1) if log.exists() else 0)
assert log.exists()

print("\nSMOKE TEST PASSED")
