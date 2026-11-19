"""Build the knowledge base. Run from the project root:

    python -m scripts.ingest            # add/update every PDF in data/documents
    python -m scripts.ingest --reset    # wipe the store and rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retriever                       # noqa: E402
from src.config import DATA_DIR                 # noqa: E402
from src.ingestor import ingest_pdf             # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest IP documents into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="delete existing chunks first")
    args = parser.parse_args()

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}. Add documents first.")
        return

    if args.reset:
        print("Resetting the vector store ...")
        retriever.reset()

    started = time.perf_counter()
    total = 0
    for pdf in pdfs:
        try:
            chunks = ingest_pdf(pdf)
        except ValueError as exc:
            print(f"SKIP {pdf.name}: {exc}")
            continue
        stored = retriever.add_chunks(chunks)
        total += stored
        pages = len({c.page for c in chunks})
        print(f"OK   {pdf.name}: {pages} pages -> {stored} chunks")

    print(f"\nDone: {total} chunks in {time.perf_counter() - started:.1f}s")
    print(f"Store now holds {retriever.stats()['chunks']} chunks.")


if __name__ == "__main__":
    main()
