"""Unit tests for the AI-free part of the pipeline: cleaning and chunking.

Run with:  pytest -q        (or: python -m pytest tests -q)
"""

from src.ingestor import chunk_text, clean_text


def test_clean_removes_wrapped_hyphen():
    assert "constitution" in clean_text("the consti-\ntution of India")


def test_clean_collapses_spaces_and_blank_lines():
    assert clean_text("a    b\n\n\n\nc") == "a b\n\nc"


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(12))
    chunks = chunk_text(text, size=400, overlap=60)
    assert chunks
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_chunks_overlap():
    text = "\n\n".join(f"Sentence number {i} " + "filler " * 40 for i in range(10))
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) > 1
    # the tail of chunk 1 should reappear near the start of chunk 2
    tail_words = chunks[0].split()[-4:]
    assert " ".join(tail_words) in chunks[1]


def test_no_text_lost():
    text = "alpha " * 300
    chunks = chunk_text(text, size=300, overlap=50)
    assert "alpha" in chunks[0] and "alpha" in chunks[-1]


def test_bad_parameters_rejected():
    for size, overlap in [(0, 0), (100, 100), (100, 200)]:
        try:
            chunk_text("x " * 100, size=size, overlap=overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for size={size}, overlap={overlap}")
