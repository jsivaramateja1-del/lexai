"""Stage 1: PDF -> clean text -> overlapping chunks.

Deliberately contains NO AI. Text extraction and splitting are pure string
work, which means this whole file can be tested without a model or an API key.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Amendment footnotes and bare page numbers carry no legal meaning but are full of
# words like "patentable" and "omitted", so they win similarity matches and then
# mislead the model. Drop those lines before chunking.
_FOOTNOTE = re.compile(
    r"\b(Ins\.|Subs\.|Omitted|Added|Rep\.|w\.e\.f\.)\b|by Act \d+ of \d+",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    """One retrievable unit of text, plus everything needed to cite it."""

    chunk_id: str
    doc_id: str
    source: str      # file name shown to the user in the citation
    page: int        # 1-based page number in the original PDF
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- extraction
def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, raw_text), ...] for a digital PDF.

    A scanned PDF returns empty strings here. Detect that early: if most pages
    come back empty, the file needs OCR before it is usable.
    """
    pages: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            pages.append((number, page.get_text("text")))
    return pages


def looks_scanned(pages: list[tuple[int, str]], min_chars: int = 50) -> bool:
    """True when fewer than half the pages contain real text."""
    if not pages:
        return True
    empty = sum(1 for _, text in pages if len(text.strip()) < min_chars)
    return empty > len(pages) / 2


# ---------------------------------------------------------------- cleaning
def clean_text(raw: str) -> str:
    """Fix the artefacts that break retrieval, without changing wording.

    - de-hyphenate words wrapped across lines  ("consti-\ntution" -> "constitution")
    - collapse runs of spaces and blank lines
    - drop amendment footnotes and bare page numbers, which are noise for search
    - strip every line so stray indentation does not become fake paragraphs
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")                    # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)             # wrapped words
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    kept = [
        line for line in lines
        if not (line and (line.isdigit() or _FOOTNOTE.search(line)))
    ]
    return _MULTI_NEWLINE.sub("\n\n", "\n".join(kept)).strip()


_JUNK_LINE = re.compile(r"^[\s*\u2014\u2013.,;:()\[\]-]{0,12}$")


def _strip_leading_junk(chunk: str) -> str:
    """Drop leading lines that carry no text: footnote stars, stray punctuation.

    A chunk that begins with "* * *" embeds a little worse and cites badly.
    """
    lines = chunk.split("\n")
    while lines and _JUNK_LINE.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- chunking
def _tail(text: str, overlap: int) -> str:
    """Last ~overlap characters of text, cut at a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Cut a paragraph longer than one chunk, at a boundary a human would choose.

    Cutting at an arbitrary character splits words ("programme pe" / "r se") and
    destroys exact phrases, which is fatal for legal text where the phrase is the
    meaning. So: prefer a clause break, then a sentence break, then a space.
    """
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            cut = max(window.rfind("; "), window.rfind(". "), window.rfind(" "))
            if cut > size * 0.7:            # only accept a boundary near the end
                end = start + cut + 1
            else:
                # otherwise extend to the next space, so no word is ever cut in two
                following = text.find(" ", end)
                if following != -1 and following - end <= 40:
                    end = following
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        # Always move forward. If the remaining text is shorter than the overlap,
        # stepping back by `overlap` would not advance, and the loop would emit a
        # staircase of ever-shorter pieces (150, 149, 148 ... 1). Step to the end.
        next_start = end - overlap
        start = next_start if next_start > start else end
    return pieces


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph and word boundaries.

    Three rules, all learned the hard way:
      1. Prefer ending a chunk where a paragraph ends, so a chunk carries one idea.
      2. Repeat the tail of each chunk at the start of the next, so a sentence cut
         at a boundary is still complete somewhere.
      3. NEVER slice at an arbitrary character. A cut inside a word destroys the
         phrase that carries the legal meaning ("computer programme per se" once
         became "computer programme pe" + "r se or algorithms", and the passage
         became unfindable by both vector and keyword search).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(_hard_split(paragraph, size, overlap) if len(paragraph) > size else [paragraph])

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if not buffer:
            buffer = unit
            continue

        candidate = f"{buffer}\n\n{unit}"
        if len(candidate) <= size:
            buffer = candidate
            continue

        chunks.append(buffer.strip())

        # Carry a word-aligned tail into the next chunk, but leave room for the
        # incoming unit so the new buffer can never exceed `size`. This is what
        # keeps the overlap from being bought with an arbitrary slice later.
        room = max(0, size - len(unit) - 2)
        tail = _tail(buffer, min(overlap, room))
        buffer = f"{tail}\n\n{unit}".strip() if tail else unit

    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in (_strip_leading_junk(chunk) for chunk in chunks) if c]


# ------------------------------------------------------------------ ingest
def ingest_pdf(pdf_path: str | Path) -> list[Chunk]:
    """PDF file -> list of Chunk objects ready for embedding."""
    pdf_path = Path(pdf_path)
    pages = extract_pages(pdf_path)
    if looks_scanned(pages):
        raise ValueError(
            f"{pdf_path.name} looks scanned (little or no extractable text). "
            "Run OCR first, then ingest it again."
        )

    doc_id = hashlib.sha1(pdf_path.name.encode()).hexdigest()[:12]
    chunks: list[Chunk] = []
    for page_number, raw in pages:
        for index, piece in enumerate(chunk_text(clean_text(raw))):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page_number}-{index}",
                    doc_id=doc_id,
                    source=pdf_path.name,
                    page=page_number,
                    text=piece,
                )
            )
    return chunks


def ingest_folder(folder: str | Path) -> list[Chunk]:
    """Every .pdf in a folder -> one flat list of chunks."""
    folder = Path(folder)
    all_chunks: list[Chunk] = []
    for pdf in sorted(folder.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf))
    return all_chunks
