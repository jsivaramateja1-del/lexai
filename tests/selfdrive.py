"""Test drive: exercise the whole system end to end, then push on the edges.

Run:  python tests/selfdrive.py

Deliberately not named test_* so pytest does not collect it: it builds a real
index in a temporary folder, renders the Streamlit app and takes a few seconds.
Every check prints PASS or FAIL, and the exit code is non-zero if anything fails.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------- fake embedder
DIM = 96


def fake_embed(text: str) -> list[float]:
    """Deterministic bag-of-words vector: enough to exercise the plumbing."""
    vector = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


import src.embedder as embedder  # noqa: E402
import src.config as config  # noqa: E402

embedder.embed_texts = lambda texts, batch_size=32: [fake_embed(t) for t in texts]
embedder.embed_query = lambda q: fake_embed(q)

import src.retriever as retriever  # noqa: E402
import src.hybrid as hybrid  # noqa: E402
import src.agent as agent  # noqa: E402
import src.pipeline as pipeline  # noqa: E402
import src.generator as generator  # noqa: E402
import src.logger as logger  # noqa: E402
from src.ingestor import chunk_text, clean_text, ingest_pdf  # noqa: E402

retriever.embed_texts = embedder.embed_texts
retriever.embed_query = embedder.embed_query

TMP = Path(tempfile.mkdtemp())
config.CHROMA_DIR = TMP / "chroma"
config.QUERY_LOG = TMP / "queries.csv"
config.ANALYTICS_DIR = TMP
retriever.CHROMA_DIR = config.CHROMA_DIR
retriever.get_client.cache_clear()
logger.QUERY_LOG = config.QUERY_LOG
logger.ANALYTICS_DIR = config.ANALYTICS_DIR

# Thresholds first: the agent's behaviour depends on them, so set them the way the
# project's .env does before testing any decision logic.
config.MAX_DISTANCE = 0.35
config.STRONG_DISTANCE = 0.28

# ------------------------------------------------------------------ 1. chunking
PDF = Path("data/documents/Patents_Act_1970.pdf")
doc = pymupdf.open(PDF)
page_chunks = []
for page_no, page in enumerate(doc, start=1):
    for piece in chunk_text(clean_text(page.get_text())):
        page_chunks.append((page_no, piece))

lengths = [len(c) for _, c in page_chunks]
check("chunks are never larger than the configured size", max(lengths) <= config.CHUNK_SIZE,
      f"max {max(lengths)}")
# The source PDF has a few column-overflow joins of its own ("th\ne applicant"), so
# this counts them rather than forbidding them: a handful out of hundreds is data
# quality, not a slicing bug.
odd = [(p, c[:40]) for p, c in page_chunks if not (c[0].isalnum() or c[0] in "([\"'")]
check("chunk starts are word-aligned except a few source artefacts", len(odd) <= 3,
      f"{len(odd)} odd starts, e.g. {odd[:2]}")
check("no chunk is an empty fragment", min(lengths) > 0)
intact = sum("computer programme per se" in c for _, c in page_chunks)
check("the statutory phrase survives chunking exactly once", intact == 1, f"found {intact}")
check("Section 3(k) is reachable on page 10",
      any(p == 10 and "computer programme per se" in c for p, c in page_chunks))

# --------------------------------------------------------------- 2. bad input
for name, kwargs in [("size=0", {"size": 0, "overlap": 0}),
                     ("overlap == size", {"size": 100, "overlap": 100}),
                     ("overlap > size", {"size": 100, "overlap": 200})]:
    try:
        chunk_text("x " * 100, **kwargs)
        check(f"chunk_text rejects {name}", False)
    except ValueError:
        check(f"chunk_text rejects {name}", True)

check("clean_text handles CRLF and soft hyphens",
      "\u00ad" not in clean_text("consti\u00adtution\r\nlaw") and "constitution" in clean_text("consti\u00adtution\r\nlaw"))
check("clean_text keeps Devanagari intact", "पेटेंट" in clean_text("पेटेंट अधिनियम"))
check("chunk_text handles Devanagari without crashing",
      len(chunk_text("पेटेंट धारा " * 200, size=300, overlap=50)) > 1)

blank = TMP / "blank.pdf"
d = pymupdf.open()
d.new_page()
d.save(blank)
d.close()
try:
    ingest_pdf(blank)
    check("a text-free PDF is rejected, not silently indexed", False)
except ValueError:
    check("a text-free PDF is rejected, not silently indexed", True)

# ----------------------------------------------------------------- 3. indexing
chunks = ingest_pdf(PDF)
stored = retriever.add_chunks(chunks)
check("ingest returns chunks", len(chunks) > 300, f"{len(chunks)}")
check("store count matches ingest", retriever.stats()["chunks"] == len(chunks))
retriever.add_chunks(chunks)
check("re-ingesting upserts instead of duplicating", retriever.stats()["chunks"] == len(chunks))

# -------------------------------------------------------------- 4. retrieval
hits = retriever.search("what is not patentable", top_k=5)
check("search returns results", len(hits) == 5)
check("every hit carries page, source and distance",
      all(h["page"] > 0 and h["source"].endswith(".pdf") and h["distance"] is not None for h in hits))
check("hybrid marks which system matched", all(h.get("matched_by") for h in hits))

kw = hybrid.keyword_search("computer programme per se", top_n=3)
check("keyword search finds the statutory phrase chunk first",
      bool(kw) and "computer programme per se" in kw[0]["text"], str(kw[0]["page"]) if kw else "none")

empty_hits = retriever.search("", top_k=3)
check("an empty question does not crash retrieval", isinstance(empty_hits, list))

# ------------------------------------------------------------------ 5. agent
evidence = agent.collect_evidence("can I patent software", use_llm=False)
check("the agent tries more than one phrasing", len(evidence["trace"]) >= 2,
      f"{len(evidence['trace'])} attempts")
check("'can I patent software' is accepted", evidence["accepted"] is True)
check("it is accepted on strong evidence, not a borderline score", evidence["strong"] is True,
      f"trace {[(t['source'], t['top_score'], t['level']) for t in evidence['trace']]}")
check("the accepted evidence is page 10", evidence["hits"][0]["page"] == 10,
      f"page {evidence['hits'][0]['page']}")
check("trace records a level for every attempt", all("level" in t for t in evidence["trace"]))
check("describe_trace produces a readable line", "->" in agent.describe_trace(evidence["trace"]))

refused = agent.collect_evidence("what is the trademark fee for a small enterprise", use_llm=False)
check("an off-topic question is not accepted", refused["accepted"] is False)

plan = agent.plan_queries("can I patent software", use_llm=False)
check("plan is deduplicated", len({p["query"].lower() for p in plan}) == len(plan))
check("plan respects the attempt budget", len(plan) <= config.MAX_ATTEMPTS)

# --------------------------------------------------------------- 6. pipeline
generator.llm_available = lambda: True
generator.call_llm = lambda messages, retries=2: {
    "text": "Computer programmes per se are not patentable. [Passage - p.10]",
    "latency_s": 0.3, "error": "",
}
answer = pipeline.answer_question("can I patent software")
check("pipeline returns an llm answer", answer["mode"] == "llm", answer["mode"])
check("answer carries sources and a trace", bool(answer["sources"]) and bool(answer["trace"]))
check("answer carries timings", answer["total_ms"] >= 0)

refusal = pipeline.answer_question("what is the trademark fee for a small enterprise")
check("pipeline refuses off-topic questions", refusal["mode"] == "no_answer", refusal["mode"])

generator.call_llm = lambda messages, retries=2: {"text": "", "latency_s": 0.2, "error": "HTTP 403"}
failed = pipeline.answer_question("can I patent software")
check("a model failure is reported as llm_error", failed["mode"] == "llm_error", failed["mode"])

generator.llm_available = lambda: False
quote = pipeline.answer_question("can I patent software")
check("quote mode still answers from passages", quote["mode"] == "extractive", quote["mode"])

check("an empty question is rejected", pipeline.answer_question("  ")["mode"] == "error")

# ----------------------------------------------------------------- 7. logging
rows = logger.QUERY_LOG.read_text(encoding="utf-8").strip().splitlines() if logger.QUERY_LOG.exists() else []
check("every question wrote a log row", len(rows) - 1 >= 4, f"{len(rows) - 1} rows")
header = rows[0].split(",") if rows else []
check("the log records the agent trace", "agent_trace" in header)
query_id = quote["query_id"]
check("feedback can be attached to a row", logger.set_feedback(query_id, "up") is True)
check("feedback for an unknown id is refused", logger.set_feedback("doesnotexist", "up") is False)

# --------------------------------------------------------------- 8. the app
try:
    retriever.stats = retriever.stats
    embedder.embedding_dimension = lambda: DIM
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=90)
    at.run()
    check("the Streamlit app renders without exceptions", not at.exception,
          str(at.exception)[:200])
    check("the sidebar is present with its controls", len(at.sidebar.markdown) > 0)
    check("the chat input exists", len(at.chat_input) > 0)
except Exception as exc:                                        # noqa: BLE001
    check("the Streamlit app renders without exceptions", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------------------- summary
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("ALL CHECKS PASSED")
