# LexAI
> Developed for CDS208 problem statement #4 — IP-SAKTI Sahayak.

A source-cited AI assistant for Indian intellectual property information. Ask a question in
plain language; the assistant retrieves the relevant passages from official IP documents and
answers **only** from them, citing the document name and page.

> **General information, not legal advice.** This prototype retrieves and quotes public
> documents. For a specific case, consult a registered patent or trademark agent.

## The problem

Indian IP law is spread across long, dense documents — the Patents Act, the Trade Marks Act,
the Copyright Act, rules, fee schedules and examination guidelines. General-purpose chatbots
answer such questions fluently and sometimes wrongly. In a legal context, a confident wrong
answer is worse than no answer.

## The approach

Retrieval-Augmented Generation (RAG): the model never answers from memory. It answers from
passages retrieved out of a knowledge base you control, and every claim carries a citation.
When nothing relevant is retrieved, it refuses.

```
data/documents/*.pdf
      |  src/ingestor.py      extract -> clean -> overlapping chunks (+ page number)
      v
   src/embedder.py            local multilingual sentence-transformers
      v
   ChromaDB (local)           cosine similarity index
      ^
      |  src/retriever.py      top-k nearest chunks for the question
      v
   src/generator.py           prompt with numbered passages -> Sarvam LLM
      v
   src/pipeline.py            relevance guard, orchestration, logging
      v
   app.py (Streamlit)  +  landing/ (HTML/CSS/JS)  +  analytics/queries.csv -> Power BI
```

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit chat UI, feedback buttons, source viewer |
| `src/config.py` | Every tunable value, read from `.env` |
| `src/ingestor.py` | PDF extraction, cleaning, overlapping chunking |
| `src/embedder.py` | One local embedding model for both documents and queries |
| `src/agent.py` | Agentic retrieval: propose phrasings, retry while the best match is weak |
| `src/hybrid.py` | Keyword index, reciprocal rank fusion, lexical relevance guard |
| `src/retriever.py` | ChromaDB storage and top-k semantic search |
| `src/generator.py` | Prompt construction and the single LLM call |
| `src/pipeline.py` | retrieve → guard → generate → log |
| `src/logger.py` | Query logging to `analytics/queries.csv` |
| `scripts/ingest.py` | Build/refresh the knowledge base |
| `tests/` | Unit tests for the AI-free parts (cleaning, chunking) |
| `landing/` | Responsive landing page with basic SEO |
| `notes/` | Learning notes and code walkthrough |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your Sarvam values
```

Add PDFs to `data/documents/`, then build the index:

```bash
python -m scripts.ingest --reset
```

Run the app:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest -q
python tests/smoke_check.py
```

## Configuration

Everything lives in `.env` (see `.env.example`). Two settings matter most:

- `LLM_ENABLED=false` runs the app in **quote mode**: it returns the best-matching passages
  verbatim, with no API call. Use this while developing so free credits are not spent.
- `MAX_DISTANCE` is the refusal threshold. Cosine distance above it means "not in the
  knowledge base". Tune it against your evaluation set, not by feel.

The Sarvam endpoint and model name are intentionally left blank in `.env.example`. Confirm
them in the official Sarvam documentation — model names and limits change.

## Knowledge base

Only public documents, each recorded with its source URL and the date it was downloaded:

| Document | Status |
| --- | --- |
| The Patents Act, 1970 (as amended) | to be added |
| The Trade Marks Act, 1999 (as amended) | to be added |
| The Copyright Act, 1957 (as amended) | to be added |
| IP office examination guidelines | to be added |

Amendments matter: an answer grounded in a superseded text is still wrong. Each document's
version and download date will be recorded in `data/documents/SOURCES.md`.

## Measured results

Real numbers from the current build, one statute (Patents Act, 1970, as amended to 01-08-2024):

| Measure | Value |
| --- | --- |
| Pages extracted | 69 |
| Chunks indexed | 507 (900 characters, 150 overlap, page-scoped, boundary-aware) |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2, local, 384 dimensions |
| Rebuild time | ~22 s |
| Refusal threshold | cosine distance 0.35 (similarity 0.65), calibrated from test questions |
| Unit tests | 42 passing, plus one end-to-end smoke check |
| Cost per answered question | roughly 2 paise at introductory Sarvam pricing |

Chunks are cut at clause, sentence or word boundaries, never mid-word: an exact phrase like
"computer programme per se" is the legal meaning, so a split inside it destroys the evidence.
A forward-progress guard keeps overlapping splits from emitting ever-shorter fragments.

A chunk-size experiment (450/80, 900/150, 1400/200 against four fixed questions) chose the
900/150 setting: it was the only configuration that returned the correct page for both
answerable questions. The highest similarity score in that experiment belonged to a wrong
answer, which is why scores are used to rank candidates, never to certify them.

## Agentic retrieval

`src/agent.py` runs a bounded decision loop instead of a single lookup:

1. **Plan** — the question itself, then a statutory-phrasing variant from a small glossary
   (`software` → `computer programme per se`, `brand` → `trade mark`), then a key-term
   variant, then model-written rewrites when a key is configured.
2. **Act** — retrieve with the first phrasing.
3. **Observe** — is the best passage within the refusal threshold?
4. **Re-act** — if not, try the next phrasing; stop at the first accepted result.

The loop is capped at `MAX_ATTEMPTS` (default 4), and every attempt is recorded in a trace
that the UI shows and the query log stores, so the behaviour is auditable rather than opaque.
It runs without an API key, because the statutory and key-term variants need no model.

The glossary exists because a statute says "computer programme per se" where a user says
"software". Dense retrieval often bridges that gap and often does not; one extra local search
closes the whole class of misses, and a test asserts that "can I patent software" is planned
with the statutory wording.

## Hybrid retrieval

Dense vectors are good at meaning and bad at exactness. "Can I patent software" returns
pages about patent applications because the topic is patents; the exact phrase "computer
programme per se" appears on one page of the Act and a vector search still missed it,
because that chunk also contains a dozen other clauses and its vector is an average.

So retrieval now fuses two rankings with Reciprocal Rank Fusion, which merges by rank
rather than by score, so a cosine distance and a TF-IDF weight never need a shared scale:

| Signal | Finds | Blind to |
| --- | --- | --- |
| Vector (sentence-transformers) | paraphrase, synonyms, meaning | exact rare wording |
| Keyword (TF-IDF, 1-3 word n-grams) | statutory phrases, section wording | paraphrase |

The relevance guard has three levels, not two:

| Level | Test | Effect |
| --- | --- | --- |
| Strong | contains the statutory phrase, or distance ≤ `STRONG_DISTANCE` (0.28) | answer, and stop searching |
| Provisional | distance ≤ `MAX_DISTANCE` (0.35) | keep as a fallback, keep searching |
| Reject | neither | discard |

The middle level exists because of a measured failure: "can I patent software" returned page 29
at 0.659 similarity, which cleared the threshold by a hair, so the agent stopped before trying
the statutory phrasing that would have found Section 3(k) on page 10. A barely-passing match is
now treated as a fallback rather than a finish line. Switches: `HYBRID_ENABLED`,
`LEXICAL_ACCEPT`, `STRONG_DISTANCE`.

## Answer modes

Every response reports which path produced it, so a failure is never ambiguous:

| Mode | Meaning |
| --- | --- |
| `llm` | The model answered from the retrieved passages, with citations |
| `extractive` | Quote mode: no key configured or `LLM_ENABLED=false`. Passages are quoted verbatim |
| `llm_error` | The model was called and failed. The passages are quoted instead, with the error shown |
| `no_answer` | Nothing passed the relevance threshold. Refused before any model call |
| `error` | The question was empty or malformed |

## Interface

`app.py` is a designed Streamlit interface rather than default chrome: a branded header with
live status chips, source cards showing page, similarity and distance, a panel showing each
retrieval attempt the agent made, feedback buttons wired to the query log, and example
questions on an empty screen. All styling is CSS injected at runtime — no build step, nothing
to install.

## Status

Working end to end on one statute: ingestion, retrieval, agentic retry, grounded answers with
page-level citations, refusal, Streamlit UI, landing page and query logging. Remaining: a 20-30
question evaluation set, two more statutes, and the Power BI dashboard.

Built for CDS208 problem statement #4 at Sai University.
