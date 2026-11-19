# IP-SAKTI Sahayak

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

## Status

Week 1 skeleton: pipeline, UI, landing page and logging in place, knowledge base empty.
Built as the CDS208 final project at Sai University.
