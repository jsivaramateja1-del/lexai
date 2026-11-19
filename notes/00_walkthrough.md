# Code walkthrough — read this before you read the code

This is the "explain it later" note. Each section says what a file does, why it exists,
and what you should be able to answer after reading it.

## The one idea

A language model knows nothing about the Patents Act specifically; it knows patterns of
language. RAG works around that by doing a search first and handing the model the search
results as its only permitted evidence. Everything in this repo is either (a) search, or
(b) handing search results to a model, or (c) making that process checkable.

## Files in the order the data flows

**`src/config.py`** — one place for chunk size, top-k, model names, thresholds. Read from
`.env` with `python-dotenv`. If a value is hardcoded in three files you will change two of
them and spend an evening debugging.

**`src/ingestor.py`** — three functions, three ideas:
1. `extract_pages` — PyMuPDF gives you text per page. Page numbers are kept because they
   become citations. A scanned PDF returns near-empty text; `looks_scanned` catches that
   and refuses rather than silently building an empty index.
2. `clean_text` — removes the damage PDF extraction causes: words hyphenated across line
   breaks, runs of spaces, stray indentation that looks like paragraphs.
3. `chunk_text` — splits into ~900-character pieces with ~150 characters of overlap.
   Chunks are packed paragraph by paragraph so a chunk is about one idea; the overlap means
   a sentence cut at a boundary still appears whole in the next chunk.

**`src/embedder.py`** — a vector is a list of numbers representing meaning. Same model for
documents and questions, always: two different models produce incompatible coordinate
systems. `normalize_embeddings=True` makes every vector unit length, which turns cosine
similarity into a plain dot product and keeps distances comparable.

**`src/retriever.py`** — ChromaDB persisted to `chroma_db/`. The collection is created with
`hnsw:space=cosine`. `upsert` (not `add`) means re-ingesting a corrected PDF replaces its
old chunks instead of duplicating them. `search` returns distance, where 0.0 is identical
meaning and 1.0 is unrelated.

**`src/generator.py`** — the only file that knows which LLM provider you use. Three parts:
`build_context` numbers the passages (`[Passage 1] Source: x.pdf, page 14`), `build_messages`
assembles system prompt + last few turns + the grounded user turn, `call_llm` posts the
request with timeout and exponential backoff on 429/5xx. `extractive_answer` is the no-API
fallback: it quotes passages verbatim, which cannot hallucinate.

**`src/pipeline.py`** — the five steps, readable top to bottom. The important line is the
relevance guard: if the best distance exceeds `MAX_DISTANCE`, return a refusal and stop.
That single check is what separates this from a chatbot that guesses.

**`src/logger.py`** — one row per question in `analytics/queries.csv`. That CSV is the
dataset for the Power BI dashboard. `set_feedback` rewrites the file to attach a thumbs
up/down to an existing row.

**`app.py`** — Streamlit. `st.session_state` holds the conversation because Streamlit
re-runs the whole script on every interaction; without session state your chat history
vanishes after each message. `@st.cache_resource` on the model loader prevents reloading
the embedding model on every keystroke-triggered rerun.

**`landing/`** — static HTML/CSS/JS. Semantic tags, meta description, Open Graph tags for
link previews, a mobile nav toggle, `aria-expanded` for screen readers.

## What to be able to explain

1. Why chunk with overlap instead of fixed 500-word blocks.
2. Why documents and questions must use the same embedding model.
3. What cosine distance 0.4 versus 0.9 means in practice.
4. Why the system can refuse to answer, and why that is a feature.
5. Where a hallucination could still slip through even with RAG, and what to do about it
   (stricter prompt, lower temperature, faithfulness evaluation).
6. Why `.env` is git-ignored and what happens if a key is committed.
