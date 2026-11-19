# CDS208 Final Project — Proposal Draft

**Student:** JOGIPARTHI VENKATA AUDI SIVA RAMA TEJA
**Programme:** B.Tech Computer Science, Semester III, School of Computing and Data Science, Sai University
**Course:** CDS208 Emerging Tools and Applications (Lab: CDS228)
**Selected problem statement:** #4 — IP-SAKTI Sahayak: AI Regulatory & Knowledge Assistant
**Technology focus:** RAG, Agentic AI, Prompt Engineering, Streamlit, Web
**Working title:** LexAI

> Edit the bracketed placeholders (student ID, dates) before submitting. Faculty approval for
> solo work is confirmed verbally — attach or mention it if asked.

## 1. Problem

Intellectual property law in India — patents, trademarks and copyright — is documented across
long, dense statutes, rules, fee schedules and examination guidelines. A student, founder or
small business owner asking a simple question ("is my software idea patentable in India?")
must locate the right Act, find the right section, and read surrounding definitions to
interpret it. General-purpose chatbots appear to solve this but frequently invent section
numbers and case law, which in a legal context is worse than no answer.

## 2. Proposed solution

LexAI is a retrieval-augmented assistant. A user asks a question in plain
language (English or an Indian language). The system retrieves the most relevant passages
from an indexed knowledge base of official Indian IP documents, passes only those passages to
a language model, and returns an answer in which every claim is cited by document name and
page number. When retrieval finds nothing sufficiently relevant, the assistant explicitly
refuses instead of guessing. A standing disclaimer states that the output is general
information, not legal advice.

## 3. Key features

1. PDF ingestion pipeline: text extraction, cleaning, overlapping chunking with page-level provenance.
2. Local multilingual embeddings, so a Hindi or Tamil question can match English statute text.
3. Semantic search over a local vector database (top-k retrieval with relevance scores).
4. Grounded answer generation through the Sarvam AI API, with a strict citation-only system prompt.
5. Honest refusal path driven by a relevance threshold.
6. Streamlit chat interface with multi-turn history, source viewer and thumbs up/down feedback.
7. Responsive HTML/CSS/JavaScript landing page with basic SEO.
8. Query logging to CSV feeding a Power BI dashboard on real usage.
9. A 20–30 question evaluation set measuring retrieval accuracy and answer faithfulness.

## 4. Technology stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Course requirement; best library support for AI work |
| PDF extraction | PyMuPDF | Fast, keeps page numbers for citations |
| Embeddings | sentence-transformers (multilingual, local) | Free, offline, no per-query cost, handles Indian languages |
| Vector store | ChromaDB (local, persistent) | Zero setup, runs on a laptop, easy to inspect |
| LLM | Sarvam AI API | Indian-language capable, free signup credits |
| Orchestration | Hand-written pipeline first, LangChain studied later | Understanding before abstraction |
| UI | Streamlit + static HTML/CSS/JS landing page | Fast iteration; landing page covers the web component |
| Analytics | CSV logging + Power BI | Uses the DAX/dashboard skills from the course |
| Version control | Git + GitHub | Meaningful commit history as evidence of process |
| Secrets | `.env` + `.env.example` | Keys never committed |

## 5. Initial module breakdown

| Module | File | Responsibility |
| --- | --- | --- |
| Configuration | `src/config.py` | All tunable values and secrets loading |
| Ingestion | `src/ingestor.py` | PDF → clean text → overlapping chunks with page numbers |
| Embedding | `src/embedder.py` | One model for documents and queries |
| Retrieval | `src/retriever.py` | ChromaDB storage, top-k semantic search |
| Generation | `src/generator.py` | Prompt construction, the single LLM call, offline fallback |
| Orchestration | `src/pipeline.py` | Retrieve → relevance guard → generate → log |
| Logging | `src/logger.py` | Query analytics CSV, feedback updates |
| Interface | `app.py` | Streamlit chat UI |
| Landing page | `landing/` | Responsive marketing page with SEO basics |
| Evaluation | `tests/`, `eval/` | Unit tests and the RAG evaluation set |

## 6. Milestones (8 weeks, solo, 2–4 hours/day)

| Week | Focus | Checkpoint |
| --- | --- | --- |
| 1 | Python, venv, JSON, HTTP, APIs, Git setup | Repo skeleton committed; proposal submitted |
| 2 | Streamlit basics, project structure, `.env`, error handling | Working app shell; landing page draft |
| 3 | LLMs, tokens, prompting, Sarvam API, multi-turn chat | Chatbot answering from prompt only |
| 4 | RAG part 1: extraction, cleaning, chunking, embeddings | Knowledge base built; evaluation questions drafted |
| 5 | RAG part 2: ChromaDB, retrieval, citations, full pipeline | End-to-end cited answers; logging live |
| 6 | Integration, LangChain comparison, UI polish, one agentic step | Polished app; query rewriter or document-selection tool |
| 7 | Logging, error handling, tests, security, Power BI, evaluation | Dashboard with 4+ visuals; measured accuracy |
| 8 | README, architecture diagram, demo video, PPT, resume bullets | Final submission package |

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Free API credits are limited | `LLM_ENABLED=false` quote mode during development; response caching; token logging per request |
| Sarvam endpoints/model names may change | All provider details isolated in `src/generator.py` and `.env`; verified against official docs before use |
| Scanned or poorly formatted PDFs | Text-density check at ingestion; only digital official PDFs used initially; OCR considered later |
| Superseded statute versions | Source register records version, URL and download date for every document |
| Hallucinated answers despite grounding | Strict citation-only prompt, low temperature, refusal threshold, faithfulness check on the evaluation set |
| Scope creep as a solo developer | MVP frozen at end of Week 5; advanced features only after MVP demo works |

## 8. Status at the progress review (20 November 2026)

Working end to end on one statute. Measured: 69 pages of the Patents Act, 1970 extracted,
482 chunks indexed at 900 characters with 150 overlap, page-level citations on every answer,
a refusal threshold of 0.35 cosine distance calibrated from four fixed test questions, and a
bounded agentic retrieval loop (plan, retrieve, check, retry) with an auditable trace.
16 unit tests and one end-to-end smoke check pass; the repository is public with typed commit
messages. Remaining: a 20-30 question evaluation set, two further statutes, and the Power BI
dashboard on the query log.

## 9. Deliverables

Public GitHub repository with meaningful commit history, working prototype, README with
architecture diagram, evaluation results, Power BI dashboard, demo video and presentation.
