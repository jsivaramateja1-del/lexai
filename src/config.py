"""Central configuration. Every tunable value lives here, read from .env.

Why one file? So that when an interviewer asks "where is your chunk size
defined?", the answer is one file, not a hunt through six modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Reads .env sitting next to this project's root. Never commit that file.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
ANALYTICS_DIR = BASE_DIR / "analytics"
QUERY_LOG = ANALYTICS_DIR / "queries.csv"

# ---------------------------------------------------------------- retrieval
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ip_documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))      # characters, not tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Agentic retrieval: propose alternative search phrasings and retry while
# the best passage is too far away. Bounded, so it cannot loop forever.
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() == "true"
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# Cosine distance above which we refuse to answer ("not in the knowledge base").
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))

# --------------------------------------------------------------------- LLM
# VERIFY EVERY VALUE BELOW IN THE OFFICIAL SARVAM DOCS before trusting it.
# Model names, endpoints and limits change; this file is the only place to edit.
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_API_URL = os.getenv("SARVAM_API_URL", "")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "")
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

DISCLAIMER = (
    "This is general information extracted from public documents, "
    "not legal advice. For your specific case, consult a registered "
    "patent or trademark agent."
)
