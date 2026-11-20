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


def _bridge_streamlit_secrets() -> None:
    """Make hosted secrets behave like environment variables.

    Streamlit Community Cloud does not ship a .env file: it injects secrets through
    st.secrets. Every module here reads os.environ, so without this bridge a
    deployed app would silently run in quote mode with no API key.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except Exception:                                            # noqa: BLE001
        return
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
    except Exception:                                            # noqa: BLE001
        # No secrets configured, or called outside a Streamlit runtime.
        return


_bridge_streamlit_secrets()

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

# Hybrid retrieval: fuse vector similarity with a TF-IDF keyword index, because
# legal questions hinge on exact statutory wording as well as on meaning.
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"

# Let a verbatim statutory phrase rescue a passage whose cosine score is weak.
LEXICAL_ACCEPT = os.getenv("LEXICAL_ACCEPT", "true").lower() == "true"
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "4"))

# Cosine distance above which we refuse to answer ("not in the knowledge base").
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))

# A match this close stops the agent immediately. Between STRONG_DISTANCE and
# MAX_DISTANCE the hit is "provisional": usable as a fallback, but not good
# enough to stop searching, because a barely-passing page is how a false accept
# happens (0.659 similarity once stopped the loop before it tried the right query).
STRONG_DISTANCE = float(os.getenv("STRONG_DISTANCE", "0.28"))

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
