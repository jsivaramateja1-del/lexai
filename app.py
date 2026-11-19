"""LexAI — Streamlit interface.

Design notes
------------
Streamlit gives you a working page in ten lines and an anonymous one in a
hundred. This file keeps the plumbing (session state, caching, error handling)
and adds a deliberate visual layer on top: a branded header, a citation card for
every source, and a visible trace of what the retrieval agent did.

All styling is CSS injected once, so there is no build step and nothing to
install. Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src import agent, retriever
from src.config import (
    AGENT_ENABLED,
    DISCLAIMER,
    LLM_ENABLED,
    MAX_ATTEMPTS,
    MAX_DISTANCE,
    SARVAM_MODEL,
    TOP_K,
)
from src.logger import set_feedback
from src.pipeline import answer_question

st.set_page_config(
    page_title="LexAI — cited answers on Indian IP law",
    page_icon="§",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- css
CSS = """
<style>
:root{
  --ground:#0e1114; --panel:#161a1f; --panel-2:#1c2228; --line:#262d35;
  --ink:#e9e6df; --ink-2:#a9b2bb; --ink-3:#77828c;
  --accent:#e0b252; --accent-soft:rgba(224,178,82,.12);
  --good:#5fbf8a; --bad:#e0705f;
  --radius:14px;
}
.stApp{background:var(--ground);color:var(--ink);}
html,body,[class*="css"]{font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;}
/* Hide the chrome, but keep the header alive: it holds the sidebar toggle.
   Hiding `header` outright made a collapsed sidebar impossible to reopen. */
#MainMenu,footer{visibility:hidden;}
[data-testid="stToolbar"]{visibility:hidden;}
[data-testid="stDecoration"]{display:none;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebarCollapseButton"]{visibility:visible !important;}
[data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
.block-container{padding-top:2.2rem;max-width:1080px;}

/* brand header */
.lex-head{display:flex;align-items:center;gap:14px;margin-bottom:6px;}
.lex-mark{width:44px;height:44px;border-radius:12px;background:var(--accent);
  color:#1a1305;display:grid;place-items:center;font-weight:700;font-size:22px;}
.lex-title{font-size:1.55rem;font-weight:650;letter-spacing:-.01em;margin:0;}
.lex-sub{color:var(--ink-3);font-size:.86rem;margin:2px 0 0;}
.lex-chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 22px;}
.chip{border:1px solid var(--line);background:var(--panel);color:var(--ink-2);
  border-radius:999px;padding:4px 11px;font-size:.74rem;letter-spacing:.02em;}
.chip b{color:var(--ink);font-weight:600;}
.chip.on{border-color:var(--accent);color:var(--accent);background:var(--accent-soft);}

/* chat */
[data-testid="stChatMessage"]{background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:16px 18px;margin-bottom:12px;}
[data-testid="stChatMessage"] p{color:var(--ink);line-height:1.62;}
[data-testid="stChatInput"]{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);}
[data-testid="stChatInput"] textarea{color:var(--ink);}

/* source cards */
.src{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;
  background:var(--panel-2);padding:12px 14px;margin-bottom:10px;}
.src-top{display:flex;justify-content:space-between;gap:12px;font-size:.78rem;
  color:var(--ink-2);margin-bottom:6px;}
.src-top b{color:var(--ink);font-weight:600;}
.src-body{color:var(--ink-2);font-size:.82rem;line-height:1.55;}

/* trace */
.trace-row{display:grid;grid-template-columns:26px 118px 1fr auto;gap:10px;
  align-items:baseline;padding:7px 0;border-top:1px solid var(--line);font-size:.8rem;}
.trace-row:first-of-type{border-top:0;}
.trace-step{color:var(--ink-3);}
.trace-src{color:var(--accent);letter-spacing:.02em;}
.trace-q{color:var(--ink-2);}
.trace-score{color:var(--ink);font-variant-numeric:tabular-nums;}
.ok{color:var(--good);} .no{color:var(--bad);} .warn{color:var(--accent);}

/* sidebar + controls */
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] *{color:var(--ink-2);}
section[data-testid="stSidebar"] h2{color:var(--ink);font-size:.9rem;letter-spacing:.06em;
  text-transform:uppercase;}
.stat{display:flex;justify-content:space-between;font-size:.82rem;padding:6px 0;
  border-bottom:1px solid var(--line);}
.stat b{color:var(--ink);font-variant-numeric:tabular-nums;}
.stButton>button{border-radius:10px;border:1px solid var(--line);background:var(--panel-2);
  color:var(--ink);font-size:.83rem;}
.stButton>button:hover{border-color:var(--accent);color:var(--accent);}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:transparent;}
[data-testid="stExpander"] summary{color:var(--ink-2);font-size:.82rem;}
.lex-note{color:var(--ink-3);font-size:.75rem;line-height:1.5;}
.lex-empty{border:1px dashed var(--line);border-radius:var(--radius);padding:22px;
  color:var(--ink-3);font-size:.88rem;line-height:1.6;}
</style>
"""


# --------------------------------------------------------------------- helpers
@st.cache_resource
def warm_embeddings() -> bool:
    """Load the embedding model once per process, not on every rerun."""
    from src.embedder import embedding_dimension

    embedding_dimension()
    return True


def header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="lex-head">
          <div class="lex-mark">§</div>
          <div>
            <p class="lex-title">LexAI</p>
            <p class="lex-sub">Source-cited answers from Indian intellectual property documents</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        chunks = retriever.stats()["chunks"]
    except Exception:                                            # noqa: BLE001
        chunks = 0

    mode = "LLM answers" if LLM_ENABLED else "quote mode"
    agent_state = f"agent · up to {MAX_ATTEMPTS} searches" if AGENT_ENABLED else "single search"
    st.markdown(
        f"""
        <div class="lex-chips">
          <span class="chip"><b>{chunks}</b> chunks indexed</span>
          <span class="chip"><b>{TOP_K}</b> passages per question</span>
          <span class="chip"><b>{MAX_DISTANCE}</b> refusal threshold</span>
          <span class="chip">{agent_state}</span>
          <span class="chip {'on' if LLM_ENABLED else ''}">{mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.header("Knowledge base")
        try:
            info = retriever.stats()
            st.markdown(
                f"""
                <div class="stat"><span>Chunks indexed</span><b>{info['chunks']}</b></div>
                <div class="stat"><span>Collection</span><b>{info['collection']}</b></div>
                <div class="stat"><span>Model</span><b>{SARVAM_MODEL or 'not set'}</b></div>
                <div class="stat"><span>Answer mode</span><b>{'LLM' if LLM_ENABLED else 'quote'}</b></div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:                                  # noqa: BLE001
            st.error(f"Vector store unavailable: {exc}")

        st.write("")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.write("")
        st.header("How it answers")
        st.markdown(
            "<p class='lex-note'>Retrieval runs first. If the best passage is further than "
            "the threshold, the assistant refuses and no model is called. When it does "
            "answer, every claim must cite a retrieved passage.</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(f"<p class='lex-note'>{DISCLAIMER}</p>", unsafe_allow_html=True)


def source_cards(passages: list[dict]) -> None:
    if not passages:
        return
    with st.expander(f"Sources ({len(passages)})", expanded=False):
        for hit in passages:
            snippet = " ".join(hit["text"].split())[:420]
            st.markdown(
                f"""
                <div class="src">
                  <div class="src-top">
                    <span><b>{hit['source']}</b> · page {hit['page']}</span>
                    <span>similarity {hit['score']:.3f} · distance {hit['distance']:.3f}</span>
                  </div>
                  <div class="src-body">{snippet} …</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def trace_panel(result: dict) -> None:
    trace = result.get("trace") or []
    if not trace:
        return
    with st.expander(f"How the agent searched ({len(trace)} attempt(s))", expanded=False):
        rows = []
        for step in trace:
            score = f"{step['top_score']:.3f}" if step["top_score"] is not None else "n/a"
            level = step.get("level", 0)
            verdict = (
                "<span class='ok'>strong</span>" if level >= 2
                else "<span class='warn'>provisional</span>" if level == 1
                else "<span class='no'>rejected</span>"
            )
            via = step.get("matched_by") or "vector"
            rows.append(
                f"<div class='trace-row'><span class='trace-step'>{step['step']}</span>"
                f"<span class='trace-src'>{step['source']}</span>"
                f"<span class='trace-q'>\"{step['query']}\"</span>"
                f"<span class='trace-score'>p.{step['top_page']} · {score} · {via} · {verdict}</span></div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def suggestions() -> None:
    st.markdown(
        "<div class='lex-empty'>Ask a question in plain language. The assistant answers only "
        "from the ingested documents and cites the page. If the documents do not cover it, "
        "it will say so.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    examples = [
        "What is not patentable?",
        "Can I patent software?",
        "What counts as an invention?",
        "How long does a patent last?",
    ]
    columns = st.columns(2)
    for index, example in enumerate(examples):
        if columns[index % 2].button(example, key=f"sug-{index}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            source_cards(message["passages"])
        if message.get("trace"):
            trace_panel(message)


def run(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents …"):
            try:
                result = answer_question(question, history=st.session_state.messages[:-1])
            except Exception as exc:                              # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                return

        st.markdown(result["answer"])
        source_cards(result["passages"])
        trace_panel(result)
        st.caption(
            f"{result['mode']} · retrieval {result['retrieval_ms']} ms · "
            f"model {result['llm_ms']} ms · total {result['total_ms']} ms"
        )
        if result.get("error"):
            st.caption(f"error: {result['error'][:220]}")

        left, right, _ = st.columns([1, 1, 6])
        if left.button("Helpful", key=f"up-{result['query_id']}", use_container_width=True):
            set_feedback(result["query_id"], "up")
            st.toast("Feedback saved")
        if right.button("Not helpful", key=f"down-{result['query_id']}", use_container_width=True):
            set_feedback(result["query_id"], "down")
            st.toast("Feedback saved")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "passages": result["passages"],
            "trace": result.get("trace", []),
        }
    )


def main() -> None:
    header()
    warm_embeddings()
    sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        empty_index = retriever.stats()["chunks"] == 0
    except Exception:                                            # noqa: BLE001
        empty_index = True
    if empty_index:
        st.warning(
            "No documents indexed yet. Put PDFs in `data/documents/` and run "
            "`python -m scripts.ingest --reset`."
        )

    if not st.session_state.messages:
        suggestions()

    for message in st.session_state.messages:
        render_message(message)

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("Ask about patents, trademarks or copyright …")
    question = pending or typed
    if question:
        run(question)


if __name__ == "__main__":
    main()
