"""Streamlit chat UI. Run with:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from src import retriever
from src.config import DISCLAIMER, LLM_ENABLED, MAX_DISTANCE, TOP_K
from src.logger import set_feedback
from src.pipeline import answer_question

st.set_page_config(page_title="IP-SAKTI Sahayak", page_icon="§", layout="wide")


@st.cache_resource
def warm_embeddings() -> bool:
    """Load the embedding model once, not on every rerun."""
    from src.embedder import embedding_dimension

    embedding_dimension()
    return True


def sidebar() -> None:
    with st.sidebar:
        st.header("Knowledge base")
        try:
            info = retriever.stats()
            st.metric("Chunks indexed", info["chunks"])
        except Exception as exc:                              # noqa: BLE001
            st.error(f"Vector store unavailable: {exc}")

        st.caption(
            f"top_k = {TOP_K} · refuse threshold = {MAX_DISTANCE} · "
            f"LLM = {'on' if LLM_ENABLED else 'off (quote mode)'}"
        )

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(DISCLAIMER)


def render_sources(result: dict) -> None:
    if not result["sources"]:
        return
    with st.expander(f"Sources ({len(result['sources'])})"):
        for hit in result["passages"]:
            st.markdown(
                f"**{hit['source']} — page {hit['page']}** · "
                f"similarity {hit['score']:.3f} · distance {hit['distance']:.3f}"
            )
            st.caption(" ".join(hit["text"].split())[:500] + " ...")
            st.divider()


def main() -> None:
    st.title("IP-SAKTI Sahayak")
    st.caption("Source-cited answers from Indian intellectual property documents.")

    warm_embeddings()
    sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if retriever.stats()["chunks"] == 0:
        st.warning(
            "No documents indexed yet. Put PDFs in `data/documents/` and run:\n\n"
            "`python -m scripts.ingest`"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for source in message["sources"]:
                        st.write(f"{source['source']} — page {source['page']}")

    question = st.chat_input("Ask about patents, trademarks or copyright ...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents ..."):
            try:
                result = answer_question(question, history=st.session_state.messages[:-1])
            except Exception as exc:                          # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                return

        st.markdown(result["answer"])
        render_sources(result)
        st.caption(
            f"mode: {result['mode']} · retrieval {result['retrieval_ms']} ms · "
            f"llm {result['llm_ms']} ms · total {result['total_ms']} ms"
        )

        col_up, col_down, _ = st.columns([1, 1, 8])
        if col_up.button("Helpful", key=f"up-{result['query_id']}"):
            set_feedback(result["query_id"], "up")
            st.toast("Feedback saved")
        if col_down.button("Not helpful", key=f"down-{result['query_id']}"):
            set_feedback(result["query_id"], "down")
            st.toast("Feedback saved")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


if __name__ == "__main__":
    main()
