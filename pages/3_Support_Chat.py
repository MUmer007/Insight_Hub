"""
3_Support_Chat.py

A page in the InsightHub multi-page Streamlit app.
This is your existing RAG chatbot from Phase 4, moved into the
multi-page structure so it shows up as a tab in the sidebar
alongside Churn Prediction and Ticket Classifier.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))
from rag_query import RagPipeline

st.set_page_config(page_title="Support Chat", page_icon="💬")
st.title("💬 Support Assistant")
st.caption("Ask about shipping, returns, or billing — answers are grounded in our actual policy docs.")


@st.cache_resource
def load_pipeline():
    return RagPipeline()


with st.spinner("Loading assistant..."):
    rag = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

if question := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(question)
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption(f"Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant uses RAG (Retrieval-Augmented Generation): "
        "your question is matched against our knowledge base, and the "
        "LLM answers using only that retrieved context."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
