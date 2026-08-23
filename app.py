"""
app.py

Phase 4/8 - A real web UI for the RAG chatbot, built with Streamlit.

Run it with:  uv run streamlit run app.py
This opens a browser tab automatically at http://localhost:8501
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable from this root-level file
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))
from rag_query import RagPipeline


st.set_page_config(
    page_title="InsightHub Support Assistant",
    page_icon="💬",
    layout="centered",
)

st.title("💬 InsightHub Support Assistant")
st.caption("Ask about shipping, returns, or billing — answers are grounded in our actual policy docs, not guesses.")


# st.cache_resource means this expensive setup (loading models, connecting to
# the database) only runs ONCE, not every time the user sends a message.
@st.cache_resource
def load_pipeline():
    return RagPipeline()


with st.spinner("Loading assistant..."):
    rag = load_pipeline()

# Keep chat history across reruns using Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the full conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

# The input box pinned at the bottom of the page
if question := st.chat_input("Type your question..."):
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show the assistant's response
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

# A button to clear the conversation and start over
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
