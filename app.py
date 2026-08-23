"""
app.py

InsightHub — main landing page.
This becomes the "home" page; Streamlit automatically picks up every
.py file inside pages/ and adds it as a tab in the sidebar.

Run it with:  uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="InsightHub", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens + CSS — kept inline so this file stays self-contained.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#F6F7F9",
    "surface": "#FFFFFF",
    "border": "#E4E7EC",
    "text": "#0B1220",
    "text_secondary": "#475467",
    "text_muted": "#98A2B3",
    "accent": "#3538CD",
    "accent_soft": "#EEF0FD",
}

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {COLORS['text']};
    }}
    .stApp {{ background: {COLORS['bg']}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    h1, h2, h3 {{
        font-family: 'Sora', sans-serif;
        letter-spacing: -0.01em;
        color: {COLORS['text']};
    }}

    p, li {{
        color: {COLORS['text_secondary']};
        font-size: 15px;
        line-height: 1.6;
    }}

    .ih-eyebrow {{
        font-family: 'Inter', sans-serif;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLORS['accent']};
        margin-bottom: 6px;
    }}

    .ih-hero {{
        padding: 8px 0 4px;
    }}

    .ih-module-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 24px;
        height: 100%;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }}
    .ih-module-card:hover {{
        border-color: {COLORS['accent']};
        box-shadow: 0 4px 16px rgba(53, 56, 205, 0.08);
    }}

    .ih-icon-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: {COLORS['accent_soft']};
        color: {COLORS['accent']};
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 14px;
    }}

    .ih-module-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 16.5px;
        color: {COLORS['text']};
        margin-bottom: 6px;
    }}

    .ih-stack {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: {COLORS['text_muted']};
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid {COLORS['border']};
    }}

    .ih-footer {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: {COLORS['text_muted']};
        text-align: center;
        padding: 24px 0 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="ih-hero">', unsafe_allow_html=True)
st.markdown('<div class="ih-eyebrow">◆ InsightHub</div>', unsafe_allow_html=True)
st.markdown("# Customer Intelligence Platform")
st.markdown(
    """
    An end-to-end AI/ML platform combining classical machine learning, NLP,
    and generative AI — built to demonstrate the full modern AI Engineer /
    Data Scientist skill set, from data pipeline to production-style
    deployment.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Module cards
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        f"""
        <div class="ih-module-card">
            <div class="ih-icon-badge">▾</div>
            <div class="ih-module-title">Churn Prediction</div>
            <p>XGBoost model predicting customer churn risk from account
            usage, billing, and engagement signals.</p>
            <div class="ih-stack">scikit-learn · XGBoost</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="ih-module-card">
            <div class="ih-icon-badge">▤</div>
            <div class="ih-module-title">Ticket Classifier</div>
            <p>NLP model auto-categorizing support tickets by topic and
            urgency using a TF-IDF baseline and a fine-tuned DistilBERT.</p>
            <div class="ih-stack">TF-IDF · HuggingFace Transformers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="ih-module-card">
            <div class="ih-icon-badge">◇</div>
            <div class="ih-module-title">Support Chat</div>
            <p>RAG chatbot answering policy and product questions,
            grounded in real company documentation with cited sources.</p>
            <div class="ih-stack">ChromaDB · Groq</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.write("")

st.markdown(
    '<p style="text-align:center; color:#98A2B3; font-size:13.5px;">Use the sidebar to navigate between pages.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="ih-footer">scikit-learn · XGBoost · HuggingFace Transformers · ChromaDB · Groq</div>',
    unsafe_allow_html=True,
)