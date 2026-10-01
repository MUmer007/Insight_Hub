"""
app.py

InsightHub — professional AI/ML platform landing page.

Run:
    uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="InsightHub",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# HTML RENDER HELPER
# ============================================================================
# Streamlit's markdown renderer follows standard Markdown rules: a blank
# line followed by indented text is treated as a CODE BLOCK, which makes
# raw HTML show up as literal visible text instead of rendering, even with
# unsafe_allow_html=True. This helper strips blank lines and per-line
# indentation before rendering, so this can't happen no matter how the
# HTML string is formatted/indented in the source code.


def render_html(markup: str):
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    st.markdown("".join(lines), unsafe_allow_html=True)


# ============================================================================
# DESIGN TOKENS
# ============================================================================

COLORS = {
    "bg": "#F8F9FC",
    "surface": "#FFFFFF",
    "surface_alt": "#F4F5F9",
    "border": "#E6E8EF",
    "text": "#101828",
    "text_secondary": "#475467",
    "text_muted": "#98A2B3",
    "accent": "#4F46E5",
    "accent_dark": "#3730A3",
    "accent_soft": "#EEF2FF",
    "success": "#12B76A",
    "success_soft": "#ECFDF3",
}


# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
        rel="stylesheet"
    >

    <style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background: {COLORS["bg"]};
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}
    .block-container {{
        max-width: 1380px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Sora', sans-serif !important;
        color: {COLORS["text"]} !important;
    }}
    p {{
        color: {COLORS["text_secondary"]};
    }}
    section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid {COLORS["border"]};
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 1.5rem;
    }}
    .sidebar-brand {{
        padding: 8px 8px 24px 8px;
    }}
    .sidebar-logo {{
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: linear-gradient(135deg, {COLORS["accent"]}, #7C3AED);
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Sora', sans-serif;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 10px;
        box-shadow: 0 5px 14px rgba(79, 70, 229, 0.22);
    }}
    .sidebar-title {{
        font-family: 'Sora', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}
    .sidebar-subtitle {{
        font-size: 12px;
        color: {COLORS["text_muted"]};
        margin-top: 2px;
    }}
    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 34px;
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: 11px;
    }}
    .brand-mark {{
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: linear-gradient(135deg, {COLORS["accent"]}, #7C3AED);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-family: 'Sora', sans-serif;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.20);
    }}
    .brand-name {{
        font-family: 'Sora', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}
    .brand-label {{
        font-size: 11px;
        color: {COLORS["text_muted"]};
        margin-top: 1px;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: {COLORS["success_soft"]};
        color: #067647;
        border: 1px solid #ABEFC6;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {COLORS["success"]};
    }}
    .hero {{
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.13), transparent 30%),
            linear-gradient(135deg, #FFFFFF 0%, #F7F7FF 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 22px;
        padding: 48px 52px;
        margin-bottom: 24px;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border: 1px solid rgba(79, 70, 229, 0.08);
        border-radius: 50%;
        right: -90px;
        bottom: -130px;
    }}
    .eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: {COLORS["accent"]};
        background: {COLORS["accent_soft"]};
        border: 1px solid #DDE3FF;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }}
    .hero-title {{
        max-width: 800px;
        font-family: 'Sora', sans-serif;
        font-size: clamp(34px, 4vw, 52px);
        line-height: 1.08;
        letter-spacing: -0.04em;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-bottom: 18px;
    }}
    .hero-title span {{
        color: {COLORS["accent"]};
    }}
    .hero-description {{
        max-width: 730px;
        font-size: 16px;
        line-height: 1.75;
        color: {COLORS["text_secondary"]};
        margin-bottom: 28px;
    }}
    .hero-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
    }}
    .tech-pill {{
        background: rgba(255,255,255,0.8);
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 7px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        color: {COLORS["text_secondary"]};
    }}
    .section-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin: 34px 0 16px;
    }}
    .section-title {{
        font-family: 'Sora', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}
    .section-description {{
        font-size: 12px;
        color: {COLORS["text_muted"]};
        margin-top: 3px;
    }}
    .section-count {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: {COLORS["text_muted"]};
    }}
    .stat-card {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 100px;
    }}
    .stat-label {{
        font-size: 11px;
        font-weight: 600;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }}
    .stat-value {{
        font-family: 'Sora', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}
    .stat-detail {{
        font-size: 11px;
        color: {COLORS["text_secondary"]};
        margin-top: 3px;
    }}
    .module-card {{
        position: relative;
        height: 100%;
        min-height: 285px;
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 17px;
        padding: 24px;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}
    .module-card:hover {{
        transform: translateY(-3px);
        border-color: #C7CCFF;
        box-shadow: 0 14px 34px rgba(16, 24, 40, 0.08);
    }}
    .module-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 21px;
    }}
    .module-icon {{
        width: 42px;
        height: 42px;
        border-radius: 11px;
        background: {COLORS["accent_soft"]};
        color: {COLORS["accent"]};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
    }}
    .module-number {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: {COLORS["text_muted"]};
    }}
    .module-title {{
        font-family: 'Sora', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-bottom: 9px;
    }}
    .module-description {{
        font-size: 13px;
        line-height: 1.65;
        color: {COLORS["text_secondary"]};
        min-height: 67px;
    }}
    .module-footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-top: 1px solid {COLORS["border"]};
        margin-top: 20px;
        padding-top: 15px;
    }}
    .module-stack {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: {COLORS["text_muted"]};
    }}
    .module-arrow {{
        color: {COLORS["accent"]};
        font-size: 17px;
        font-weight: 600;
    }}
    .architecture {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 17px;
        padding: 24px;
    }}
    .architecture-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .architecture-node {{
        flex: 1;
        min-width: 130px;
        background: {COLORS["surface_alt"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 11px;
        padding: 14px;
        text-align: center;
    }}
    .node-title {{
        font-size: 12px;
        font-weight: 700;
        color: {COLORS["text"]};
    }}
    .node-detail {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        color: {COLORS["text_muted"]};
        margin-top: 4px;
    }}
    .architecture-arrow {{
        color: #B0B5C1;
        font-size: 17px;
    }}
    .footer {{
        border-top: 1px solid {COLORS["border"]};
        margin-top: 40px;
        padding: 22px 0 5px;
        display: flex;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: wrap;
        font-size: 11px;
        color: {COLORS["text_muted"]};
    }}
    .footer-tech {{
        font-family: 'JetBrains Mono', monospace;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SIDEBAR BRANDING
# ============================================================================

with st.sidebar:
    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">◆</div>
            <div class="sidebar-title">InsightHub</div>
            <div class="sidebar-subtitle">Customer Intelligence Platform</div>
        </div>
        """
    )

    st.markdown("---")
    st.caption("PLATFORM")
    st.info("Use the navigation above to access each intelligence module.")
    st.markdown("---")
    st.caption("SYSTEM")

    render_html(
        """
        <div style="font-size:12px; color:#667085; line-height:1.8;">
            <div>Environment &nbsp; <b style="color:#12B76A;">● Online</b></div>
            <div>Models &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3 active</div>
            <div>Version &nbsp;&nbsp;&nbsp;&nbsp; v1.0.0</div>
        </div>
        """
    )


# ============================================================================
# TOP BAR
# ============================================================================

render_html(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">◆</div>
            <div>
                <div class="brand-name">InsightHub</div>
                <div class="brand-label">AI / ML Customer Intelligence</div>
            </div>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            All systems operational
        </div>
    </div>
    """
)


# ============================================================================
# HERO
# ============================================================================

render_html(
    """
    <div class="hero">
        <div class="eyebrow">◆ Intelligence Platform</div>
        <div class="hero-title">
            Turn customer data into
            <span>actionable intelligence.</span>
        </div>
        <div class="hero-description">
            InsightHub brings predictive analytics, natural language
            processing, and retrieval-augmented generation together in one
            production-style AI platform.
        </div>
        <div class="hero-meta">
            <span class="tech-pill">Machine Learning</span>
            <span class="tech-pill">NLP</span>
            <span class="tech-pill">Generative AI</span>
            <span class="tech-pill">RAG</span>
        </div>
    </div>
    """
)


# ============================================================================
# PLATFORM OVERVIEW
# ============================================================================

render_html(
    """
    <div class="section-header">
        <div>
            <div class="section-title">Platform overview</div>
            <div class="section-description">Core intelligence capabilities available in InsightHub</div>
        </div>
        <div class="section-count">03 MODULES</div>
    </div>
    """
)

stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")

with stat1:
    render_html(
        """
        <div class="stat-card">
            <div class="stat-label">AI Modules</div>
            <div class="stat-value">03</div>
            <div class="stat-detail">Production-style workflows</div>
        </div>
        """
    )

with stat2:
    render_html(
        """
        <div class="stat-card">
            <div class="stat-label">Model Types</div>
            <div class="stat-value">05+</div>
            <div class="stat-detail">ML, NLP & GenAI</div>
        </div>
        """
    )

with stat3:
    render_html(
        """
        <div class="stat-card">
            <div class="stat-label">Data Layer</div>
            <div class="stat-value">RAG</div>
            <div class="stat-detail">Context-aware retrieval</div>
        </div>
        """
    )

with stat4:
    render_html(
        """
        <div class="stat-card">
            <div class="stat-label">Platform</div>
            <div class="stat-value">Live</div>
            <div class="stat-detail">Streamlit application</div>
        </div>
        """
    )


# ============================================================================
# MODULES
# ============================================================================

render_html(
    """
    <div class="section-header">
        <div>
            <div class="section-title">Intelligence modules</div>
            <div class="section-description">Select a capability from the sidebar to explore it</div>
        </div>
        <div class="section-count">AI WORKSPACE</div>
    </div>
    """
)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    render_html(
        """
        <div class="module-card">
            <div class="module-top">
                <div class="module-icon">↗</div>
                <div class="module-number">01 / PREDICTIVE</div>
            </div>
            <div class="module-title">Churn Prediction</div>
            <div class="module-description">
                Identify customers at risk of churn using behavioral,
                billing, usage, and engagement signals.
            </div>
            <div class="module-footer">
                <div class="module-stack">XGBoost · scikit-learn</div>
                <div class="module-arrow">→</div>
            </div>
        </div>
        """
    )

with col2:
    render_html(
        """
        <div class="module-card">
            <div class="module-top">
                <div class="module-icon">≡</div>
                <div class="module-number">02 / NLP</div>
            </div>
            <div class="module-title">Ticket Classifier</div>
            <div class="module-description">
                Automatically classify support tickets by topic and
                urgency using classical NLP and transformer models.
            </div>
            <div class="module-footer">
                <div class="module-stack">TF-IDF · DistilBERT</div>
                <div class="module-arrow">→</div>
            </div>
        </div>
        """
    )

with col3:
    render_html(
        """
        <div class="module-card">
            <div class="module-top">
                <div class="module-icon">◇</div>
                <div class="module-number">03 / GENAI</div>
            </div>
            <div class="module-title">Support Chat</div>
            <div class="module-description">
                Ask product and policy questions and receive grounded
                answers backed by relevant company documentation.
            </div>
            <div class="module-footer">
                <div class="module-stack">ChromaDB · Groq</div>
                <div class="module-arrow">→</div>
            </div>
        </div>
        """
    )


# ============================================================================
# ARCHITECTURE
# ============================================================================

render_html(
    """
    <div class="section-header">
        <div>
            <div class="section-title">Architecture</div>
            <div class="section-description">From raw customer data to intelligent applications</div>
        </div>
    </div>
    """
)

render_html(
    """
    <div class="architecture">
        <div class="architecture-row">
            <div class="architecture-node">
                <div class="node-title">Customer Data</div>
                <div class="node-detail">CSV · DB · Documents</div>
            </div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">
                <div class="node-title">Processing</div>
                <div class="node-detail">ETL · Cleaning · Features</div>
            </div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">
                <div class="node-title">AI Models</div>
                <div class="node-detail">ML · NLP · LLM</div>
            </div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">
                <div class="node-title">Intelligence</div>
                <div class="node-detail">Predictions · Insights</div>
            </div>
            <div class="architecture-arrow">→</div>
            <div class="architecture-node">
                <div class="node-title">Experience</div>
                <div class="node-detail">Dashboard · Chat</div>
            </div>
        </div>
    </div>
    """
)


# ============================================================================
# FOOTER
# ============================================================================

render_html(
    """
    <div class="footer">
        <div>InsightHub · Customer Intelligence Platform</div>
        <div class="footer-tech">scikit-learn · XGBoost · Transformers · ChromaDB · Groq</div>
        <div>v1.0.0</div>
    </div>
    """
)