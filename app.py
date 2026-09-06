"""
SmartFilteringRAG — Streamlit UI
================================
A premium dark-themed interface for the SmartFilteringRAG pipeline.
Wraps existing pipeline functions without modifying them.

Two query modes:
  • Structured dataset — metadata filter generation + pre-filtered vector search
  • Documents          — PDF/DOCX chunks, retrieval straight to answer synthesis

Orchestration runs through the LangGraph agent in rag/graph.py; the nodes it
runs are shown in the execution-trace panel alongside the latency breakdown.
"""

# Load .env BEFORE any rag.* imports (mongodb_helper reads MONGO_URI at import time)
from dotenv import load_dotenv
load_dotenv()

import html
import json
import logging
import os
import re
import time
from copy import deepcopy

import pandas as pd
import streamlit as st
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from rag.auto_metadata import extract_metadata_schema
from rag.config_loader import config
from rag.doc_ingest import (
    SUPPORTED_EXTENSIONS,
    default_document_collection,
    ingest_documents,
    list_ingested_sources,
)
from rag.graph import (
    MODE_DOCUMENTS,
    MODE_STRUCTURED,
    PipelineContext,
    make_faithfulness_llm,
    run_graph,
)
from rag.ingest import ingest_dataset
from rag.persistence import (
    history_entries_from_turns,
    log_turn,
    new_session_id,
    rehydrate_session,
)
from rag.metadata_filter import MetadataFilter
from rag.prompts import enforce_constraints
from rag.query_resolver import resolve_query
from rag.utils.mongodb_helper import get_mongo_collection
from rag.utils.prepare_test_data import get_docs_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFilteringRAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark theme, glassmorphism, gradients, micro-animations
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import premium font ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ───────────────────────────────────────────────────── */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: rgba(22, 27, 34, 0.75);
    --bg-glass: rgba(30, 37, 48, 0.55);
    --border-glass: rgba(99, 140, 255, 0.15);
    --accent-blue: #58a6ff;
    --accent-purple: #bc8cff;
    --accent-green: #3fb950;
    --accent-orange: #f0883e;
    --accent-red: #f85149;
    --accent-cyan: #39d2c0;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --gradient-main: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #f778ba 100%);
    --gradient-card: linear-gradient(145deg, rgba(30,37,48,0.7) 0%, rgba(22,27,34,0.9) 100%);
    --shadow-glow: 0 0 20px rgba(88, 166, 255, 0.08);
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
}

/* ── Global overrides ─────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
}

/* ── Title gradient ───────────────────────────────────────────────────── */
.main-title {
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    line-height: 1.2;
}
.sub-title {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* ── Glassmorphism card ───────────────────────────────────────────────── */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-glow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(88, 166, 255, 0.12);
}

/* ── Result card ──────────────────────────────────────────────────────── */
.result-card {
    background: var(--gradient-card);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow-glow);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    animation: fadeSlideIn 0.35s ease forwards;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 30px rgba(88, 166, 255, 0.14);
}
.result-card-content {
    color: var(--text-primary);
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.result-card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Metadata badges ──────────────────────────────────────────────────── */
.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid rgba(139, 148, 158, 0.2);
    background: rgba(139, 148, 158, 0.08);
    color: var(--text-secondary);
    transition: all 0.2s ease;
}
.meta-badge.matched {
    background: rgba(88, 166, 255, 0.12);
    border-color: rgba(88, 166, 255, 0.35);
    color: var(--accent-blue);
    font-weight: 600;
    box-shadow: 0 0 8px rgba(88, 166, 255, 0.15);
    animation: pulseGlow 2s ease-in-out infinite;
}
.meta-badge .meta-key {
    opacity: 0.7;
}

/* ── Status dot ───────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-dot.green { background: var(--accent-green); box-shadow: 0 0 6px rgba(63,185,80,0.5); }
.status-dot.red   { background: var(--accent-red);   box-shadow: 0 0 6px rgba(248,81,73,0.5); }

/* ── Latency pill ─────────────────────────────────────────────────────── */
.latency-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.45rem 0.9rem;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    font-weight: 600;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    color: var(--text-primary);
}
.latency-pill .latency-value {
    color: var(--accent-cyan);
}

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}

/* ── Sidebar info card ────────────────────────────────────────────────── */
.sidebar-info-card {
    background: rgba(30, 37, 48, 0.5);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.sidebar-info-card .info-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.sidebar-info-card .info-value {
    font-size: 0.95rem;
    color: var(--text-primary);
    font-weight: 600;
}

/* ── Expander styling ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Animations ───────────────────────────────────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(88,166,255,0.15); }
    50%      { box-shadow: 0 0 14px rgba(88,166,255,0.3); }
}

/* ── Streamlit overrides ──────────────────────────────────────────────── */
.stChatInput textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Inter', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 0.8rem 1rem;
}

/* ══════════════════════════════════════════════════════════════════════ */
/* Answer synthesis panel                                                 */
/* ══════════════════════════════════════════════════════════════════════ */
.answer-card {
    position: relative;
    background: linear-gradient(145deg, rgba(40,32,60,0.55) 0%, rgba(22,27,34,0.92) 100%);
    border: 1px solid rgba(188, 140, 255, 0.28);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.7rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 0 26px rgba(188, 140, 255, 0.09);
    animation: fadeSlideIn 0.35s ease forwards;
    overflow: hidden;
}
.answer-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-main);
    opacity: 0.85;
}
.answer-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
}
.answer-body {
    color: var(--text-primary);
    font-size: 1.06rem;
    line-height: 1.75;
}
.answer-body ul {
    margin: 0.5rem 0 0.2rem 1.1rem;
    padding: 0;
}
.answer-body li { margin-bottom: 0.3rem; }
.answer-body strong { color: #fff; font-weight: 600; }

/* Inline citation chips inside the answer text */
.cite-ref {
    display: inline-block;
    min-width: 1.25rem;
    text-align: center;
    padding: 0.05rem 0.35rem;
    margin: 0 0.12rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    vertical-align: 0.12rem;
    background: rgba(88, 166, 255, 0.16);
    border: 1px solid rgba(88, 166, 255, 0.4);
    color: var(--accent-blue);
    cursor: help;
}

/* Faithfulness indicator */
.faith-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
    cursor: help;
}
.faith-meter {
    height: 5px;
    width: 100%;
    border-radius: 3px;
    background: rgba(139, 148, 158, 0.16);
    margin-top: 0.9rem;
    overflow: hidden;
}
.faith-meter-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.faith-note {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.45rem;
    line-height: 1.5;
}
.unsupported-claim {
    display: block;
    font-size: 0.82rem;
    color: #ffb9a8;
    background: rgba(248, 81, 73, 0.09);
    border-left: 2px solid var(--accent-red);
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    margin-top: 0.4rem;
}

/* Citation source list */
.citation-item {
    display: flex;
    gap: 0.7rem;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    background: rgba(30, 37, 48, 0.5);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-sm);
}
.citation-num {
    flex-shrink: 0;
    width: 1.6rem;
    height: 1.6rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
    background: rgba(88, 166, 255, 0.16);
    border: 1px solid rgba(88, 166, 255, 0.4);
    color: var(--accent-blue);
}
.citation-label {
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--text-primary);
}
.citation-snippet {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-top: 0.2rem;
}

/* Clarification prompt */
.clarify-card {
    background: linear-gradient(145deg, rgba(60,45,25,0.45) 0%, rgba(22,27,34,0.9) 100%);
    border: 1px solid rgba(240, 136, 62, 0.35);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 0 22px rgba(240, 136, 62, 0.08);
    animation: fadeSlideIn 0.35s ease forwards;
}
.clarify-question {
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffd9b8;
    line-height: 1.6;
}

/* ══════════════════════════════════════════════════════════════════════ */
/* Agent execution trace                                                  */
/* ══════════════════════════════════════════════════════════════════════ */
.trace-step {
    display: flex;
    gap: 0.8rem;
    padding: 0.55rem 0.2rem 0.55rem 0;
    position: relative;
}
.trace-rail {
    flex-shrink: 0;
    width: 26px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.trace-node-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.trace-line {
    flex: 1;
    width: 2px;
    background: rgba(99, 140, 255, 0.18);
    margin-top: 3px;
    min-height: 12px;
}
.trace-body { flex: 1; min-width: 0; }
.trace-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.trace-node-name {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-muted);
    background: rgba(139, 148, 158, 0.1);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
}
.trace-decision {
    font-size: 0.84rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
}
.trace-detail {
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    line-height: 1.5;
    word-break: break-word;
}
.trace-time {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--accent-cyan);
    margin-left: auto;
    white-space: nowrap;
}
.mode-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(57, 210, 192, 0.12);
    border: 1px solid rgba(57, 210, 192, 0.35);
    color: var(--accent-cyan);
}

/* ══════════════════════════════════════════════════════════════════════ */
/* Slot-based memory panel                                                */
/* ══════════════════════════════════════════════════════════════════════ */
.slot-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.35rem;
    background: rgba(30, 37, 48, 0.5);
    border: 1px solid var(--border-glass);
    border-left-width: 3px;
    border-radius: var(--radius-sm);
    font-size: 0.84rem;
}
.slot-field {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600;
    color: var(--text-primary);
}
.slot-value {
    color: var(--text-secondary);
    word-break: break-word;
}
.slot-tag {
    margin-left: auto;
    flex-shrink: 0;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.slot-correction-note {
    font-size: 0.78rem;
    color: #ffd9b8;
    background: rgba(240, 136, 62, 0.10);
    border-left: 2px solid var(--accent-orange);
    border-radius: 4px;
    padding: 0.45rem 0.65rem;
    margin: 0.15rem 0 0.5rem;
    line-height: 1.5;
}
.slot-empty {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-style: italic;
    padding: 0.4rem 0;
}
.restored-banner {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.6rem 1rem;
    margin-bottom: 0.9rem;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--accent-cyan);
    background: rgba(57, 210, 192, 0.09);
    border: 1px solid rgba(57, 210, 192, 0.3);
    animation: fadeSlideIn 0.35s ease forwards;
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract filter keys from a MongoDB pre_filter dict
# ──────────────────────────────────────────────────────────────────────────────
def _extract_filter_keys(filter_dict: dict) -> set:
    """Recursively pull out all non-operator field names from a MongoDB filter."""
    keys = set()
    if not isinstance(filter_dict, dict):
        return keys
    for k, v in filter_dict.items():
        if k.startswith("$"):
            if isinstance(v, list):
                for item in v:
                    keys |= _extract_filter_keys(item)
            elif isinstance(v, dict):
                keys |= _extract_filter_keys(v)
        else:
            keys.add(k)
    return keys


# ──────────────────────────────────────────────────────────────────────────────
# Helper: render a result card with highlighted matched metadata
# ──────────────────────────────────────────────────────────────────────────────
def _render_result_card(doc, matched_keys: set, index: int):
    """Render a single document result as a glassmorphism card."""
    content = doc.page_content
    metadata = doc.metadata

    badges_html = ""
    for key, val in metadata.items():
        is_matched = key in matched_keys
        cls = "meta-badge matched" if is_matched else "meta-badge"
        icon = "✦ " if is_matched else ""
        display_val = ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)
        badges_html += (
            f'<span class="{cls}">'
            f'<span class="meta-key">{icon}{key}:</span> {display_val}'
            f"</span>"
        )

    st.markdown(
        f"""
        <div class="result-card" style="animation-delay: {index * 0.08}s">
            <div class="result-card-content">{content}</div>
            <div class="result-card-meta">{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: answer panel rendering (feature 1)
# ──────────────────────────────────────────────────────────────────────────────
_FAITH_STYLE = {
    "grounded":   ("✅", "Grounded",            "#3fb950"),
    "partial":    ("⚠️", "Partially grounded",  "#f0883e"),
    "ungrounded": ("🚫", "Not grounded",        "#f85149"),
    "unknown":    ("❔", "Unverified",           "#8b949e"),
}


def _answer_to_html(answer: str, citation_lookup: dict) -> str:
    """Render the answer as safe HTML, turning [n] markers into citation chips."""
    text = html.escape(answer or "")

    # Minimal markdown: **bold** and *italic*
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)

    # Citation markers → chips whose tooltip names the source
    def _chip(match):
        num = match.group(1)
        source = citation_lookup.get(int(num), f"Document {num}")
        return f'<span class="cite-ref" title="{html.escape(str(source))}">{num}</span>'

    text = re.sub(r"\[(\d+)\]", _chip, text)

    # Bullet lists
    lines = text.split("\n")
    out, in_list = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:].strip()}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line + "<br>" if stripped else "<br>")
    if in_list:
        out.append("</ul>")

    return "".join(out).replace("<br><br>", "<br>")


def _render_answer_panel(entry: dict):
    """Answer + citations + faithfulness indicator, above the result cards."""
    answer = entry.get("answer") or ""
    citations = entry.get("citations") or []
    faith = entry.get("faithfulness") or {}

    if not answer:
        return

    verdict = faith.get("verdict", "unknown")
    icon, label, color = _FAITH_STYLE.get(verdict, _FAITH_STYLE["unknown"])
    score = float(faith.get("score", 0.0) or 0.0)
    method = faith.get("method", "")
    reason = faith.get("reason", "")
    unsupported = faith.get("unsupported_claims") or []

    lookup = {c["id"]: c["label"] for c in citations}
    body_html = _answer_to_html(answer, lookup)

    tooltip = html.escape(reason or f"Faithfulness verdict: {label}")
    method_note = {
        "llm": "LLM auditor",
        "heuristic": "lexical fallback check",
        "skipped": "not run",
    }.get(method, method)

    claims_html = "".join(
        f'<span class="unsupported-claim">⚠ Unsupported: {html.escape(str(c))}</span>'
        for c in unsupported[:4]
    )

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-head">
                <span class="section-label" style="margin-bottom:0">💬 Answer</span>
                <span class="faith-badge" title="{tooltip}"
                      style="background:{color}1f;border:1px solid {color}59;color:{color};">
                    {icon} {label} · {score:.0%}
                </span>
            </div>
            <div class="answer-body">{body_html}</div>
            <div class="faith-meter">
                <div class="faith-meter-fill"
                     style="width:{max(score, 0.02) * 100:.0f}%;background:{color};"></div>
            </div>
            <div class="faith-note">
                Faithfulness: <strong style="color:{color}">{label}</strong> —
                {html.escape(reason) if reason else "no detail provided"}
                <em>({method_note})</em>
            </div>
            {claims_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if citations:
        with st.expander(f"📚 Sources cited ({len(citations)})", expanded=False):
            items = ""
            for c in citations:
                locator = ""
                if c.get("source_file"):
                    page = c.get("page_number")
                    locator = (
                        f' <span style="color:#6e7681;font-weight:400">'
                        f'· chunk #{c.get("chunk_index")}</span>'
                        if page is not None
                        else ""
                    )
                items += (
                    f'<div class="citation-item">'
                    f'<div class="citation-num">{c["id"]}</div>'
                    f"<div><div class=\"citation-label\">{html.escape(str(c['label']))}"
                    f"{locator}</div>"
                    f'<div class="citation-snippet">{html.escape(str(c.get("snippet", "")))}</div>'
                    f"</div></div>"
                )
            st.markdown(items, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: slot-memory panel rendering
# ──────────────────────────────────────────────────────────────────────────────
_SLOT_STATUS_STYLE = {
    "set":     ("SET",     "#58a6ff"),
    "updated": ("UPDATED", "#f0883e"),
    "cleared": ("CLEARED", "#f85149"),
    "carried": ("CARRIED", "#6e7681"),
}


def _format_slot_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _render_slot_panel(entry: dict):
    """Show per-field slot state and what changed this turn."""
    rows = entry.get("slot_display") or []
    corrections = entry.get("slot_corrections") or []
    resolution_mode = entry.get("resolution_mode", "legacy")

    st.markdown(
        '<div class="section-label">🎰 Slot Memory (per-field state)</div>',
        unsafe_allow_html=True,
    )

    if resolution_mode != "slots":
        st.caption(
            "⏭️ Slot tracking not used for this turn — the original follow-up "
            "resolver handled it (document mode, no schema, or a fallback)."
        )
        return

    # Corrections first: they are the whole point of slot-level tracking.
    for c in corrections:
        before = c.get("from") or {}
        after = c.get("to")
        before_txt = (
            f"{before.get('operator', '')} {_format_slot_value(before.get('value'))}"
            if before else "(unset)"
        )
        after_txt = (
            f"{after.get('operator', '')} {_format_slot_value(after.get('value'))}"
            if after else "(cleared)"
        )
        st.markdown(
            f'<div class="slot-correction-note">✏️ <strong>Correction detected</strong> on '
            f'<code>{html.escape(str(c.get("field", "")))}</code>: '
            f"{html.escape(before_txt)} → {html.escape(after_txt)}"
            f'{" — " + html.escape(str(c.get("reason"))) if c.get("reason") else ""}'
            f"<br><span style=\"opacity:.75\">Only this field changed; every other slot "
            f"carried forward.</span></div>",
            unsafe_allow_html=True,
        )

    if not rows:
        st.markdown(
            '<div class="slot-empty">No slots set — nothing is being carried between turns yet.</div>',
            unsafe_allow_html=True,
        )
        return

    html_rows = ""
    for r in rows:
        label, color = _SLOT_STATUS_STYLE.get(r.get("status", "carried"), ("—", "#6e7681"))
        value_txt = _format_slot_value(r.get("value"))
        struck = "text-decoration:line-through;opacity:.6" if r.get("status") == "cleared" else ""
        html_rows += (
            f'<div class="slot-row" style="border-left-color:{color}">'
            f'<span class="slot-field">{html.escape(str(r.get("field", "")))}</span>'
            f'<span class="slot-value" style="{struck}">'
            f'{html.escape(str(r.get("operator", "")))} '
            f"{html.escape(value_txt)}</span>"
            f'<span class="slot-tag" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55">{label}</span>'
            f"</div>"
        )
    st.markdown(html_rows, unsafe_allow_html=True)

    if entry.get("slot_filter"):
        st.caption("Slot state compiled to a MongoDB filter:")
        st.json(entry["slot_filter"], expanded=False)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: agent execution trace rendering (feature 3)
# ──────────────────────────────────────────────────────────────────────────────
_TRACE_STATUS_COLOR = {
    "ok": "#3fb950",
    "branch": "#bc8cff",
    "skip": "#6e7681",
    "warn": "#f0883e",
}


def _render_trace(trace: list, relaxations: list = None):
    """Render the LangGraph node trace as a vertical timeline."""
    if not trace:
        st.caption("No trace recorded for this turn.")
        return

    steps = ""
    for i, step in enumerate(trace):
        color = _TRACE_STATUS_COLOR.get(step.get("status", "ok"), "#8b949e")
        is_last = i == len(trace) - 1
        line = "" if is_last else '<div class="trace-line"></div>'
        elapsed = step.get("elapsed", 0.0) or 0.0
        time_html = (
            f'<span class="trace-time">{elapsed:.2f}s</span>' if elapsed >= 0.005 else ""
        )
        detail = step.get("detail") or ""
        detail_html = (
            f'<div class="trace-detail">{html.escape(str(detail))}</div>' if detail else ""
        )

        steps += (
            f'<div class="trace-step">'
            f'<div class="trace-rail">'
            f'<div class="trace-node-dot" style="background:{color};'
            f'box-shadow:0 0 8px {color}80"></div>{line}</div>'
            f'<div class="trace-body">'
            f'<div class="trace-title">{i + 1}. {html.escape(str(step.get("title", "")))}'
            f'<span class="trace-node-name">{html.escape(str(step.get("node", "")))}</span>'
            f"{time_html}</div>"
            f'<div class="trace-decision" style="color:{color}">'
            f'{html.escape(str(step.get("decision", "")))}</div>'
            f"{detail_html}"
            f"</div></div>"
        )

    st.markdown(steps, unsafe_allow_html=True)

    if relaxations:
        st.markdown("---")
        st.markdown(
            '<div class="section-label">🔓 Filter Relaxations</div>',
            unsafe_allow_html=True,
        )
        for r in relaxations:
            st.caption(f"Retry {r.get('attempt')}: {r.get('description')}")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.caption("Before")
                st.json(r.get("before", {}), expanded=False)
            with rc2:
                st.caption("After")
                st.json(r.get("after", {}) or {"(no filter)": "pure vector search"},
                        expanded=False)


# ──────────────────────────────────────────────────────────────────────────────
# Initialise session state
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "dataset_mode" not in st.session_state:
    st.session_state.dataset_mode = "default"  # "default" or "custom"
if "custom_metadata_field_info" not in st.session_state:
    st.session_state.custom_metadata_field_info = None
if "custom_content_description" not in st.session_state:
    st.session_state.custom_content_description = None
if "custom_content_column" not in st.session_state:
    st.session_state.custom_content_column = None
if "custom_collection_name" not in st.session_state:
    st.session_state.custom_collection_name = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "schema_detected" not in st.session_state:
    st.session_state.schema_detected = False
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []   # [{query, filter_used}]
if "active_filters" not in st.session_state:
    st.session_state.active_filters = {}          # last merged_filter

# ── New: query mode + document-path state ────────────────────────────────
if "query_mode" not in st.session_state:
    st.session_state.query_mode = MODE_STRUCTURED   # "structured" | "documents"
if "doc_collection_name" not in st.session_state:
    st.session_state.doc_collection_name = default_document_collection()
if "doc_ingestion_done" not in st.session_state:
    st.session_state.doc_ingestion_done = False
if "doc_ingest_summary" not in st.session_state:
    st.session_state.doc_ingest_summary = None

# ── New: slot-based multi-turn memory ────────────────────────────────────
if "slot_state" not in st.session_state:
    st.session_state.slot_state = {}      # {field: serialised Slot}
if "turn_index" not in st.session_state:
    st.session_state.turn_index = 0


# ── New: conversation persistence (MongoDB) ──────────────────────────────
# The session id lives in the URL (?sid=...) rather than only in
# st.session_state, because session_state is wiped by a page reload — putting
# it in the query string is what makes a reload actually round-trip.
if "session_id" not in st.session_state:
    _sid_from_url = st.query_params.get("sid")
    st.session_state.session_restored = False
    st.session_state.restored_turn_count = 0

    if _sid_from_url:
        st.session_state.session_id = str(_sid_from_url)
        _restored = rehydrate_session(st.session_state.session_id)
        if _restored.found:
            st.session_state.history = history_entries_from_turns(_restored.turns)
            st.session_state.conversation_history = _restored.conversation_history
            st.session_state.active_filters = _restored.active_filters
            st.session_state.slot_state = _restored.slot_state
            st.session_state.turn_index = _restored.turn_index
            st.session_state.session_restored = True
            st.session_state.restored_turn_count = len(_restored.turns)
            if _restored.mode in (MODE_STRUCTURED, MODE_DOCUMENTS):
                st.session_state.query_mode = _restored.mode
    else:
        st.session_state.session_id = new_session_id()
        try:
            st.query_params["sid"] = st.session_state.session_id
        except Exception as _e:          # older Streamlit / embedded contexts
            logger.warning(f"Could not put session id in the URL: {_e}")


def _reset_conversation_state():
    """Clear per-conversation state (used when switching modes/datasets).

    Starts a NEW persisted session so a fresh conversation never appends to
    the previous one's stored turns.
    """
    st.session_state.history = []
    st.session_state.conversation_history = []
    st.session_state.active_filters = {}
    st.session_state.slot_state = {}
    st.session_state.turn_index = 0
    st.session_state.session_restored = False
    st.session_state.restored_turn_count = 0
    st.session_state.session_id = new_session_id()
    try:
        st.query_params["sid"] = st.session_state.session_id
    except Exception as _e:
        logger.warning(f"Could not update session id in the URL: {_e}")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:1.6rem;font-weight:800;margin-bottom:0;">'
        '🔎 <span style="background:linear-gradient(135deg,#58a6ff,#bc8cff);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        "SmartFilteringRAG</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8b949e;font-size:0.82rem;margin-top:0;">'
        "Intelligent metadata-aware vector search</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Query mode switch ────────────────────────────────────────────────
    st.markdown('<div class="section-label">🎛️ Query Mode</div>', unsafe_allow_html=True)

    _MODE_LABELS = {
        MODE_STRUCTURED: "📊 Structured dataset",
        MODE_DOCUMENTS: "📄 Documents (PDF / DOCX)",
    }
    _selected_mode = st.radio(
        "Query mode",
        options=[MODE_STRUCTURED, MODE_DOCUMENTS],
        format_func=lambda m: _MODE_LABELS[m],
        index=0 if st.session_state.query_mode == MODE_STRUCTURED else 1,
        key="query_mode_radio",
        label_visibility="collapsed",
    )
    if _selected_mode != st.session_state.query_mode:
        st.session_state.query_mode = _selected_mode
        _reset_conversation_state()
        st.rerun()

    _is_doc_mode = st.session_state.query_mode == MODE_DOCUMENTS

    st.caption(
        "Chunk retrieval → cited answer. Metadata filtering is skipped."
        if _is_doc_mode
        else "Natural language → metadata pre-filter → vector search → cited answer."
    )

    st.markdown("---")

    # ── Dataset info ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📂 Active Dataset</div>', unsafe_allow_html=True)

    if _is_doc_mode:
        _active_collection = st.session_state.doc_collection_name
        _mode_label = "Documents (PDF / DOCX)"
    else:
        _active_collection = (
            st.session_state.custom_collection_name
            if st.session_state.dataset_mode == "custom" and st.session_state.custom_collection_name
            else config["collection_name"]
        )
        _mode_label = "Custom" if st.session_state.dataset_mode == "custom" else "Default (Movies)"

    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="info-label">Mode</div>
            <div class="info-value">{_mode_label}</div>
        </div>
        <div class="sidebar-info-card">
            <div class="info-label">Database</div>
            <div class="info-value">{config["database_name"]}</div>
        </div>
        <div class="sidebar-info-card">
            <div class="info-label">Collection</div>
            <div class="info-value">{_active_collection}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document count
    try:
        _col = get_mongo_collection(
            db_name=config["database_name"],
            collection_name=_active_collection,
        )
        doc_count = _col.count_documents({})
    except Exception:
        doc_count = "—"

    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="info-label">{"Total Chunks" if _is_doc_mode else "Total Documents"}</div>
            <div class="info-value">{doc_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Model info ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🤖 Models</div>', unsafe_allow_html=True)
    _faith_model = os.getenv("FAITHFULNESS_MODEL") or config.get(
        "faithfulness_model"
    ) or config.get("model", "—")
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="info-label">LLM</div>
            <div class="info-value">{config.get("model", "—")}</div>
        </div>
        <div class="sidebar-info-card">
            <div class="info-label">Faithfulness Auditor</div>
            <div class="info-value">{_faith_model}</div>
        </div>
        <div class="sidebar-info-card">
            <div class="info-label">Embedding</div>
            <div class="info-value">{config.get("embedding_model", "—")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Environment status ────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚙️ Environment</div>', unsafe_allow_html=True)

    def _env_dot(var_name: str) -> str:
        exists = bool(os.getenv(var_name))
        color = "green" if exists else "red"
        label = "Set" if exists else "Missing"
        return (
            f'<span class="status-dot {color}"></span>'
            f"<code>{var_name}</code> — {label}"
        )

    st.markdown(
        f"""
        <div class="sidebar-info-card" style="line-height:1.9">
            {_env_dot("MONGO_URI")}<br>
            {_env_dot("OPEN_AI_API_KEY")}<br>
            {_env_dot("OPEN_API_BASE")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # DOCUMENT MODE — PDF / DOCX upload
    # ══════════════════════════════════════════════════════════════════════
    if _is_doc_mode:
        st.markdown(
            '<div class="section-label">📄 Upload Documents</div>', unsafe_allow_html=True
        )

        uploaded_docs = st.file_uploader(
            "Upload PDF or DOCX",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key="document_uploader",
            label_visibility="collapsed",
        )

        if uploaded_docs:
            st.caption(f"📎 {len(uploaded_docs)} file(s) ready")
            for f in uploaded_docs:
                st.caption(f"• {f.name} — {f.size / 1024:.0f} KB")

        _doc_coll = st.text_input(
            "Collection Name",
            value=st.session_state.doc_collection_name,
            key="doc_coll_name",
        )
        st.session_state.doc_collection_name = _doc_coll

        c_size, c_over = st.columns(2)
        with c_size:
            _chunk_size = st.number_input(
                "Chunk size",
                min_value=200,
                max_value=4000,
                value=int(config.get("chunk_size", 1000)),
                step=100,
                key="doc_chunk_size",
            )
        with c_over:
            _chunk_overlap = st.number_input(
                "Overlap",
                min_value=0,
                max_value=1000,
                value=int(config.get("chunk_overlap", 150)),
                step=25,
                key="doc_chunk_overlap",
            )

        _replace = st.checkbox(
            "Replace existing chunks",
            value=True,
            key="doc_replace",
            help="Uncheck to append these documents to the existing collection.",
        )

        if uploaded_docs and st.button(
            "✅ Ingest Documents", use_container_width=True, type="primary"
        ):
            progress_bar = st.progress(0, text="Starting ingestion…")
            status_text = st.empty()
            try:
                def _doc_progress_cb(step, pct):
                    progress_bar.progress(pct, text=step)
                    status_text.caption(step)

                files = [(f.name, f.getvalue()) for f in uploaded_docs]
                summary = ingest_documents(
                    files=files,
                    collection_name=st.session_state.doc_collection_name,
                    replace=_replace,
                    chunk_size=int(_chunk_size),
                    chunk_overlap=int(_chunk_overlap),
                    progress_callback=_doc_progress_cb,
                )
                st.session_state.doc_ingestion_done = True
                st.session_state.doc_ingest_summary = {
                    "files": summary.files_ingested,
                    "chunks": summary.chunks_inserted,
                    "pages": summary.pages_extracted,
                    "skipped": summary.skipped,
                    "warnings": summary.warnings,
                    "per_file": summary.per_file,
                }
                _reset_conversation_state()
                progress_bar.progress(
                    1.0, text=f"✅ {summary.chunks_inserted} chunks ingested!"
                )
                st.success(
                    f"Ingested {summary.chunks_inserted} chunks from "
                    f"{len(summary.files_ingested)} file(s) "
                    f"({summary.pages_extracted} pages)."
                )
                for w in summary.warnings:
                    st.warning(w)
                for s in summary.skipped:
                    st.error(f"Skipped {s['file']}: {s['reason']}")
                st.warning("⏳ Atlas Vector Search index may take ~30s to become active.")
            except Exception as e:
                st.error(f"Document ingestion failed: {e}")

        # ── Previously ingested sources ──────────────────────────────────
        _summary = st.session_state.doc_ingest_summary
        if _summary and _summary.get("per_file"):
            st.markdown(
                '<div class="section-label">📚 Ingested</div>', unsafe_allow_html=True
            )
            for pf in _summary["per_file"]:
                st.markdown(
                    f'<div class="sidebar-info-card">'
                    f'<div class="info-label">{pf["file"]}</div>'
                    f'<div class="info-value">{pf["chunks"]} chunks · '
                    f'{pf["pages"]} page(s)</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            _existing = list_ingested_sources(st.session_state.doc_collection_name)
            if _existing:
                st.markdown(
                    '<div class="section-label">📚 In Collection</div>',
                    unsafe_allow_html=True,
                )
                for src in _existing[:10]:
                    st.caption(f"• {src}")

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURED MODE — CSV / JSON upload (existing flow, unchanged)
    # ══════════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            '<div class="section-label">📤 Upload Your Dataset</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV or JSON",
            type=["csv", "json"],
            key="dataset_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                st.session_state.uploaded_df = df
                st.caption(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                st.dataframe(df.head(5), use_container_width=True, height=180)
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                st.session_state.uploaded_df = None

        # ── Schema Detection ─────────────────────────────────────────────
        if st.session_state.uploaded_df is not None and not st.session_state.schema_detected:
            if st.button("🔍 Detect Schema", use_container_width=True):
                with st.spinner("Analyzing schema with LLM…"):
                    try:
                        openai_api_key = os.getenv("OPEN_AI_API_KEY")
                        openai_api_base = os.getenv("OPEN_API_BASE")
                        default_headers = os.getenv("OPEN_API_DEFAULT_HEADERS")
                        default_headers = json.loads(default_headers) if default_headers else None
                        _llm = ChatOpenAI(
                            model=config["model"],
                            max_tokens=1024,
                            openai_api_key=openai_api_key,
                            openai_api_base=openai_api_base,
                            default_headers=default_headers,
                        )
                        fields, desc, content_col = extract_metadata_schema(
                            st.session_state.uploaded_df, _llm
                        )
                        st.session_state.custom_metadata_field_info = fields
                        st.session_state.custom_content_description = desc
                        st.session_state.custom_content_column = content_col
                        st.session_state.schema_detected = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Schema detection failed: {e}")

        # ── Schema Review & Edit ─────────────────────────────────────────
        if st.session_state.schema_detected and not st.session_state.ingestion_done:
            st.markdown(
                '<div class="section-label">🧬 Detected Schema</div>',
                unsafe_allow_html=True,
            )

            # Content column and description (editable)
            new_content_col = st.text_input(
                "Content Column",
                value=st.session_state.custom_content_column,
                key="edit_content_col",
            )
            st.session_state.custom_content_column = new_content_col

            new_desc = st.text_input(
                "Content Description",
                value=st.session_state.custom_content_description,
                key="edit_content_desc",
            )
            st.session_state.custom_content_description = new_desc

            # Metadata fields (editable)
            st.caption("Metadata Fields:")
            updated_fields = []
            for i, field in enumerate(st.session_state.custom_metadata_field_info):
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        new_field_desc = st.text_input(
                            f"{field.name}",
                            value=field.description,
                            key=f"field_desc_{i}",
                        )
                    with c2:
                        type_options = ["string", "integer", "float", "[string]"]
                        current_idx = (
                            type_options.index(field.type)
                            if field.type in type_options
                            else 0
                        )
                        new_type = st.selectbox(
                            "Type",
                            type_options,
                            index=current_idx,
                            key=f"field_type_{i}",
                            label_visibility="collapsed",
                        )
                    updated_fields.append(
                        AttributeInfo(
                            name=field.name, description=new_field_desc, type=new_type
                        )
                    )
            st.session_state.custom_metadata_field_info = updated_fields

            # Collection name
            coll_name = st.text_input(
                "Collection Name",
                value=f"custom-{uploaded_file.name.rsplit('.', 1)[0]}"
                if uploaded_file
                else "custom-dataset",
                key="edit_coll_name",
            )
            st.session_state.custom_collection_name = coll_name

            # Ingest button
            if st.button("✅ Confirm & Ingest", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Starting ingestion…")
                status_text = st.empty()
                try:
                    def _progress_cb(step, pct):
                        progress_bar.progress(pct, text=step)
                        status_text.caption(step)

                    doc_count = ingest_dataset(
                        df=st.session_state.uploaded_df,
                        metadata_field_info=st.session_state.custom_metadata_field_info,
                        content_column=st.session_state.custom_content_column,
                        collection_name=st.session_state.custom_collection_name,
                        progress_callback=_progress_cb,
                    )
                    st.session_state.dataset_mode = "custom"
                    st.session_state.ingestion_done = True
                    st.session_state.history = []  # clear old query history
                    progress_bar.progress(1.0, text=f"✅ {doc_count} documents ingested!")
                    st.success(f"Ingested {doc_count} documents into '{coll_name}'.")
                    st.warning("⏳ Atlas Vector Search index may take ~30s to become active.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        # ── Reset to Default ─────────────────────────────────────────────
        if st.session_state.dataset_mode == "custom":
            st.markdown("---")
            if st.button("🔄 Reset to Default (Movies)", use_container_width=True):
                st.session_state.dataset_mode = "default"
                st.session_state.custom_metadata_field_info = None
                st.session_state.custom_content_description = None
                st.session_state.custom_content_column = None
                st.session_state.custom_collection_name = None
                st.session_state.uploaded_df = None
                st.session_state.schema_detected = False
                st.session_state.ingestion_done = False
                st.session_state.history = []
                st.session_state.conversation_history = []
                st.session_state.active_filters = {}
                st.rerun()

    # ── Session / persistence ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">💾 Session</div>', unsafe_allow_html=True)

    _sid = st.session_state.get("session_id", "")
    _restored = st.session_state.get("session_restored", False)
    _persist_label = "Restored from MongoDB" if _restored else "Live (new session)"
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="info-label">Session ID</div>
            <div class="info-value" style="font-family:ui-monospace,monospace;font-size:0.78rem">
                {_sid[:12]}…
            </div>
        </div>
        <div class="sidebar-info-card">
            <div class="info-label">Conversation Log</div>
            <div class="info-value" style="font-size:0.85rem">{_persist_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Turns are saved to MongoDB and reload with the page. "
        "Bookmark the URL (it carries `?sid=`) to come back to this conversation."
    )

    # ── Reset Conversation ────────────────────────────────────────────────
    if st.session_state.conversation_history:
        if st.button("🗑️ Reset Conversation", use_container_width=True):
            # Starts a brand-new persisted session; earlier turns stay in Mongo
            # under the old session id.
            _reset_conversation_state()
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">SmartFilteringRAG</h1>', unsafe_allow_html=True)

_IS_DOC_MODE = st.session_state.query_mode == MODE_DOCUMENTS
if _IS_DOC_MODE:
    _subtitle = (
        "Ask a question about your uploaded documents — the agent retrieves the "
        "most relevant chunks and answers with citations back to file and page."
    )
    _mode_pill = "📄 Document mode"
else:
    _subtitle = (
        "Ask a natural-language question — the agent builds metadata filters, "
        "resolves time ranges, runs vector search, and answers with citations."
    )
    _mode_pill = "📊 Structured mode"

st.markdown(
    f'<p class="sub-title">{_subtitle} '
    f'<span class="mode-pill">{_mode_pill}</span></p>',
    unsafe_allow_html=True,
)

# ── Session restored from MongoDB ─────────────────────────────────────────
if st.session_state.get("session_restored"):
    _n = st.session_state.get("restored_turn_count", 0)
    _slots = st.session_state.get("slot_state") or {}
    _slot_note = (
        f" · slot memory restored ({', '.join(sorted(_slots))})" if _slots else ""
    )
    st.markdown(
        f'<div class="restored-banner">'
        f"<span>&#8635;</span>"
        f"<span>Session restored — {_n} previous turn(s) loaded from MongoDB"
        f"{html.escape(_slot_note)}.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner — LangGraph orchestration with step-by-step transparency
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(query: str, mode: str = None):
    """
    Execute the full SmartFilteringRAG pipeline and return intermediate
    results for each step with timing information.

    Orchestration runs through the LangGraph agent (rag/graph.py), which calls
    the same MetadataFilter / query_resolver / MongoDB logic as before — the
    graph only decides which of those steps to run.

    Returns a dict with keys:
        step1_filter, step2_filter, merged_filter,
        time_step1, time_step2, time_retrieval, time_resolve,
        documents, new_query,
        is_followup, filter_action, carried_filters, resolved_query,
        answer, citations, faithfulness, trace, relaxations,
        clarification_question, time_synthesis, time_faithfulness, mode
    """
    mode = mode or st.session_state.get("query_mode", MODE_STRUCTURED)

    openai_api_key = os.getenv("OPEN_AI_API_KEY")
    openai_api_base = os.getenv("OPEN_API_BASE")
    default_headers = os.getenv("OPEN_API_DEFAULT_HEADERS")
    default_headers = json.loads(default_headers) if default_headers else None

    llm = ChatOpenAI(
        model=config["model"],
        max_tokens=1024,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        default_headers=default_headers,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )

    database_name = config["database_name"]

    # ── Resolve collection + schema for the active mode ──────────────────
    if mode == MODE_DOCUMENTS:
        collection_name = st.session_state.get(
            "doc_collection_name", default_document_collection()
        )
        metadata_field_info = []
        document_content_description = "A chunk of text from an uploaded document"
    elif st.session_state.get("dataset_mode") == "custom" and st.session_state.get(
        "custom_metadata_field_info"
    ):
        metadata_field_info = st.session_state.custom_metadata_field_info
        document_content_description = st.session_state.custom_content_description
        collection_name = st.session_state.custom_collection_name
    else:
        document_content_description, metadata_field_info = get_docs_metadata()
        collection_name = config["collection_name"]

    collection = get_mongo_collection(db_name=database_name, collection_name=collection_name)

    # MetadataFilter is constructed exactly as before; in document mode the
    # graph never calls it.
    mf = MetadataFilter(
        collection=collection,
        llm=llm,
        metadata_field_info=metadata_field_info,
        document_content_description=document_content_description,
    )

    try:
        faithfulness_llm = make_faithfulness_llm()
    except Exception as e:
        logger.warning(f"Could not build faithfulness auditor, reusing main LLM: {e}")
        faithfulness_llm = llm

    ctx = PipelineContext(
        llm=llm,
        embeddings=embeddings,
        collection=collection,
        metadata_field_info=metadata_field_info,
        document_content_description=document_content_description,
        metadata_filter=mf,
        faithfulness_llm=faithfulness_llm,
        mode=mode,
        conversation_history=st.session_state.conversation_history,
        active_filters=st.session_state.active_filters,
        max_relaxations=int(config.get("max_filter_relaxations", 2)),
        enable_synthesis=bool(config.get("enable_answer_synthesis", True)),
        enable_faithfulness=bool(config.get("enable_faithfulness_check", True)),
        enable_clarification=bool(config.get("enable_clarification", True)),
        enable_slot_memory=bool(config.get("enable_slot_memory", True)),
        slot_state=st.session_state.get("slot_state", {}),
        turn_index=int(st.session_state.get("turn_index", 0)),
    )

    state = run_graph(query, ctx, mode=mode)

    # Flatten graph state into the result shape the UI already expects.
    return {
        "step1_filter": state.get("step1_filter", {}),
        "step2_filter": state.get("step2_filter"),
        "merged_filter": state.get("merged_filter", {}),
        "time_step1": state.get("time_step1", 0.0),
        "time_step2": state.get("time_step2", 0.0),
        "time_retrieval": state.get("time_retrieval", 0.0),
        "time_resolve": state.get("time_resolve", 0.0),
        "documents": state.get("documents", []),
        "new_query": state.get("new_query", query),
        "is_followup": state.get("is_followup", False),
        "filter_action": state.get("filter_action", "fresh"),
        "carried_filters": state.get("carried_filters", {}),
        "resolved_query": state.get("resolved_query", query),
        # ── New: synthesis + agentic orchestration ───────────────────────
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "faithfulness": state.get("faithfulness", {}),
        "trace": state.get("trace", []),
        "relaxations": state.get("relaxations", []),
        "clarification_question": state.get("clarification_question", ""),
        "needs_clarification": state.get("needs_clarification", False),
        "time_synthesis": state.get("time_synthesis", 0.0),
        "time_faithfulness": state.get("time_faithfulness", 0.0),
        "mode": mode,
        # ── New: slot-based multi-turn memory ────────────────────────────
        "resolution_mode": state.get("resolution_mode", "legacy"),
        "slot_state": state.get("slot_state", {}),
        "slot_ops": state.get("slot_ops", []),
        "slot_corrections": state.get("slot_corrections", []),
        "slot_display": state.get("slot_display", []),
        "slot_filter": state.get("slot_filter", {}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Render past history (results scroll area above input)
# ──────────────────────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    _entry_mode = entry.get("mode", MODE_STRUCTURED)
    _entry_is_doc = _entry_mode == MODE_DOCUMENTS

    # ── User query bubble ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 3px solid #58a6ff;">
            <div class="section-label">Your Query</div>
            <div style="font-size:1.05rem;font-weight:500;color:var(--text-primary);">
                {entry["query"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Clarification branch: the agent asked instead of guessing ─────────
    if entry.get("clarification_question"):
        st.markdown(
            f"""
            <div class="clarify-card">
                <div class="section-label">🤔 Need a little more context</div>
                <div class="clarify-question">{html.escape(entry["clarification_question"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🧭 Agent Execution Trace", expanded=False):
            _render_trace(entry.get("trace", []), entry.get("relaxations", []))
        st.markdown("<br>", unsafe_allow_html=True)
        continue

    # ── Answer + citations + faithfulness (above the result cards) ────────
    _render_answer_panel(entry)

    # ── Latency metrics ───────────────────────────────────────────────────
    _t = entry["timings"]
    if _entry_is_doc:
        lcol0, lcol1, lcol2, lcol3, lcol4 = st.columns(5)
        with lcol0:
            st.metric("🔗 Resolve", f'{_t.get("resolve", 0.0):.2f}s')
        with lcol1:
            st.metric("🔍 Chunk Retrieval", f'{_t["retrieval"]:.2f}s')
        with lcol2:
            st.metric("💬 Answer", f'{_t.get("synthesis", 0.0):.2f}s')
        with lcol3:
            st.metric("🛡️ Faithfulness", f'{_t.get("faithfulness", 0.0):.2f}s')
        with lcol4:
            total = sum(
                _t.get(k, 0.0)
                for k in ("resolve", "step1", "step2", "retrieval", "synthesis", "faithfulness")
            )
            st.metric("Σ Total", f"{total:.2f}s")
    else:
        lcol0, lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(6)
        with lcol0:
            st.metric("🔗 Resolve", f'{_t.get("resolve", 0.0):.2f}s')
        with lcol1:
            st.metric("⚡ Filter Generation", f'{_t["step1"]:.2f}s')
        with lcol2:
            st.metric("⏳ Time-Range Query", f'{_t["step2"]:.2f}s')
        with lcol3:
            st.metric("🔍 Vector Retrieval", f'{_t["retrieval"]:.2f}s')
        with lcol4:
            st.metric("💬 Answer + Check", f'{_t.get("synthesis", 0.0) + _t.get("faithfulness", 0.0):.2f}s')
        with lcol5:
            total = sum(
                _t.get(k, 0.0)
                for k in ("resolve", "step1", "step2", "retrieval", "synthesis", "faithfulness")
            )
            st.metric("Σ Total", f"{total:.2f}s")

    # ── Agent trace + filter transparency, side by side ───────────────────
    tcol, fcol = st.columns(2)

    with tcol:
        _n_nodes = len(entry.get("trace", []))
        with st.expander(f"🧭 Agent Execution Trace ({_n_nodes} nodes)", expanded=False):
            _render_trace(entry.get("trace", []), entry.get("relaxations", []))

    with fcol:
        with st.expander("🔍 What the system understood", expanded=False):
            # ── Conversation context ─────────────────────────────────────
            _is_fu = entry.get("is_followup", False)
            _action = entry.get("filter_action", "fresh")
            _resolved = entry.get("resolved_query", entry["query"])
            _carried = entry.get("carried_filters", {})

            _ctx_type = "🔗 Follow-up query" if _is_fu else "🆕 Fresh query"
            _ctx_color = "#bc8cff" if _is_fu else "#58a6ff"
            st.markdown(
                f'<div class="sidebar-info-card" style="border-left:3px solid {_ctx_color};margin-bottom:0.6rem">'
                f'<div class="info-label">Conversation Context</div>'
                f'<div class="info-value" style="font-size:0.9rem">{_ctx_type} — action: <code>{_action}</code></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if _is_fu and _resolved != entry["query"]:
                st.caption(f"📝 Resolved query: *\"{_resolved}\"*")
            if _carried:
                st.markdown(
                    '<div class="section-label" style="margin-top:0.4rem">🔗 Carried Filters (from previous turn)</div>',
                    unsafe_allow_html=True,
                )
                st.json(_carried, expanded=False)
                st.markdown("---")

            _render_slot_panel(entry)
            st.markdown("---")

            if _entry_is_doc:
                st.info(
                    "📄 Document mode — metadata filter generation is skipped; "
                    "chunks are retrieved by pure vector similarity."
                )
            else:
                st.markdown(
                    '<div class="section-label">Step 1 — Metadata Filter</div>',
                    unsafe_allow_html=True,
                )
                if entry["filters"]["step1"]:
                    st.json(entry["filters"]["step1"], expanded=True)
                else:
                    st.info("No metadata filter generated (NO_FILTER)")

                st.markdown(
                    '<div class="section-label">Step 2 — Time Range Filter</div>',
                    unsafe_allow_html=True,
                )
                if entry["filters"]["step2"]:
                    st.json(entry["filters"]["step2"], expanded=True)
                else:
                    st.caption("⏭️ Not triggered — no temporal keywords detected")

                st.markdown("---")
                st.markdown(
                    '<div class="section-label">Final Merged Pre-Filter → MongoDB Atlas Vector Search</div>',
                    unsafe_allow_html=True,
                )
                if entry["filters"]["merged"]:
                    st.json(entry["filters"]["merged"], expanded=True)
                else:
                    st.info("No pre-filter applied — pure vector search")

    # ── Result cards ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-label">📄 Retrieved '
        f'{"Chunks" if _entry_is_doc else "Documents"}</div>',
        unsafe_allow_html=True,
    )
    matched_keys = _extract_filter_keys(entry["filters"]["merged"])
    if entry["documents"]:
        for idx, doc in enumerate(entry["documents"]):
            _render_result_card(doc, matched_keys, idx)
    else:
        st.warning("No documents retrieved for this query.")

    st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chat input (pinned at bottom)
# ──────────────────────────────────────────────────────────────────────────────
_placeholder = (
    "Ask about your documents — e.g. 'What were the main risks mentioned?'"
    if _IS_DOC_MODE
    else "Ask a question — e.g. 'Recommend a latest anime movie'"
)
user_query = st.chat_input(_placeholder)

if user_query:
    with st.spinner("🧠 Running SmartFilteringRAG agent…"):
        pipeline_result = run_pipeline(user_query)

    # Update conversation state (skip when the agent asked for clarification —
    # an unresolved turn should not pollute the follow-up context).
    _turn_no = int(st.session_state.get("turn_index", 0))

    if not pipeline_result.get("clarification_question"):
        st.session_state.conversation_history.append({
            "query": user_query,
            "filter_used": pipeline_result["merged_filter"],
        })
        st.session_state.active_filters = pipeline_result["merged_filter"]
        # Carry slot state into the next turn (slot path only; the legacy
        # resolver leaves it untouched).
        if pipeline_result.get("resolution_mode") == "slots":
            st.session_state.slot_state = pipeline_result.get("slot_state", {})

    # Always advance the turn counter — including for clarification turns — so
    # every persisted turn gets its own slot in the log rather than overwriting
    # the previous one.
    st.session_state.turn_index = _turn_no + 1

    # Store in session state for re-rendering
    st.session_state.history.append(
        {
            "query": user_query,
            "documents": pipeline_result["documents"],
            "filters": {
                "step1": pipeline_result["step1_filter"],
                "step2": pipeline_result["step2_filter"],
                "merged": pipeline_result["merged_filter"],
            },
            "timings": {
                "resolve": pipeline_result["time_resolve"],
                "step1": pipeline_result["time_step1"],
                "step2": pipeline_result["time_step2"],
                "retrieval": pipeline_result["time_retrieval"],
                "synthesis": pipeline_result["time_synthesis"],
                "faithfulness": pipeline_result["time_faithfulness"],
            },
            "is_followup": pipeline_result["is_followup"],
            "filter_action": pipeline_result["filter_action"],
            "carried_filters": pipeline_result["carried_filters"],
            "resolved_query": pipeline_result["resolved_query"],
            # ── New ──────────────────────────────────────────────────────
            "answer": pipeline_result["answer"],
            "citations": pipeline_result["citations"],
            "faithfulness": pipeline_result["faithfulness"],
            "trace": pipeline_result["trace"],
            "relaxations": pipeline_result["relaxations"],
            "clarification_question": pipeline_result["clarification_question"],
            "mode": pipeline_result["mode"],
            # ── New: slot-based multi-turn memory ────────────────────────
            "resolution_mode": pipeline_result["resolution_mode"],
            "slot_state": pipeline_result["slot_state"],
            "slot_ops": pipeline_result["slot_ops"],
            "slot_corrections": pipeline_result["slot_corrections"],
            "slot_display": pipeline_result["slot_display"],
            "slot_filter": pipeline_result["slot_filter"],
        }
    )

    # ── Persist this turn (best-effort; never blocks the UI) ─────────────
    _persisted = log_turn(
        session_id=st.session_state.get("session_id", ""),
        turn_index=_turn_no,
        query=user_query,
        result=pipeline_result,
        mode=pipeline_result["mode"],
    )
    if not _persisted:
        logger.warning("Turn was not persisted — continuing without persistence.")

    st.rerun()
