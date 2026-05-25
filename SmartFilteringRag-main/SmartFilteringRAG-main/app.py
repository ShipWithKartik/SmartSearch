"""
SmartFilteringRAG — Streamlit UI
================================
A premium dark-themed interface for the SmartFilteringRAG pipeline.
Wraps existing pipeline functions without modifying them.
"""

# Load .env BEFORE any rag.* imports (mongodb_helper reads MONGO_URI at import time)
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
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
from rag.ingest import ingest_dataset
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

    # ── Dataset info ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📂 Active Dataset</div>', unsafe_allow_html=True)

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
            <div class="info-label">Total Documents</div>
            <div class="info-value">{doc_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Model info ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🤖 Models</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div class="info-label">LLM</div>
            <div class="info-value">{config.get("model", "—")}</div>
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

    # ── Upload Your Dataset ───────────────────────────────────────────────
    st.markdown('<div class="section-label">📤 Upload Your Dataset</div>', unsafe_allow_html=True)

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

    # ── Schema Detection ─────────────────────────────────────────────────
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

    # ── Schema Review & Edit ─────────────────────────────────────────────
    if st.session_state.schema_detected and not st.session_state.ingestion_done:
        st.markdown('<div class="section-label">🧬 Detected Schema</div>', unsafe_allow_html=True)

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
                    current_idx = type_options.index(field.type) if field.type in type_options else 0
                    new_type = st.selectbox(
                        "Type",
                        type_options,
                        index=current_idx,
                        key=f"field_type_{i}",
                        label_visibility="collapsed",
                    )
                updated_fields.append(
                    AttributeInfo(name=field.name, description=new_field_desc, type=new_type)
                )
        st.session_state.custom_metadata_field_info = updated_fields

        # Collection name
        coll_name = st.text_input(
            "Collection Name",
            value=f"custom-{uploaded_file.name.rsplit('.', 1)[0]}" if uploaded_file else "custom-dataset",
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

    # ── Reset to Default ─────────────────────────────────────────────────
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

    # ── Reset Conversation ────────────────────────────────────────────────
    if st.session_state.conversation_history:
        st.markdown("---")
        if st.button("🗑️ Reset Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.active_filters = {}
            st.session_state.history = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">SmartFilteringRAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Ask a natural-language question — the system builds '
    "metadata filters, resolves time ranges, and runs vector search for you.</p>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner — replicates orchestration for step-by-step transparency
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(query: str):
    """
    Execute the full SmartFilteringRAG pipeline and return intermediate
    results for each step with timing information.

    Returns a dict with keys:
        step1_filter, step2_filter, merged_filter,
        time_step1, time_step2, time_retrieval, time_resolve,
        documents, new_query,
        is_followup, filter_action, carried_filters, resolved_query
    """
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
    embeddings = HuggingFaceEmbeddings(model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

    database_name = config["database_name"]

    # ── Dynamic metadata: use custom schema or default movies ────────────
    if st.session_state.get("dataset_mode") == "custom" and st.session_state.get("custom_metadata_field_info"):
        metadata_field_info = st.session_state.custom_metadata_field_info
        document_content_description = st.session_state.custom_content_description
        collection_name = st.session_state.custom_collection_name
    else:
        document_content_description, metadata_field_info = get_docs_metadata()
        collection_name = config["collection_name"]

    collection = get_mongo_collection(db_name=database_name, collection_name=collection_name)

    mf = MetadataFilter(
        collection=collection,
        llm=llm,
        metadata_field_info=metadata_field_info,
        document_content_description=document_content_description,
    )

    result = {
        "step1_filter": {},
        "step2_filter": None,
        "merged_filter": {},
        "time_step1": 0.0,
        "time_step2": 0.0,
        "time_retrieval": 0.0,
        "time_resolve": 0.0,
        "documents": [],
        "new_query": query,
        "is_followup": False,
        "filter_action": "fresh",
        "carried_filters": {},
        "resolved_query": query,
    }

    # ── Step 0: Query Resolution (follow-up detection) ──────────────────────
    t_resolve = time.perf_counter()
    resolved = resolve_query(
        current_query=query,
        conversation_history=st.session_state.conversation_history,
        active_filters=st.session_state.active_filters,
        llm=llm,
    )
    result["time_resolve"] = time.perf_counter() - t_resolve
    result["is_followup"] = resolved.is_followup
    result["filter_action"] = resolved.filter_action
    result["carried_filters"] = resolved.carried_filters
    result["resolved_query"] = resolved.resolved_query

    # Use the resolved query for the pipeline
    effective_query = resolved.resolved_query

    # ── If keep_all: skip filter generation, reuse active filters ───────────
    if resolved.filter_action == "keep_all" and resolved.carried_filters:
        result["step1_filter"] = deepcopy({"pre_filter": resolved.carried_filters}) if resolved.carried_filters else {}
        result["merged_filter"] = resolved.carried_filters
        result["new_query"] = effective_query
    else:
        # ── Step 1: Metadata filter (query constructor → translator → enforce)
        t0 = time.perf_counter()
        wrapped_query = f"Answer the below question:\n\nQuestion: {effective_query}\n"
        query_constructor = mf.create_query_constructor()
        structured_query = query_constructor.invoke(wrapped_query)
        new_query, new_kwargs = mf.translator.visit_structured_query(structured_query)
        pre_filter = enforce_constraints(new_kwargs)
        result["time_step1"] = time.perf_counter() - t0
        result["step1_filter"] = deepcopy(pre_filter) if pre_filter else {}

        # ── Step 2: Time-based filter (only if step 1 produced a filter) ────
        step2_filter = None
        if pre_filter:
            t1 = time.perf_counter()
            try:
                time_based_pre_filter, new_query = mf.generate_time_based_filter(
                    pre_filter, new_query
                )
                if time_based_pre_filter:
                    step2_filter = deepcopy(time_based_pre_filter)
                    pre_filter["pre_filter"] = {
                        "$and": [pre_filter["pre_filter"], time_based_pre_filter["pre_filter"]]
                    }
            except Exception:
                logger.warning("Time-based filter step failed or returned NO_FILTER")
            result["time_step2"] = time.perf_counter() - t1
        result["step2_filter"] = step2_filter

        final_pre_filter = pre_filter.get("pre_filter", {}) if pre_filter else {}
        result["merged_filter"] = final_pre_filter
        result["new_query"] = new_query if new_query else effective_query

    # ── Step 3: Vector search retrieval ─────────────────────────────────────
    final_pre_filter = result["merged_filter"]
    t2 = time.perf_counter()
    vectorstore = MongoDBAtlasVectorSearch(collection, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"pre_filter": final_pre_filter})
    docs = retriever.invoke(result["new_query"])
    result["time_retrieval"] = time.perf_counter() - t2
    result["documents"] = docs

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Render past history (results scroll area above input)
# ──────────────────────────────────────────────────────────────────────────────
for entry in st.session_state.history:
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

    # ── Latency metrics ───────────────────────────────────────────────────
    lcol0, lcol1, lcol2, lcol3, lcol4 = st.columns(5)
    with lcol0:
        st.metric("🔗 Resolve", f'{entry["timings"].get("resolve", 0.0):.2f}s')
    with lcol1:
        st.metric("⚡ Filter Generation", f'{entry["timings"]["step1"]:.2f}s')
    with lcol2:
        st.metric("⏳ Time-Range Query", f'{entry["timings"]["step2"]:.2f}s')
    with lcol3:
        st.metric("🔍 Vector Retrieval", f'{entry["timings"]["retrieval"]:.2f}s')
    with lcol4:
        total = entry["timings"].get("resolve", 0.0) + entry["timings"]["step1"] + entry["timings"]["step2"] + entry["timings"]["retrieval"]
        st.metric("Σ Total", f"{total:.2f}s")

    # ── Filter transparency panel ─────────────────────────────────────────
    with st.expander("🔍 What the system understood", expanded=False):
        # ── Conversation context ─────────────────────────────────────────
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

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown(
                '<div class="section-label">Step 1 — Metadata Filter</div>',
                unsafe_allow_html=True,
            )
            if entry["filters"]["step1"]:
                st.json(entry["filters"]["step1"], expanded=True)
            else:
                st.info("No metadata filter generated (NO_FILTER)")

        with fcol2:
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
        '<div class="section-label">📄 Retrieved Documents</div>',
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
user_query = st.chat_input("Ask a question — e.g. 'Recommend a latest anime movie'")

if user_query:
    with st.spinner("🧠 Running SmartFilteringRAG pipeline…"):
        pipeline_result = run_pipeline(user_query)

    # Update conversation state
    st.session_state.conversation_history.append({
        "query": user_query,
        "filter_used": pipeline_result["merged_filter"],
    })
    st.session_state.active_filters = pipeline_result["merged_filter"]

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
            },
            "is_followup": pipeline_result["is_followup"],
            "filter_action": pipeline_result["filter_action"],
            "carried_filters": pipeline_result["carried_filters"],
            "resolved_query": pipeline_result["resolved_query"],
        }
    )
    st.rerun()
