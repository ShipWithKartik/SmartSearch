"""
answer_synthesis.py — Grounded Answer Generation + Faithfulness Check
=====================================================================
Runs AFTER retrieval. Takes the documents the existing pipeline already
retrieved and produces a natural-language answer that cites the specific
documents supporting each claim, then verifies that answer against those
same documents.

This module is purely additive: it never touches retrieval, filtering, or
ingestion. Callers hand it a list of LangChain Documents and get back a
dataclass they can render.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FaithfulnessReport:
    """Verdict on whether an answer is grounded in the retrieved documents."""

    verdict: str = "unknown"          # "grounded" | "partial" | "ungrounded" | "unknown"
    score: float = 0.0                # 0.0 - 1.0
    unsupported_claims: List[str] = field(default_factory=list)
    reason: str = ""
    method: str = "llm"               # "llm" | "heuristic" | "skipped"

    @property
    def label(self) -> str:
        return {
            "grounded": "Grounded",
            "partial": "Partially grounded",
            "ungrounded": "Not grounded",
            "unknown": "Unverified",
        }.get(self.verdict, "Unverified")

    @property
    def color(self) -> str:
        """Hex colour matching the app's existing accent palette."""
        return {
            "grounded": "#3fb950",     # --accent-green
            "partial": "#f0883e",      # --accent-orange
            "ungrounded": "#f85149",   # --accent-red
            "unknown": "#8b949e",      # --text-secondary
        }.get(self.verdict, "#8b949e")

    @property
    def icon(self) -> str:
        return {
            "grounded": "✅",
            "partial": "⚠️",
            "ungrounded": "🚫",
            "unknown": "❔",
        }.get(self.verdict, "❔")

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "unsupported_claims": list(self.unsupported_claims),
            "reason": self.reason,
            "method": self.method,
        }


@dataclass
class SynthesizedAnswer:
    """A cited answer plus its faithfulness verdict."""

    answer: str = ""
    citations: List[Dict] = field(default_factory=list)   # [{id, label, snippet, ...}]
    faithfulness: FaithfulnessReport = field(default_factory=FaithfulnessReport)
    time_synthesis: float = 0.0
    time_faithfulness: float = 0.0
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Document formatting / labelling
# ──────────────────────────────────────────────────────────────────────────────
def document_label(doc: Document, index: int) -> str:
    """Human-readable label for a document, used in the citation list.

    Unstructured chunks (from doc_ingest) carry source_file/page_number, so they
    become "report.pdf - p.4". Structured rows fall back to a distinctive
    metadata value, then to a plain ordinal.
    """
    meta = doc.metadata or {}

    source_file = meta.get("source_file")
    if source_file:
        page = meta.get("page_number")
        if page:
            return f"{source_file} — p.{page}"
        return str(source_file)

    # Structured records: prefer a short, name-like metadata field.
    for key in ("title", "name", "product_name", "movie", "director", "artist"):
        if meta.get(key):
            return str(meta[key])

    return f"Document {index}"


def format_documents_for_prompt(documents: List[Document], max_chars: int = 1200) -> str:
    """Render retrieved documents as a numbered context block for the LLM."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        content = (doc.page_content or "").strip()
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + " ..."

        meta = {k: v for k, v in (doc.metadata or {}).items() if k != "embedding"}
        meta_str = json.dumps(meta, default=str) if meta else "{}"

        blocks.append(
            f"[{i}] source: {document_label(doc, i)}\n"
            f"    metadata: {meta_str}\n"
            f"    content: {content}"
        )
    return "\n\n".join(blocks)


def build_citations(documents: List[Document], max_snippet: int = 240) -> List[Dict]:
    """Build the citation list the UI renders under the answer."""
    citations = []
    for i, doc in enumerate(documents, start=1):
        content = (doc.page_content or "").strip()
        snippet = content[:max_snippet].rstrip() + (" ..." if len(content) > max_snippet else "")
        meta = doc.metadata or {}
        citations.append(
            {
                "id": i,
                "label": document_label(doc, i),
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
                "snippet": snippet,
            }
        )
    return citations


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
_SYNTHESIS_PROMPT = """\
You are a careful research assistant. Answer the user's question using ONLY the \
numbered documents provided below. These documents are your entire world - you \
have no other knowledge.

## Retrieved Documents
{context}

## User Question
{query}

## Rules
- Use ONLY facts stated in the documents above. Never add outside knowledge, \
never guess, never fill gaps with plausible-sounding detail.
- Cite the supporting document after EVERY claim, using bracketed numbers that \
match the document numbers above: [1], [2], or [1][3] when several support it.
- Every sentence that states a fact must carry at least one citation.
- If the documents do not contain enough information to answer, say so plainly \
and state what IS available. Do not invent an answer.
- Do not refer to "the documents" or "the context" as objects - just answer the \
question naturally, with citations.
- Be concise: 1-5 sentences for a simple question, a short list when the answer \
is genuinely a list.
- Plain text or simple markdown only. No preamble like "Based on the documents".

## Answer
"""

_FAITHFULNESS_PROMPT = """\
You are a strict fact-checking auditor. You are given source documents and an \
answer that was supposedly derived from them. Your ONLY job is to decide whether \
every factual claim in the answer is supported by the documents.

## Source Documents
{context}

## Answer To Audit
{answer}

## Instructions
1. Break the answer into individual factual claims.
2. For each claim, check whether the documents explicitly support it.
3. A claim is UNSUPPORTED if it adds detail, numbers, names, dates, causes, or \
conclusions that do not appear in the documents - even if it sounds correct.
4. Statements that merely say information is unavailable are always supported.
5. Ignore style, tone, and citation formatting. Judge only factual grounding.

Return ONLY valid JSON, no markdown fences and no text outside the JSON:

{{"verdict": "grounded" | "partial" | "ungrounded",
  "score": <float 0.0-1.0, fraction of claims that are supported>,
  "unsupported_claims": ["exact claim text", ...],
  "reason": "<one short sentence>"}}

Use "grounded" when every claim is supported, "partial" when some are not, and \
"ungrounded" when the answer is largely unsupported by the documents.
"""

NO_DOCS_ANSWER = (
    "No documents were retrieved for this query, so there is nothing to answer from. "
    "Try relaxing the filters or rephrasing the question."
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _strip_code_fences(raw: str) -> str:
    """Remove ``` fences that chat models like to wrap JSON in."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that", "this",
    "these", "those", "it", "its", "there", "here", "which", "who", "whom", "what",
    "has", "have", "had", "not", "no", "do", "does", "did", "can", "could", "will",
    "would", "should", "may", "might", "about", "into", "than", "then", "also",
}


def _content_tokens(text: str) -> set:
    tokens = re.findall(r"[a-z0-9][a-z0-9'\-\.]*", (text or "").lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def heuristic_faithfulness(answer: str, documents: List[Document]) -> FaithfulnessReport:
    """Cheap, LLM-free grounding estimate used as a fallback.

    Splits the answer into sentences and measures what fraction of each
    sentence's content words appear anywhere in the retrieved documents. It is
    deliberately crude: a fallback signal, never a replacement for the LLM audit.
    """
    if not (answer or "").strip():
        return FaithfulnessReport(
            verdict="unknown", score=0.0, reason="Empty answer.", method="heuristic"
        )
    if not documents:
        return FaithfulnessReport(
            verdict="unknown",
            score=0.0,
            reason="No documents to verify against.",
            method="heuristic",
        )

    corpus_tokens = set()
    for doc in documents:
        corpus_tokens |= _content_tokens(doc.page_content)
        for value in (doc.metadata or {}).values():
            corpus_tokens |= _content_tokens(str(value))

    # Drop citation markers so "[1]" never counts as content.
    cleaned = re.sub(r"\[\d+\]", " ", answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]

    supported, unsupported = 0, []
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        if not tokens:
            supported += 1
            continue
        overlap = len(tokens & corpus_tokens) / len(tokens)
        if overlap >= 0.5:
            supported += 1
        else:
            unsupported.append(sentence)

    total = max(len(sentences), 1)
    score = supported / total

    if score >= 0.95:
        verdict = "grounded"
    elif score >= 0.5:
        verdict = "partial"
    else:
        verdict = "ungrounded"

    return FaithfulnessReport(
        verdict=verdict,
        score=round(score, 2),
        unsupported_claims=unsupported[:5],
        reason=f"Lexical overlap check: {supported}/{total} sentences matched the retrieved text.",
        method="heuristic",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def synthesize_answer(
    query: str,
    documents: List[Document],
    llm: BaseChatModel,
) -> Tuple[str, List[Dict]]:
    """Generate a cited natural-language answer from retrieved documents only.

    Args:
        query: The user's question (resolved form, if a follow-up).
        documents: Documents returned by vector search.
        llm: Configured chat model (Groq by default).

    Returns:
        (answer_text, citations)
    """
    if not documents:
        return NO_DOCS_ANSWER, []

    context = format_documents_for_prompt(documents)
    prompt = _SYNTHESIS_PROMPT.format(context=context, query=query)

    logger.info(f"Synthesizing answer over {len(documents)} documents")
    response = llm.invoke(prompt)
    answer = (response.content or "").strip()

    return answer, build_citations(documents)


def check_faithfulness(
    answer: str,
    documents: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> FaithfulnessReport:
    """Audit an answer against the documents it was supposed to come from.

    Uses an LLM auditor when one is supplied, and falls back to the lexical
    heuristic if the auditor is unavailable or returns unparseable output.
    """
    if not (answer or "").strip() or not documents:
        return FaithfulnessReport(
            verdict="unknown",
            score=0.0,
            reason="Nothing to verify.",
            method="skipped",
        )

    if llm is None:
        return heuristic_faithfulness(answer, documents)

    context = format_documents_for_prompt(documents)
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(_strip_code_fences(response.content))

        verdict = str(parsed.get("verdict", "unknown")).lower()
        if verdict not in ("grounded", "partial", "ungrounded"):
            verdict = "unknown"

        try:
            score = float(parsed.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        claims = parsed.get("unsupported_claims") or []
        if not isinstance(claims, list):
            claims = [str(claims)]

        report = FaithfulnessReport(
            verdict=verdict,
            score=round(score, 2),
            unsupported_claims=[str(c) for c in claims][:8],
            reason=str(parsed.get("reason", "")).strip(),
            method="llm",
        )
        logger.info(f"Faithfulness check -> {report.verdict} ({report.score})")
        return report

    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Faithfulness JSON parse failed ({e}) - falling back to heuristic")
        return heuristic_faithfulness(answer, documents)
    except Exception as e:
        logger.error(f"Faithfulness check failed ({e}) - falling back to heuristic")
        return heuristic_faithfulness(answer, documents)


def generate_grounded_answer(
    query: str,
    documents: List[Document],
    llm: BaseChatModel,
    faithfulness_llm: Optional[BaseChatModel] = None,
    run_faithfulness: bool = True,
) -> SynthesizedAnswer:
    """Synthesize a cited answer and audit it, with timings for the UI panel."""
    result = SynthesizedAnswer()

    if not documents:
        result.answer = NO_DOCS_ANSWER
        result.faithfulness = FaithfulnessReport(
            verdict="unknown", reason="No documents retrieved.", method="skipped"
        )
        return result

    t0 = time.perf_counter()
    try:
        result.answer, result.citations = synthesize_answer(query, documents, llm)
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        result.error = str(e)
        result.answer = ""
        result.citations = build_citations(documents)
        result.faithfulness = FaithfulnessReport(
            verdict="unknown", reason=f"Synthesis failed: {e}", method="skipped"
        )
        result.time_synthesis = time.perf_counter() - t0
        return result
    result.time_synthesis = time.perf_counter() - t0

    if run_faithfulness:
        t1 = time.perf_counter()
        result.faithfulness = check_faithfulness(
            result.answer, documents, faithfulness_llm or llm
        )
        result.time_faithfulness = time.perf_counter() - t1
    else:
        result.faithfulness = FaithfulnessReport(
            verdict="unknown", reason="Faithfulness check disabled.", method="skipped"
        )

    return result
