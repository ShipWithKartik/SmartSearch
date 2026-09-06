"""
graph.py — Agentic Pipeline Orchestration (LangGraph)
======================================================
Rebuilds the top-level orchestration (previously a fixed
resolve -> filter -> time-filter -> vector-search sequence) as a LangGraph
state machine.

Nothing inside the existing components changes: MetadataFilter, resolve_query
and the MongoDB tooling are called exactly as before — they simply become nodes
that the graph decides when to run.

Conditional branches added on top of the old fixed sequence:

  1. Zero results with a filter applied -> relax the most restrictive clause
     and retry (bounded by max_filter_relaxations).
  2. No discernible structured constraints -> skip filter generation entirely
     and go straight to vector search.
  3. Follow-up resolution is ambiguous -> ask a clarifying question instead of
     guessing at what the user meant.

Every node appends to `trace`, so the UI can show which nodes ran, why, and in
what order.
"""

import json
import logging
import operator
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Tuple

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from rag.answer_synthesis import FaithfulnessReport, generate_grounded_answer
from rag.config_loader import config
from rag.prompts import enforce_constraints
from rag.query_resolver import resolve_query
from rag.slot_memory import (
    resolve_with_slots,
    slot_state_from_dict,
    slot_state_to_dict,
    slot_state_to_display,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODE_STRUCTURED = "structured"
MODE_DOCUMENTS = "documents"


# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    """State passed between graph nodes.

    Fields annotated with operator.add accumulate across node visits (the
    retrieve node can run several times when filters are relaxed); everything
    else is overwritten by the most recent node to set it.
    """

    # Input
    query: str
    mode: str

    # Query resolution
    resolved_query: str
    effective_query: str
    is_followup: bool
    filter_action: str
    carried_filters: Dict
    needs_clarification: bool
    ambiguity_reason: str
    clarification_question: str

    # Slot-based multi-turn memory (rag/slot_memory.py)
    resolution_mode: str           # "slots" | "legacy"
    slot_state: Dict               # serialised {field: slot dict}
    slot_ops: List[Dict]           # the diff applied this turn
    slot_corrections: List[Dict]   # slots the user explicitly corrected
    slot_display: List[Dict]       # per-slot rows for the transparency panel
    slot_filter: Dict              # slot state compiled to a MongoDB filter

    # Constraint detection
    has_constraints: bool
    constraint_reason: str
    constraint_signals: List[str]

    # Filtering
    step1_filter: Dict
    step2_filter: Optional[Dict]
    merged_filter: Dict
    new_query: str

    # Retrieval
    documents: List[Any]
    retry_count: int
    relaxations: Annotated[List[Dict], operator.add]

    # Synthesis
    answer: str
    citations: List[Dict]
    faithfulness: Dict

    # Instrumentation
    time_resolve: float
    time_step1: float
    time_step2: float
    time_retrieval: Annotated[float, operator.add]
    time_synthesis: float
    time_faithfulness: float
    time_clarify: float
    trace: Annotated[List[Dict], operator.add]


@dataclass
class PipelineContext:
    """Everything the nodes need that is not per-turn state."""

    llm: BaseChatModel
    embeddings: Any
    collection: Any
    metadata_field_info: List[AttributeInfo] = field(default_factory=list)
    document_content_description: str = ""
    metadata_filter: Any = None                    # rag.metadata_filter.MetadataFilter
    faithfulness_llm: Optional[BaseChatModel] = None
    mode: str = MODE_STRUCTURED
    conversation_history: List[Dict] = field(default_factory=list)
    active_filters: Dict = field(default_factory=dict)
    max_relaxations: int = 2
    enable_synthesis: bool = True
    enable_faithfulness: bool = True
    enable_clarification: bool = True
    # Slot-based memory. slot_state carries in the previous turn's slots
    # (serialised form). Disabled or unavailable -> the original resolver runs.
    enable_slot_memory: bool = True
    slot_state: Dict = field(default_factory=dict)
    turn_index: int = 0


def _trace(node: str, title: str, decision: str, detail: str = "",
           elapsed: float = 0.0, status: str = "ok") -> Dict:
    """One entry in the visible execution trace."""
    return {
        "node": node,
        "title": title,
        "decision": decision,
        "detail": detail,
        "elapsed": round(elapsed, 3),
        "status": status,      # "ok" | "skip" | "branch" | "warn"
    }


# ──────────────────────────────────────────────────────────────────────────────
# Branch 3 helper — is the follow-up resolution ambiguous?
# ──────────────────────────────────────────────────────────────────────────────
# Words that only make sense against something said earlier.
_DANGLING_REFERENCES = (
    "those", "them", "these", "the same", "same", "more", "others", "any others",
    "what else", "the other", "previous", "again", "that one", "this one",
    "the first", "the last", "it", "both",
)

_CLARIFY_PROMPT = """\
A user asked a search system this question:

"{query}"

{history_note}

The question refers to something from an earlier turn, but there is not enough \
context to know what. Ask ONE short clarifying question (max 25 words) that would \
let you run the search. Ask only about what is genuinely missing.

Return only the question text — no preamble, no quotes, no explanation.
"""


def assess_ambiguity(
    query: str,
    resolved_query: str,
    is_followup: bool,
    filter_action: str,
    conversation_history: List[Dict],
) -> Tuple[bool, str]:
    """Decide whether follow-up resolution left the query genuinely unresolved.

    Deliberately conservative — it only fires on short queries that lean on a
    dangling reference AND that the resolver failed to rewrite. A query that was
    successfully rewritten, or that carries its own content words, proceeds
    normally, so ordinary queries never get diverted into clarification.

    Returns:
        (needs_clarification, reason)
    """
    q = (query or "").strip().lower()
    if not q:
        return True, "Empty query."

    words = q.split()
    resolved = (resolved_query or "").strip().lower()

    has_reference = any(
        re.search(rf"\b{re.escape(ref)}\b", q) for ref in _DANGLING_REFERENCES
    )
    if not has_reference:
        return False, ""

    # A long query carries enough of its own content to search on, even if it
    # also contains a pronoun.
    if len(words) > 6:
        return False, ""

    # The resolver rewrote it into something self-contained — trust that.
    if resolved and resolved != q:
        return False, ""

    if not conversation_history:
        return True, (
            "The query refers back to earlier results, but there is no "
            "conversation history to resolve it against."
        )

    if is_followup or filter_action == "modify":
        return True, (
            "Detected as a follow-up, but the resolver could not rewrite it into "
            "a self-contained query."
        )

    return True, "The query is too short and refers to unstated prior context."


# ──────────────────────────────────────────────────────────────────────────────
# Branch 2 helper — does the query contain structured constraints?
# ──────────────────────────────────────────────────────────────────────────────
_CONSTRAINT_PATTERNS = [
    (r"\b(under|over|above|below|less than|greater than|more than|at least|at most|"
     r"between|cheaper|pricier|expensive|older|newer|shorter|longer|higher|lower)\b",
     "comparison phrase"),
    (r"\b(before|after|since|during|released in|from)\s+\d{4}\b", "date comparison"),
    (r"\b(latest|most recent|recent|earliest|oldest|newest|first|last)\b",
     "temporal keyword"),
    (r"\b(19|20)\d{2}\b", "year"),
    (r"[$€£]\s?\d", "currency amount"),
    (r"\b(top|best|highest|lowest)\s+(rated|priced|\d+)\b", "ranking phrase"),
    (r"\b\d+(\.\d+)?\s*(stars?|out of|/10|%)\b", "numeric rating"),
]


def _keywords_from_description(description: str) -> List[str]:
    """Pull the enumerated values out of a field description.

    Descriptions in this project follow the convention
    "Keywords for filtering: ['anime', 'action', ...]" — those values are strong
    evidence that a query mentioning one implies a metadata constraint.
    """
    keywords = []
    for group in re.findall(r"\[([^\]]*)\]", description or ""):
        keywords.extend(re.findall(r"['\"]([^'\"]+)['\"]", group))
    return [k.strip().lower() for k in keywords if len(k.strip()) > 2]


def detect_structured_constraints(
    query: str,
    metadata_field_info: List[AttributeInfo],
) -> Tuple[bool, str, List[str]]:
    """Heuristically decide whether the query has anything worth filtering on.

    Runs before filter generation so a purely semantic query ("something
    heartwarming about friendship") can skip an LLM round-trip entirely.
    Deterministic and free — and because a miss only costs a filter that the
    query constructor would very likely have returned NO_FILTER for anyway, the
    downside of a false negative is small.

    Returns:
        (has_constraints, reason, matched_signals)
    """
    q = (query or "").lower()
    if not q.strip():
        return False, "Empty query.", []

    signals: List[str] = []

    for field_info in metadata_field_info or []:
        name = (
            field_info.name if isinstance(field_info, AttributeInfo)
            else field_info.get("name", "")
        )
        description = (
            field_info.description if isinstance(field_info, AttributeInfo)
            else field_info.get("description", "")
        )

        # Field name mentioned directly ("genre", "rating", "price").
        readable = str(name).replace("_", " ").lower()
        if readable and re.search(rf"\b{re.escape(readable)}\b", q):
            signals.append(f'field "{name}" mentioned')
            continue

        # A known value of that field mentioned ("anime", "Electronics").
        for keyword in _keywords_from_description(str(description)):
            if re.search(rf"\b{re.escape(keyword)}\b", q):
                signals.append(f'value "{keyword}" of field "{name}"')
                break

    for pattern, label in _CONSTRAINT_PATTERNS:
        if re.search(pattern, q):
            signals.append(label)

    # Deduplicate, preserving order.
    seen, unique = set(), []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    if unique:
        return True, "Structured signals found: " + "; ".join(unique[:4]), unique

    return (
        False,
        "No field names, known values, comparisons, or dates found — this reads as "
        "a purely semantic query.",
        [],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Branch 1 helper — relax the most restrictive clause of a filter
# ──────────────────────────────────────────────────────────────────────────────
_RANGE_OPS = {"$gt", "$gte", "$lt", "$lte"}


def _describe_clause(clause: Dict) -> str:
    """Short human-readable rendering of a single filter clause."""
    try:
        return json.dumps(clause, default=str)
    except Exception:
        return str(clause)


def _clause_restrictiveness(clause: Dict) -> int:
    """Score a clause — higher means it rules out more documents.

    Exact equality is the most restrictive, then single-value $in, then ranges,
    then multi-value $in (which is closer to a widening OR).
    """
    if not isinstance(clause, dict):
        return 1

    score = 0
    for key, value in clause.items():
        if key in ("$and", "$or") and isinstance(value, list):
            score += sum(_clause_restrictiveness(v) for v in value)
            continue

        if isinstance(value, dict):
            for op, operand in value.items():
                if op == "$eq":
                    score += 3
                elif op == "$in":
                    score += 3 if isinstance(operand, list) and len(operand) <= 1 else 1
                elif op in ("$ne", "$nin"):
                    score += 2
                elif op in _RANGE_OPS:
                    score += 2
                else:
                    score += 1
        else:
            score += 3  # bare equality, e.g. {"director": "Nolan"}

    return score


def relax_filter(pre_filter: Dict) -> Tuple[Dict, str]:
    """Drop the most restrictive clause from a MongoDB pre-filter.

    Returns:
        (relaxed_filter, description_of_what_was_dropped). The relaxed filter is
        {} once nothing is left to drop, which turns the next retrieval into a
        pure vector search.
    """
    if not pre_filter or not isinstance(pre_filter, dict):
        return {}, "no filter to relax"

    # Top-level $and: drop its most restrictive member.
    if "$and" in pre_filter and isinstance(pre_filter["$and"], list):
        clauses = [c for c in pre_filter["$and"] if c]
        if len(clauses) <= 1:
            dropped = _describe_clause(clauses[0]) if clauses else "the only clause"
            return {}, f"dropped the last remaining clause {dropped} → pure vector search"

        scores = [_clause_restrictiveness(c) for c in clauses]
        victim_index = scores.index(max(scores))
        victim = clauses[victim_index]
        remaining = [c for i, c in enumerate(clauses) if i != victim_index]

        relaxed = remaining[0] if len(remaining) == 1 else {"$and": remaining}
        return relaxed, f"dropped most restrictive clause {_describe_clause(victim)}"

    # Several fields at the top level: drop the most restrictive field.
    field_keys = [k for k in pre_filter if not k.startswith("$")]
    if len(field_keys) > 1:
        scores = {k: _clause_restrictiveness({k: pre_filter[k]}) for k in field_keys}
        victim = max(scores, key=scores.get)
        relaxed = {k: v for k, v in pre_filter.items() if k != victim}
        return relaxed, f"dropped most restrictive field \"{victim}\""

    # A single condition — the only way to relax it is to remove it.
    return {}, f"dropped the only clause {_describe_clause(pre_filter)} → pure vector search"


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────
def _make_resolve_node(ctx: PipelineContext):
    def resolve_node(state: PipelineState) -> Dict:
        """Wraps the existing resolve_query() — unchanged — as a graph node."""
        query = state["query"]
        t0 = time.perf_counter()

        resolved = resolve_query(
            current_query=query,
            conversation_history=ctx.conversation_history,
            active_filters=ctx.active_filters,
            llm=ctx.llm,
        )
        elapsed = time.perf_counter() - t0

        needs_clarification, reason = (False, "")
        if ctx.enable_clarification:
            needs_clarification, reason = assess_ambiguity(
                query=query,
                resolved_query=resolved.resolved_query,
                is_followup=resolved.is_followup,
                filter_action=resolved.filter_action,
                conversation_history=ctx.conversation_history,
            )

        if needs_clarification:
            decision = "Ambiguous — needs clarification"
            status = "warn"
            detail = reason
        elif resolved.is_followup:
            decision = f"Follow-up ({resolved.filter_action})"
            status = "ok"
            detail = f'Rewritten to: "{resolved.resolved_query}"'
        else:
            decision = "Fresh query"
            status = "ok"
            detail = "No prior context carried forward."

        return {
            "resolved_query": resolved.resolved_query,
            "effective_query": resolved.resolved_query,
            "is_followup": resolved.is_followup,
            "filter_action": resolved.filter_action,
            "carried_filters": resolved.carried_filters,
            "needs_clarification": needs_clarification,
            "ambiguity_reason": reason,
            "new_query": resolved.resolved_query,
            "time_resolve": elapsed,
            "trace": [
                _trace(
                    "resolve_query",
                    "Query Resolution",
                    decision,
                    detail,
                    elapsed,
                    status,
                )
            ],
        }

    return resolve_node


def _make_slot_resolve_node(ctx: PipelineContext):
    """Slot-based resolution — the default path for structured mode.

    Emits every state key the legacy resolver did, so the routers and all
    downstream nodes are untouched. Whenever slots don't apply (document mode,
    no schema, feature disabled, or any LLM/parse failure) this delegates to
    `_make_resolve_node`, the original resolver, unchanged.
    """
    legacy_node = _make_resolve_node(ctx)

    def _fallback(state: PipelineState, reason: str) -> Dict:
        """Run the original resolver, annotating why slots were bypassed.

        The reason is folded into the existing trace entry rather than added as
        a new one, so a fallback turn's trace is shape-identical to what the
        pipeline produced before slot memory existed.
        """
        result = legacy_node(state)
        result["resolution_mode"] = "legacy"
        result["slot_state"] = dict(ctx.slot_state or {})
        result["slot_ops"] = []
        result["slot_corrections"] = []
        result["slot_display"] = slot_state_to_display(
            slot_state_from_dict(ctx.slot_state), []
        )
        result["slot_filter"] = {}

        if reason:
            entries = list(result.get("trace", []))
            if entries:
                entries[-1] = dict(entries[-1])
                detail = entries[-1].get("detail", "")
                entries[-1]["detail"] = (
                    f"{detail} (slot memory: {reason})" if detail else f"Slot memory: {reason}"
                )
                result["trace"] = entries
        return result

    def slot_resolve_node(state: PipelineState) -> Dict:
        query = state["query"]

        if not ctx.enable_slot_memory:
            return _fallback(state, "Slot memory disabled in config.")
        if state.get("mode") == MODE_DOCUMENTS:
            return _fallback(
                state,
                "Document mode has no metadata schema — follow-ups use the original resolver.",
            )
        if not ctx.metadata_field_info:
            return _fallback(state, "No metadata schema available for this dataset.")

        t0 = time.perf_counter()
        resolution = resolve_with_slots(
            current_query=query,
            slot_state=slot_state_from_dict(ctx.slot_state),
            metadata_field_info=ctx.metadata_field_info,
            llm=ctx.llm,
            conversation_history=ctx.conversation_history,
            turn_index=ctx.turn_index,
        )
        elapsed = time.perf_counter() - t0

        if resolution.used_fallback:
            return _fallback(state, f"Slot resolution unavailable: {resolution.fallback_reason}")

        needs_clarification, reason = (False, "")
        if ctx.enable_clarification:
            needs_clarification, reason = assess_ambiguity(
                query=query,
                resolved_query=resolution.effective_query,
                is_followup=resolution.is_followup,
                filter_action=resolution.filter_action,
                conversation_history=ctx.conversation_history,
            )

        changed = [f"{op.op}:{op.field_name}" for op in resolution.ops]
        if needs_clarification:
            decision, status = "Ambiguous — needs clarification", "warn"
            detail = reason
        elif resolution.corrections:
            fields = ", ".join(c["field"] for c in resolution.corrections)
            decision, status = f"Correction detected on: {fields}", "branch"
            detail = (
                f"Slot diff: {changed or 'none'}. Other slots carried forward unchanged."
            )
        elif resolution.ops:
            decision, status = f"Slot diff applied ({resolution.filter_action})", "ok"
            detail = f"Ops: {changed}. Active slots: {sorted(resolution.slot_state)}"
        else:
            decision, status = f"No slot changes ({resolution.filter_action})", "ok"
            detail = f"Active slots carried forward: {sorted(resolution.slot_state)}"

        return {
            "resolved_query": resolution.effective_query,
            "effective_query": resolution.effective_query,
            "new_query": resolution.effective_query,
            "is_followup": resolution.is_followup,
            "filter_action": resolution.filter_action,
            "carried_filters": resolution.carried_filters,
            "needs_clarification": needs_clarification,
            "ambiguity_reason": reason,
            "resolution_mode": "slots",
            "slot_state": slot_state_to_dict(resolution.slot_state),
            "slot_ops": [op.to_dict() for op in resolution.ops],
            "slot_corrections": resolution.corrections,
            "slot_display": resolution.to_display(),
            "slot_filter": resolution.carried_filters,
            "time_resolve": elapsed,
            "trace": [
                _trace("slot_resolve", "Slot Memory", decision, detail, elapsed, status)
            ],
        }

    return slot_resolve_node


def _make_clarify_node(ctx: PipelineContext):
    def clarify_node(state: PipelineState) -> Dict:
        """Ask one clarifying question instead of guessing at an ambiguous query."""
        query = state["query"]
        reason = state.get("ambiguity_reason", "")
        t0 = time.perf_counter()

        history_note = (
            "There is no earlier conversation to draw on."
            if not ctx.conversation_history
            else "Earlier turns: "
            + json.dumps(
                [h.get("query") for h in ctx.conversation_history[-3:]], default=str
            )
        )

        question = ""
        try:
            response = ctx.llm.invoke(
                _CLARIFY_PROMPT.format(query=query, history_note=history_note)
            )
            question = (response.content or "").strip().strip('"')
        except Exception as e:
            logger.warning(f"Clarification generation failed: {e}")

        if not question:
            question = (
                "Could you say a bit more about what you're looking for? "
                "I don't have enough context to resolve that query."
            )

        elapsed = time.perf_counter() - t0
        return {
            "clarification_question": question,
            "answer": "",
            "documents": [],
            "citations": [],
            "merged_filter": {},
            "time_clarify": elapsed,
            "trace": [
                _trace(
                    "ask_clarification",
                    "Clarification",
                    "Asked the user a follow-up question",
                    reason,
                    elapsed,
                    "branch",
                )
            ],
        }

    return clarify_node


def _make_constraint_node(ctx: PipelineContext):
    def constraint_node(state: PipelineState) -> Dict:
        """Decide whether filter generation is worth running at all."""
        t0 = time.perf_counter()
        query = state.get("effective_query") or state["query"]

        has_constraints, reason, signals = detect_structured_constraints(
            query, ctx.metadata_field_info
        )
        elapsed = time.perf_counter() - t0

        return {
            "has_constraints": has_constraints,
            "constraint_reason": reason,
            "constraint_signals": signals,
            "trace": [
                _trace(
                    "detect_constraints",
                    "Constraint Detection",
                    "Structured constraints found"
                    if has_constraints
                    else "No structured constraints — skipping filter generation",
                    reason,
                    elapsed,
                    "ok" if has_constraints else "branch",
                )
            ],
        }

    return constraint_node


def _make_carry_filters_node(ctx: PipelineContext):
    def carry_filters_node(state: PipelineState) -> Dict:
        """Reuse the previous turn's filters ("show me more") — as before."""
        carried = state.get("carried_filters") or {}
        return {
            "step1_filter": deepcopy({"pre_filter": carried}) if carried else {},
            "step2_filter": None,
            "merged_filter": carried,
            "new_query": state.get("effective_query") or state["query"],
            "trace": [
                _trace(
                    "carry_filters",
                    "Filter Reuse",
                    "Carried the previous turn's filter forward",
                    _describe_clause(carried),
                    0.0,
                    "branch",
                )
            ],
        }

    return carry_filters_node


def _make_filter_node(ctx: PipelineContext):
    def filter_node(state: PipelineState) -> Dict:
        """Step 1 + Step 2 of the original pipeline, verbatim, as one node.

        Calls MetadataFilter exactly as run_pipeline used to: query constructor
        -> translator -> enforce_constraints, then the time-range agent.
        """
        mf = ctx.metadata_filter
        effective_query = state.get("effective_query") or state["query"]

        out: Dict = {"step2_filter": None}
        trace_entries: List[Dict] = []

        # ── Step 1: metadata filter ──────────────────────────────────────
        t0 = time.perf_counter()
        try:
            wrapped_query = f"Answer the below question:\n\nQuestion: {effective_query}\n"
            query_constructor = mf.create_query_constructor()
            structured_query = query_constructor.invoke(wrapped_query)
            new_query, new_kwargs = mf.translator.visit_structured_query(structured_query)
            pre_filter = enforce_constraints(new_kwargs)
        except Exception as e:
            logger.error(f"Filter generation failed: {e}")
            elapsed = time.perf_counter() - t0
            return {
                "step1_filter": {},
                "step2_filter": None,
                "merged_filter": {},
                "new_query": effective_query,
                "time_step1": elapsed,
                "trace": [
                    _trace(
                        "generate_filter",
                        "Metadata Filter",
                        "Filter generation failed — falling back to pure vector search",
                        str(e),
                        elapsed,
                        "warn",
                    )
                ],
            }

        step1_elapsed = time.perf_counter() - t0
        out["step1_filter"] = deepcopy(pre_filter) if pre_filter else {}
        out["time_step1"] = step1_elapsed

        trace_entries.append(
            _trace(
                "generate_filter",
                "Metadata Filter",
                "Filter generated" if pre_filter else "NO_FILTER returned",
                _describe_clause(pre_filter.get("pre_filter", {})) if pre_filter else "",
                step1_elapsed,
                "ok" if pre_filter else "skip",
            )
        )

        # ── Step 2: time-based filter (only when step 1 produced a filter) ──
        step2_filter = None
        step2_elapsed = 0.0
        if pre_filter:
            t1 = time.perf_counter()
            try:
                time_based_pre_filter, new_query = mf.generate_time_based_filter(
                    pre_filter, new_query
                )
                if time_based_pre_filter:
                    step2_filter = deepcopy(time_based_pre_filter)
                    pre_filter["pre_filter"] = {
                        "$and": [
                            pre_filter["pre_filter"],
                            time_based_pre_filter["pre_filter"],
                        ]
                    }
            except Exception:
                logger.warning("Time-based filter step failed or returned NO_FILTER")
            step2_elapsed = time.perf_counter() - t1

            trace_entries.append(
                _trace(
                    "time_range_filter",
                    "Time-Range Filter",
                    "Temporal filter merged" if step2_filter else "Not triggered",
                    _describe_clause(step2_filter.get("pre_filter", {}))
                    if step2_filter
                    else "No temporal keywords resolved to a date range.",
                    step2_elapsed,
                    "ok" if step2_filter else "skip",
                )
            )
        else:
            trace_entries.append(
                _trace(
                    "time_range_filter",
                    "Time-Range Filter",
                    "Skipped — no base filter to extend",
                    "",
                    0.0,
                    "skip",
                )
            )

        out["step2_filter"] = step2_filter
        out["time_step2"] = step2_elapsed
        out["merged_filter"] = pre_filter.get("pre_filter", {}) if pre_filter else {}
        out["new_query"] = new_query if new_query else effective_query
        out["trace"] = trace_entries
        return out

    return filter_node


def _make_retrieve_node(ctx: PipelineContext):
    def retrieve_node(state: PipelineState) -> Dict:
        """Vector search — identical call shape to the original pipeline."""
        pre_filter = state.get("merged_filter") or {}
        search_query = state.get("new_query") or state.get("effective_query") or state["query"]

        t0 = time.perf_counter()
        try:
            vectorstore = MongoDBAtlasVectorSearch(ctx.collection, ctx.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"pre_filter": pre_filter})
            documents = retriever.invoke(search_query)
            error = None
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            documents, error = [], str(e)
        elapsed = time.perf_counter() - t0

        attempt = int(state.get("retry_count", 0)) + 1
        if error:
            decision, status, detail = "Retrieval failed", "warn", error
        else:
            decision = f"Retrieved {len(documents)} document(s)"
            status = "ok" if documents else "warn"
            detail = (
                f"Pre-filter: {_describe_clause(pre_filter)}"
                if pre_filter
                else "Pure vector search (no pre-filter)"
            )
            if attempt > 1:
                detail = f"Attempt {attempt}. " + detail

        return {
            "documents": documents,
            "time_retrieval": elapsed,
            "trace": [
                _trace("vector_search", "Vector Search", decision, detail, elapsed, status)
            ],
        }

    return retrieve_node


def _make_relax_node(ctx: PipelineContext):
    def relax_node(state: PipelineState) -> Dict:
        """Branch 1: zero results with a filter applied — widen and retry."""
        current = state.get("merged_filter") or {}
        relaxed, description = relax_filter(current)
        retry_count = int(state.get("retry_count", 0)) + 1

        return {
            "merged_filter": relaxed,
            "retry_count": retry_count,
            "relaxations": [
                {
                    "attempt": retry_count,
                    "before": deepcopy(current),
                    "after": deepcopy(relaxed),
                    "description": description,
                }
            ],
            "trace": [
                _trace(
                    "relax_filter",
                    f"Filter Relaxation (retry {retry_count})",
                    "Zero results — relaxed the filter and retrying",
                    description,
                    0.0,
                    "branch",
                )
            ],
        }

    return relax_node


def _make_synthesize_node(ctx: PipelineContext):
    def synthesize_node(state: PipelineState) -> Dict:
        """Feature 1: cited answer generation + faithfulness audit."""
        documents = state.get("documents") or []
        question = state.get("effective_query") or state["query"]

        if not ctx.enable_synthesis:
            return {
                "answer": "",
                "citations": [],
                "faithfulness": FaithfulnessReport(
                    verdict="unknown",
                    reason="Answer synthesis disabled in config.",
                    method="skipped",
                ).to_dict(),
                "trace": [
                    _trace(
                        "synthesize_answer",
                        "Answer Synthesis",
                        "Disabled in config",
                        "",
                        0.0,
                        "skip",
                    )
                ],
            }

        result = generate_grounded_answer(
            query=question,
            documents=documents,
            llm=ctx.llm,
            faithfulness_llm=ctx.faithfulness_llm or ctx.llm,
            run_faithfulness=ctx.enable_faithfulness,
        )

        trace_entries = [
            _trace(
                "synthesize_answer",
                "Answer Synthesis",
                "Answer generated with citations"
                if result.answer and not result.error
                else ("Synthesis failed" if result.error else "Nothing to answer from"),
                f"{len(result.citations)} source(s) cited"
                if result.citations
                else (result.error or "No documents retrieved."),
                result.time_synthesis,
                "ok" if result.answer and not result.error else "warn",
            )
        ]

        if ctx.enable_faithfulness:
            report = result.faithfulness
            trace_entries.append(
                _trace(
                    "faithfulness_check",
                    "Faithfulness Check",
                    f"{report.label} ({report.score:.0%})",
                    report.reason or f"Method: {report.method}",
                    result.time_faithfulness,
                    {"grounded": "ok", "partial": "warn", "ungrounded": "warn"}.get(
                        report.verdict, "skip"
                    ),
                )
            )

        return {
            "answer": result.answer,
            "citations": result.citations,
            "faithfulness": result.faithfulness.to_dict(),
            "time_synthesis": result.time_synthesis,
            "time_faithfulness": result.time_faithfulness,
            "trace": trace_entries,
        }

    return synthesize_node


# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────
def _make_route_after_resolve(ctx: PipelineContext):
    def route_after_resolve(state: PipelineState) -> str:
        # Branch 3 — ambiguous follow-up.
        if state.get("needs_clarification"):
            return "clarify"

        # Document mode skips metadata filtering entirely (feature 2).
        if state.get("mode") == MODE_DOCUMENTS:
            return "retrieve"

        # "show me more" — reuse the previous filter, as the original did.
        if state.get("filter_action") == "keep_all" and state.get("carried_filters"):
            return "carry_filters"

        return "detect_constraints"

    return route_after_resolve


def _make_route_after_constraints(ctx: PipelineContext):
    def route_after_constraints(state: PipelineState) -> str:
        # Branch 2 — nothing to filter on, go straight to vector search.
        return "generate_filter" if state.get("has_constraints") else "retrieve"

    return route_after_constraints


def _make_route_after_retrieve(ctx: PipelineContext):
    def route_after_retrieve(state: PipelineState) -> str:
        # Branch 1 — empty results with a filter still applied, retries left.
        documents = state.get("documents") or []
        if documents:
            return "synthesize"
        if not state.get("merged_filter"):
            return "synthesize"          # nothing left to relax
        if int(state.get("retry_count", 0)) >= ctx.max_relaxations:
            return "synthesize"
        return "relax"

    return route_after_retrieve


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────
def build_graph(ctx: PipelineContext):
    """Compile the agentic pipeline graph for a given context."""
    graph = StateGraph(PipelineState)

    # Slot-based resolution is the default; it delegates to the original
    # _make_resolve_node internally whenever slots do not apply.
    graph.add_node("resolve", _make_slot_resolve_node(ctx))
    graph.add_node("clarify", _make_clarify_node(ctx))
    graph.add_node("detect_constraints", _make_constraint_node(ctx))
    graph.add_node("carry_filters", _make_carry_filters_node(ctx))
    graph.add_node("generate_filter", _make_filter_node(ctx))
    graph.add_node("retrieve", _make_retrieve_node(ctx))
    graph.add_node("relax", _make_relax_node(ctx))
    graph.add_node("synthesize", _make_synthesize_node(ctx))

    graph.set_entry_point("resolve")

    graph.add_conditional_edges(
        "resolve",
        _make_route_after_resolve(ctx),
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
            "carry_filters": "carry_filters",
            "detect_constraints": "detect_constraints",
        },
    )
    graph.add_edge("clarify", END)

    graph.add_conditional_edges(
        "detect_constraints",
        _make_route_after_constraints(ctx),
        {"generate_filter": "generate_filter", "retrieve": "retrieve"},
    )

    graph.add_edge("carry_filters", "retrieve")
    graph.add_edge("generate_filter", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        _make_route_after_retrieve(ctx),
        {"relax": "relax", "synthesize": "synthesize"},
    )
    graph.add_edge("relax", "retrieve")
    graph.add_edge("synthesize", END)

    return graph.compile()


def initial_state(query: str, mode: str = MODE_STRUCTURED) -> Dict:
    """Fresh state for one turn, with accumulator fields zeroed."""
    return {
        "query": query,
        "mode": mode,
        "effective_query": query,
        "resolved_query": query,
        "new_query": query,
        "documents": [],
        "retry_count": 0,
        "relaxations": [],
        "step1_filter": {},
        "step2_filter": None,
        "merged_filter": {},
        "citations": [],
        "answer": "",
        "faithfulness": {},
        "needs_clarification": False,
        "clarification_question": "",
        "resolution_mode": "legacy",
        "slot_state": {},
        "slot_ops": [],
        "slot_corrections": [],
        "slot_display": [],
        "slot_filter": {},
        "time_resolve": 0.0,
        "time_step1": 0.0,
        "time_step2": 0.0,
        "time_retrieval": 0.0,
        "time_synthesis": 0.0,
        "time_faithfulness": 0.0,
        "time_clarify": 0.0,
        "trace": [],
    }


def run_graph(query: str, ctx: PipelineContext, mode: Optional[str] = None) -> Dict:
    """Run one turn through the graph and return the final state."""
    mode = mode or ctx.mode or MODE_STRUCTURED
    app = build_graph(ctx)
    final_state = app.invoke(
        initial_state(query, mode),
        config={"recursion_limit": 25 + 4 * max(ctx.max_relaxations, 0)},
    )
    logger.info(
        "Graph finished — nodes: "
        + " → ".join(entry["node"] for entry in final_state.get("trace", []))
    )
    return final_state


# ──────────────────────────────────────────────────────────────────────────────
# LLM construction helpers (Groq via the OpenAI-compatible endpoint)
# ──────────────────────────────────────────────────────────────────────────────
def make_llm(model: Optional[str] = None, temperature: Optional[float] = None,
             max_tokens: int = 1024, api_key: Optional[str] = None,
             api_base: Optional[str] = None):
    """Build a ChatOpenAI pointed at the configured (Groq) endpoint.

    Same provider and credentials the rest of the project already uses by
    default; no new dependency is introduced. api_key/api_base let a caller
    point this at a different OpenAI-compatible endpoint (e.g. Gemini's, for
    the faithfulness auditor) without touching the main pipeline's Groq setup.
    """
    from langchain_openai import ChatOpenAI

    default_headers = os.getenv("OPEN_API_DEFAULT_HEADERS")
    default_headers = json.loads(default_headers) if default_headers else None

    kwargs = {
        "model": model or config["model"],
        "max_tokens": max_tokens,
        "openai_api_key": api_key or os.getenv("OPEN_AI_API_KEY"),
        "openai_api_base": api_base or os.getenv("OPEN_API_BASE"),
        "default_headers": default_headers,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def make_faithfulness_llm():
    """Auditor model for the faithfulness check.

    Defaults to the same Groq model as everything else, at temperature 0 so the
    verdict is stable. FAITHFULNESS_MODEL in .env can point this at a stronger
    (or different-provider) model without any code change — self-grading with
    the same model and settings is the weakest link in the check.

    FAITHFULNESS_API_KEY / FAITHFULNESS_API_BASE let that model live on a
    different OpenAI-compatible endpoint than the main Groq pipeline — e.g.
    Google's Gemini endpoint (https://generativelanguage.googleapis.com/v1beta/openai/).
    When they're unset, the auditor reuses the main Groq credentials, exactly
    as before.
    """
    return make_llm(
        model=os.getenv("FAITHFULNESS_MODEL") or config.get("faithfulness_model"),
        temperature=0,
        max_tokens=768,
        api_key=os.getenv("FAITHFULNESS_API_KEY"),
        api_base=os.getenv("FAITHFULNESS_API_BASE"),
    )
