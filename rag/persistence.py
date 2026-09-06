"""
persistence.py — Conversation State -> MongoDB
===============================================
Persists every turn of a conversation to a MongoDB collection so a browser
session can be rehydrated after a reload, and so real usage accumulates into a
corpus that the eval harness (eval/run_eval.py) can consume later — the turn
document deliberately mirrors the eval report's per-case shape.

Design rule: persistence is strictly best-effort. Every read and write is
wrapped so that an unreachable cluster degrades the app to "no history" rather
than breaking it. Nothing in here raises to the caller.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rag.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def default_conversation_collection() -> str:
    """Collection holding conversation turns.

    CONVERSATION_COLLECTION_NAME in .env overrides the config.yaml value.
    """
    import os

    return (
        os.getenv("CONVERSATION_COLLECTION_NAME")
        or config.get("conversation_collection_name")
        or "conversation_logs"
    )


def new_session_id() -> str:
    """A stable id for one browser session."""
    return uuid.uuid4().hex


@dataclass
class RehydratedSession:
    """What a reload can recover from MongoDB."""

    session_id: str = ""
    found: bool = False
    turns: List[Dict] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    active_filters: Dict = field(default_factory=dict)
    slot_state: Dict = field(default_factory=dict)
    turn_index: int = 0
    mode: Optional[str] = None
    error: str = ""


def _collection(collection_name: Optional[str] = None):
    """Get the conversation collection, or None if Mongo is unreachable.

    Imported lazily so that a missing MONGO_URI can never break module import.
    """
    try:
        from rag.utils.mongodb_helper import get_mongo_collection

        return get_mongo_collection(
            db_name=config["database_name"],
            collection_name=collection_name or default_conversation_collection(),
        )
    except Exception as e:
        logger.warning(f"Conversation persistence unavailable (collection): {e}")
        return None


def _doc_ids(documents: List[Any]) -> List[str]:
    """Stable-ish identifiers for retrieved documents.

    Atlas vector search returns LangChain Documents whose _id is not always
    carried through, so fall back to a content fingerprint — that is what the
    eval harness matches on too.
    """
    ids = []
    for d in documents or []:
        meta = getattr(d, "metadata", {}) or {}
        doc_id = meta.get("_id") or meta.get("id")
        if doc_id is not None:
            ids.append(str(doc_id))
            continue
        content = (getattr(d, "page_content", "") or "").strip()
        ids.append(content[:80])
    return ids


def build_turn_document(
    session_id: str,
    turn_index: int,
    query: str,
    result: Dict,
    mode: str = "structured",
) -> Dict:
    """Shape one turn for storage.

    Mirrors the per-case fields the eval report emits, so logged production
    turns can later be folded into an eval set.
    """
    return {
        "session_id": session_id,
        "turn_index": int(turn_index),
        "created_at": datetime.now(timezone.utc),
        "mode": mode,
        "query": query,
        # Resolution / memory
        "resolution_mode": result.get("resolution_mode", "legacy"),
        "resolved_query": result.get("resolved_query", ""),
        "is_followup": bool(result.get("is_followup", False)),
        "filter_action": result.get("filter_action", "fresh"),
        "slot_state": result.get("slot_state", {}) or {},
        "slot_ops": result.get("slot_ops", []) or [],
        "slot_corrections": result.get("slot_corrections", []) or [],
        # Filtering
        "merged_filter": result.get("merged_filter", {}) or {},
        "step1_filter": result.get("step1_filter", {}) or {},
        "step2_filter": result.get("step2_filter"),
        "relaxations": result.get("relaxations", []) or [],
        # Retrieval
        "retrieved_doc_ids": _doc_ids(result.get("documents", [])),
        "retrieved_count": len(result.get("documents", []) or []),
        # Generation
        "answer": result.get("answer", "") or "",
        "citations": result.get("citations", []) or [],
        "faithfulness": result.get("faithfulness", {}) or {},
        "clarification_question": result.get("clarification_question", "") or "",
        # Instrumentation
        "timings": {
            "resolve": result.get("time_resolve", 0.0),
            "step1": result.get("time_step1", 0.0),
            "step2": result.get("time_step2", 0.0),
            "retrieval": result.get("time_retrieval", 0.0),
            "synthesis": result.get("time_synthesis", 0.0),
            "faithfulness": result.get("time_faithfulness", 0.0),
        },
        "trace_nodes": [t.get("node") for t in (result.get("trace") or [])],
    }


def log_turn(
    session_id: str,
    turn_index: int,
    query: str,
    result: Dict,
    mode: str = "structured",
    collection_name: Optional[str] = None,
) -> bool:
    """Persist one turn. Returns True on success, False if it degraded.

    Never raises: a failed write must not break the UI.
    """
    if not session_id:
        return False

    col = _collection(collection_name)
    if col is None:
        return False

    try:
        doc = build_turn_document(session_id, turn_index, query, result, mode)
        # Upsert on (session, turn) so a Streamlit rerun cannot duplicate a turn.
        col.update_one(
            {"session_id": session_id, "turn_index": int(turn_index)},
            {"$set": doc},
            upsert=True,
        )
        logger.info(f"Persisted turn {turn_index} of session {session_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not persist turn {turn_index} ({e}) — continuing without persistence")
        return False


def load_session(
    session_id: str,
    collection_name: Optional[str] = None,
) -> List[Dict]:
    """Every stored turn for a session, oldest first. [] if unavailable."""
    if not session_id:
        return []

    col = _collection(collection_name)
    if col is None:
        return []

    try:
        return list(
            col.find({"session_id": session_id}, {"_id": 0}).sort("turn_index", 1)
        )
    except Exception as e:
        logger.warning(f"Could not load session {session_id}: {e}")
        return []


def rehydrate_session(
    session_id: str,
    collection_name: Optional[str] = None,
) -> RehydratedSession:
    """Rebuild conversation history, active filters and slot state from Mongo.

    Returns a RehydratedSession with found=False when there is nothing stored
    or the database is unreachable — the caller then simply starts empty.
    """
    out = RehydratedSession(session_id=session_id)

    try:
        turns = load_session(session_id, collection_name)
    except Exception as e:                                   # defensive
        out.error = str(e)
        return out

    if not turns:
        return out

    out.found = True
    out.turns = turns

    for turn in turns:
        # A clarification turn was never actually resolved, so it must not
        # become follow-up context — same rule the live app applies.
        if turn.get("clarification_question"):
            continue
        out.conversation_history.append(
            {
                "query": turn.get("query", ""),
                "filter_used": turn.get("merged_filter", {}) or {},
            }
        )
        out.active_filters = turn.get("merged_filter", {}) or {}
        if turn.get("resolution_mode") == "slots":
            out.slot_state = turn.get("slot_state", {}) or {}
        out.mode = turn.get("mode", out.mode)

    try:
        out.turn_index = max(int(t.get("turn_index", 0)) for t in turns) + 1
    except Exception:
        out.turn_index = len(turns)

    logger.info(
        f"Rehydrated session {session_id}: {len(turns)} turn(s), "
        f"slots={sorted(out.slot_state)}"
    )
    return out


def history_entries_from_turns(turns: List[Dict]) -> List[Dict]:
    """Rebuild the UI's `st.session_state.history` entries from stored turns.

    Retrieved Documents themselves are not stored (only their ids), so restored
    entries carry no result cards — the answer, citations, filters, slots and
    trace are all preserved.
    """
    entries = []
    for turn in turns or []:
        entries.append(
            {
                "query": turn.get("query", ""),
                "documents": [],          # not persisted; ids only
                "restored": True,
                "restored_doc_count": turn.get("retrieved_count", 0),
                "filters": {
                    "step1": turn.get("step1_filter", {}) or {},
                    "step2": turn.get("step2_filter"),
                    "merged": turn.get("merged_filter", {}) or {},
                },
                "timings": turn.get("timings", {}) or {},
                "is_followup": turn.get("is_followup", False),
                "filter_action": turn.get("filter_action", "fresh"),
                "carried_filters": {},
                "resolved_query": turn.get("resolved_query", "") or turn.get("query", ""),
                "answer": turn.get("answer", "") or "",
                "citations": turn.get("citations", []) or [],
                "faithfulness": turn.get("faithfulness", {}) or {},
                "trace": [],              # node names only; timeline not restored
                "trace_nodes": turn.get("trace_nodes", []) or [],
                "relaxations": turn.get("relaxations", []) or [],
                "clarification_question": turn.get("clarification_question", "") or "",
                "mode": turn.get("mode", "structured"),
                "resolution_mode": turn.get("resolution_mode", "legacy"),
                "slot_state": turn.get("slot_state", {}) or {},
                "slot_ops": turn.get("slot_ops", []) or [],
                "slot_corrections": turn.get("slot_corrections", []) or [],
                "slot_display": _display_from_slot_state(turn.get("slot_state", {}) or {}),
                "slot_filter": _compile_quietly(turn.get("slot_state", {}) or {}),
            }
        )
    return entries


def _display_from_slot_state(slot_state: Dict) -> List[Dict]:
    """Slot rows for a restored turn (everything reads as carried)."""
    try:
        from rag.slot_memory import slot_state_from_dict, slot_state_to_display

        return slot_state_to_display(slot_state_from_dict(slot_state), [])
    except Exception as e:
        logger.warning(f"Could not rebuild slot display: {e}")
        return []


def _compile_quietly(slot_state: Dict) -> Dict:
    try:
        from rag.slot_memory import compile_slots_to_filter, slot_state_from_dict

        return compile_slots_to_filter(slot_state_from_dict(slot_state))
    except Exception:
        return {}


def delete_session(session_id: str, collection_name: Optional[str] = None) -> bool:
    """Remove every stored turn for a session. Best-effort."""
    if not session_id:
        return False
    col = _collection(collection_name)
    if col is None:
        return False
    try:
        col.delete_many({"session_id": session_id})
        return True
    except Exception as e:
        logger.warning(f"Could not delete session {session_id}: {e}")
        return False


def ensure_indexes(collection_name: Optional[str] = None) -> bool:
    """Create the (session_id, turn_index) index used by every query."""
    col = _collection(collection_name)
    if col is None:
        return False
    try:
        col.create_index([("session_id", 1), ("turn_index", 1)], unique=True)
        return True
    except Exception as e:
        logger.warning(f"Could not create conversation index: {e}")
        return False
