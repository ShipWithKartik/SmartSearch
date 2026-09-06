"""
slot_memory.py — Slot-Based Multi-Turn Memory
==============================================
Replaces all-or-nothing filter carry-forward with explicit per-field slots.

The original resolver (rag/query_resolver.py, still present and still used as a
fallback) could only do three things with the previous turn's filter: keep all
of it, drop all of it, or ask the LLM to rewrite the whole query into a new
self-contained sentence. That loses information on corrections — "no, I meant
before 1990, not after" would re-derive every constraint from scratch and could
silently drop unrelated ones.

Here, each known metadata field of the active schema gets its own slot. On each
turn the LLM emits a *diff* against current slot state:

    {"ops": [{"op": "update", "field": "release_date", "operator": "lt",
              "value": "1990-01-01", "is_correction": true}], ...}

Only the named slots change; everything else carries forward untouched. The
resulting slot state is then rendered back into a self-contained natural
language query, which feeds the existing filter pipeline unchanged.

Document mode has no metadata schema, so callers should skip slots there and
keep using the original resolver — see rag/graph.py.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Operators a slot may carry. Mirrors the comparators the MongoDB translator
# already supports, so compiled slots are directly usable as a pre-filter.
VALID_OPERATORS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "nin")

_OPERATOR_TO_MONGO = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}

_OPERATOR_PHRASE = {
    "eq": "is",
    "ne": "is not",
    "gt": "greater than",
    "gte": "at least",
    "lt": "before/less than",
    "lte": "at most",
    "in": "one of",
    "nin": "not one of",
}

VALID_OPS = ("set", "update", "clear")


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Slot:
    """One field-level constraint currently held in memory."""

    field_name: str
    operator: str = "eq"
    value: Any = None
    phrase: str = ""          # human-readable rendering, e.g. "genre is anime"
    turn_set: int = 0         # which turn last wrote this slot
    corrected: bool = False   # last write was an explicit user correction

    def to_dict(self) -> Dict:
        return {
            "field": self.field_name,
            "operator": self.operator,
            "value": self.value,
            "phrase": self.phrase,
            "turn_set": self.turn_set,
            "corrected": self.corrected,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Slot":
        return Slot(
            field_name=d.get("field", ""),
            operator=d.get("operator", "eq"),
            value=d.get("value"),
            phrase=d.get("phrase", ""),
            turn_set=int(d.get("turn_set", 0) or 0),
            corrected=bool(d.get("corrected", False)),
        )


@dataclass
class SlotOp:
    """A single requested change to slot state."""

    op: str                       # "set" | "update" | "clear"
    field_name: str
    operator: str = "eq"
    value: Any = None
    is_correction: bool = False
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "op": self.op,
            "field": self.field_name,
            "operator": self.operator,
            "value": self.value,
            "is_correction": self.is_correction,
            "reason": self.reason,
        }


@dataclass
class SlotResolution:
    """Result of one slot-based resolution turn."""

    ops: List[SlotOp] = field(default_factory=list)
    slot_state: Dict[str, Slot] = field(default_factory=dict)
    previous_state: Dict[str, Slot] = field(default_factory=dict)
    semantic_query: str = ""       # the non-slot, meaning-bearing remainder
    effective_query: str = ""      # self-contained query for the filter pipeline
    is_followup: bool = False
    filter_action: str = "fresh"   # backward-compatible with ResolvedQuery
    carried_filters: Dict = field(default_factory=dict)
    corrections: List[Dict] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str = ""

    def changed_fields(self) -> List[str]:
        return [op.field_name for op in self.ops]

    def to_display(self) -> List[Dict]:
        """Per-slot rows for the transparency panel."""
        return slot_state_to_display(self.slot_state, self.ops, self.previous_state)


# ──────────────────────────────────────────────────────────────────────────────
# Slot state helpers
# ──────────────────────────────────────────────────────────────────────────────
def slot_state_to_dict(slot_state: Dict[str, Slot]) -> Dict[str, Dict]:
    """Serialise slot state (for session_state / MongoDB persistence)."""
    return {name: slot.to_dict() for name, slot in (slot_state or {}).items()}


def slot_state_from_dict(data: Optional[Dict]) -> Dict[str, Slot]:
    """Rehydrate slot state from its serialised form."""
    out: Dict[str, Slot] = {}
    for name, raw in (data or {}).items():
        try:
            out[name] = Slot.from_dict(raw) if isinstance(raw, dict) else raw
        except Exception as e:
            logger.warning(f"Skipping unreadable slot '{name}': {e}")
    return out


def field_names(metadata_field_info: List[AttributeInfo]) -> List[str]:
    """Names of every known metadata field for the active schema."""
    names = []
    for f in metadata_field_info or []:
        name = f.name if isinstance(f, AttributeInfo) else f.get("name", "")
        if name:
            names.append(str(name))
    return names


def _field_type(f) -> str:
    return (f.type if isinstance(f, AttributeInfo) else f.get("type", "string")) or "string"


def _field_description(f) -> str:
    return (
        f.description if isinstance(f, AttributeInfo) else f.get("description", "")
    ) or ""


def build_phrase(field_name: str, operator: str, value: Any) -> str:
    """Human-readable rendering of one slot, used in the UI and rewritten query."""
    if isinstance(value, list):
        rendered = " or ".join(str(v) for v in value)
    else:
        rendered = str(value)
    return f"{str(field_name).replace('_', ' ')} {_OPERATOR_PHRASE.get(operator, operator)} {rendered}"


def apply_ops(
    slot_state: Dict[str, Slot],
    ops: List[SlotOp],
    turn_index: int = 0,
) -> Tuple[Dict[str, Slot], List[Dict]]:
    """Apply a diff to slot state.

    Only the fields named in `ops` change — every other slot carries forward
    exactly as it was. That is the whole point of slot memory.

    Returns:
        (new_slot_state, corrections) where corrections describes any slot whose
        value was explicitly corrected by the user this turn.
    """
    new_state: Dict[str, Slot] = {name: slot for name, slot in (slot_state or {}).items()}
    corrections: List[Dict] = []

    for op in ops or []:
        name = op.field_name
        if not name:
            continue

        if op.op == "clear":
            if name in new_state:
                removed = new_state.pop(name)
                if op.is_correction:
                    corrections.append(
                        {
                            "field": name,
                            "from": removed.to_dict(),
                            "to": None,
                            "reason": op.reason,
                        }
                    )
            continue

        if op.op not in ("set", "update"):
            logger.warning(f"Ignoring unknown slot op '{op.op}' for field '{name}'")
            continue

        previous = new_state.get(name)
        operator = op.operator if op.operator in VALID_OPERATORS else "eq"
        slot = Slot(
            field_name=name,
            operator=operator,
            value=op.value,
            phrase=build_phrase(name, operator, op.value),
            turn_set=turn_index,
            corrected=bool(op.is_correction),
        )
        new_state[name] = slot

        if op.is_correction:
            corrections.append(
                {
                    "field": name,
                    "from": previous.to_dict() if previous else None,
                    "to": slot.to_dict(),
                    "reason": op.reason,
                }
            )

    return new_state, corrections


def compile_slots_to_filter(slot_state: Dict[str, Slot]) -> Dict:
    """Compile slot state into a MongoDB pre-filter.

    Shape matches what the existing pipeline produces, so the result is a
    drop-in for `carried_filters` / `merged_filter`:
        one slot   -> {"genre": {"$in": ["anime"]}}
        many slots -> {"$and": [ ... ]}
    """
    clauses: List[Dict] = []

    for name, slot in sorted((slot_state or {}).items()):
        if slot is None or slot.value is None:
            continue
        mongo_op = _OPERATOR_TO_MONGO.get(slot.operator)
        if not mongo_op:
            continue

        value = slot.value
        # $in/$nin need a list; a scalar list with eq is best expressed as $in.
        if mongo_op in ("$in", "$nin") and not isinstance(value, list):
            value = [value]
        if mongo_op == "$eq" and isinstance(value, list):
            mongo_op, value = "$in", value

        clauses.append({name: {mongo_op: value}})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def render_slots_as_query(semantic_query: str, slot_state: Dict[str, Slot]) -> str:
    """Render slot state back into a self-contained natural-language query.

    This is what feeds the existing filter pipeline: the query constructor sees
    one complete sentence exactly as it always has, but its content is now
    derived deterministically from slot state instead of a free-form LLM
    rewrite, so unrelated constraints can't silently vanish on a correction.
    """
    core = (semantic_query or "").strip()
    phrases = [
        slot.phrase or build_phrase(name, slot.operator, slot.value)
        for name, slot in sorted((slot_state or {}).items())
        if slot is not None and slot.value is not None
    ]

    if not phrases:
        return core
    constraints = ", ".join(phrases)
    if not core:
        return f"Find records where {constraints}"
    return f"{core} where {constraints}"


def slot_state_to_display(
    slot_state: Dict[str, Slot],
    ops: Optional[List[SlotOp]] = None,
    previous_state: Optional[Dict[str, Slot]] = None,
) -> List[Dict]:
    """Rows describing each slot and what happened to it this turn (for the UI)."""
    op_by_field = {op.field_name: op for op in (ops or [])}
    rows: List[Dict] = []

    for name, slot in sorted((slot_state or {}).items()):
        op = op_by_field.get(name)
        if op is None:
            status = "carried"
        elif op.op == "set":
            status = "set"
        elif op.op == "update":
            status = "updated"
        else:
            status = "set"
        rows.append(
            {
                "field": name,
                "operator": slot.operator,
                "value": slot.value,
                "phrase": slot.phrase,
                "status": status,
                "corrected": bool(op.is_correction) if op else False,
                "reason": op.reason if op else "",
            }
        )

    # Slots removed this turn no longer appear in slot_state — surface them too.
    for op in ops or []:
        if op.op == "clear":
            prev = (previous_state or {}).get(op.field_name)
            rows.append(
                {
                    "field": op.field_name,
                    "operator": prev.operator if prev else "",
                    "value": prev.value if prev else None,
                    "phrase": prev.phrase if prev else "",
                    "status": "cleared",
                    "corrected": bool(op.is_correction),
                    "reason": op.reason,
                }
            )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# LLM prompt — emit a diff, not a rewrite
# ──────────────────────────────────────────────────────────────────────────────
_SLOT_PROMPT = """\
You maintain the search state of a conversational search system.

The state is a set of SLOTS — one per metadata field. Your job is to read the \
user's new message and output ONLY the CHANGES to those slots. Slots you do not \
mention stay exactly as they are.

## Available fields (the only valid slot names)
{schema}

## Current slot state
{current_state}

## Recent conversation
{history}

## User's new message
"{query}"

## Your task

Output a JSON diff with these keys:

- **ops**: list of slot changes. Each op is:
    {{"op": "set" | "update" | "clear",
      "field": "<one of the field names above>",
      "operator": "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin",
      "value": <string, number, or list of strings>,
      "is_correction": <true if the user is CORRECTING an earlier constraint>,
      "reason": "<a few words on why>"}}

  - "set"    — this field had no value before, or the query is brand new
  - "update" — this field already had a value and the user is changing it
  - "clear"  — the user wants this constraint removed entirely
                 (omit "operator"/"value" for clear)

- **semantic_query**: the part of the user's intent that is NOT a metadata \
constraint — the topic/meaning to search for semantically. Empty string if the \
message is purely constraints.

- **is_followup**: true if this message builds on the conversation above, false \
if it starts a fresh topic.

- **reset_all**: true ONLY if the user has clearly changed subject and every \
existing slot should be discarded. When true, you do not need clear-ops for \
each field.

## CRITICAL — corrections

Watch for the user correcting themselves. These are UPDATES TO ONE SLOT, and \
must NEVER discard the other slots:
  - "no, I meant before 1990, not after"  -> update release_date, is_correction: true
  - "actually make it a comedy"           -> update genre, is_correction: true
  - "sorry, I said 8, I meant above 9"    -> update rating, is_correction: true
  - "not Nolan, the other one"            -> update/clear director, is_correction: true

A correction changes ONLY the field being corrected. Everything else carries forward.

## Rules
- Return ONLY valid JSON. No markdown fences, no commentary.
- Use ONLY field names from the list above. Never invent fields.
- For list-valued fields (type "[string]") prefer operator "in" with a list value.
- For dates use "YYYY-MM-DD" and range operators (lt/gte/...), never "eq" on a partial date.
- "show me more" / "any others" -> ops: [], is_followup: true (state is unchanged).
- If the message is a brand-new topic, set reset_all: true and emit set-ops for
  the new constraints only.
- Emit NO ops for constraints that are already correct in the current state.

## Examples

Current state: {{}}
Message: "show me anime movies"
{{"ops": [{{"op": "set", "field": "genre", "operator": "in", "value": ["anime"], "is_correction": false, "reason": "new genre constraint"}}], "semantic_query": "movies", "is_followup": false, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["anime"]}}}}
Message: "what about before 1990"
{{"ops": [{{"op": "set", "field": "release_date", "operator": "lt", "value": "1990-01-01", "is_correction": false, "reason": "added date bound"}}], "semantic_query": "", "is_followup": true, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["anime"]}}, "release_date": {{"operator": "gt", "value": "1990-01-01"}}}}
Message: "no, I meant before 1990, not after"
{{"ops": [{{"op": "update", "field": "release_date", "operator": "lt", "value": "1990-01-01", "is_correction": true, "reason": "user corrected the direction of the date bound"}}], "semantic_query": "", "is_followup": true, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["thriller"]}}, "director": {{"operator": "eq", "value": "Christopher Nolan"}}}}
Message: "actually make it a comedy"
{{"ops": [{{"op": "update", "field": "genre", "operator": "in", "value": ["comedy"], "is_correction": true, "reason": "user corrected the genre; director stays"}}], "semantic_query": "", "is_followup": true, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["anime"]}}, "rating": {{"operator": "gt", "value": 8}}}}
Message: "forget the rating"
{{"ops": [{{"op": "clear", "field": "rating", "is_correction": false, "reason": "user dropped the rating constraint"}}], "semantic_query": "", "is_followup": true, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["anime"]}}}}
Message: "show me more"
{{"ops": [], "semantic_query": "", "is_followup": true, "reset_all": false}}

Current state: {{"genre": {{"operator": "in", "value": ["anime"]}}}}
Message: "what laptops do you have under $500"
{{"ops": [], "semantic_query": "laptops", "is_followup": false, "reset_all": true}}

Return the JSON now:
"""


def _format_schema(metadata_field_info: List[AttributeInfo]) -> str:
    lines = []
    for f in metadata_field_info or []:
        name = f.name if isinstance(f, AttributeInfo) else f.get("name", "")
        lines.append(f'- "{name}" (type: {_field_type(f)}) — {_field_description(f)}')
    return "\n".join(lines) if lines else "(no fields)"


def _format_state(slot_state: Dict[str, Slot]) -> str:
    if not slot_state:
        return "{}"
    compact = {
        name: {"operator": s.operator, "value": s.value}
        for name, s in sorted(slot_state.items())
        if s is not None
    }
    return json.dumps(compact, default=str)


def _format_history(conversation_history: List[Dict], limit: int = 3) -> str:
    if not conversation_history:
        return "(no previous turns)"
    recent = conversation_history[-limit:]
    return json.dumps([h.get("query", "") for h in recent], default=str)


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Some models prepend prose; salvage the outermost JSON object.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return text


def parse_slot_response(raw_text: str, allowed_fields: List[str]) -> Tuple[List[SlotOp], str, bool, bool]:
    """Parse the LLM's diff. Raises ValueError if it isn't usable.

    Returns:
        (ops, semantic_query, is_followup, reset_all)
    """
    parsed = json.loads(_strip_code_fences(raw_text))
    if not isinstance(parsed, dict):
        raise ValueError("Slot response was not a JSON object")

    allowed = set(allowed_fields or [])
    ops: List[SlotOp] = []

    for raw_op in parsed.get("ops") or []:
        if not isinstance(raw_op, dict):
            continue
        name = str(raw_op.get("field", "")).strip()
        if not name or name not in allowed:
            logger.warning(f"Dropping slot op for unknown field '{name}'")
            continue

        op_kind = str(raw_op.get("op", "set")).strip().lower()
        if op_kind not in VALID_OPS:
            op_kind = "set"

        operator = str(raw_op.get("operator", "eq")).strip().lower()
        if operator not in VALID_OPERATORS:
            operator = "eq"

        ops.append(
            SlotOp(
                op=op_kind,
                field_name=name,
                operator=operator,
                value=raw_op.get("value"),
                is_correction=bool(raw_op.get("is_correction", False)),
                reason=str(raw_op.get("reason", "")).strip(),
            )
        )

    semantic_query = str(parsed.get("semantic_query", "") or "").strip()
    is_followup = bool(parsed.get("is_followup", False))
    reset_all = bool(parsed.get("reset_all", False))

    return ops, semantic_query, is_followup, reset_all


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def resolve_with_slots(
    current_query: str,
    slot_state: Optional[Dict[str, Slot]],
    metadata_field_info: List[AttributeInfo],
    llm: BaseChatModel,
    conversation_history: Optional[List[Dict]] = None,
    turn_index: int = 0,
) -> SlotResolution:
    """Resolve a turn as a diff against per-field slot state.

    Args:
        current_query: The user's new message.
        slot_state: Slots carried in from the previous turn.
        metadata_field_info: Schema of the active dataset — defines valid slots.
        llm: Chat model (Groq by default) used to produce the diff.
        conversation_history: Recent turns, for context.
        turn_index: Index of this turn, recorded on any slot written.

    Returns:
        SlotResolution. On any failure the caller should fall back to the
        original resolver — `used_fallback` says whether that is needed.
    """
    previous_state = dict(slot_state or {})
    allowed = field_names(metadata_field_info)

    if not allowed:
        return SlotResolution(
            slot_state=previous_state,
            previous_state=previous_state,
            semantic_query=current_query,
            effective_query=current_query,
            used_fallback=True,
            fallback_reason="No metadata schema available for slot tracking.",
        )

    prompt = _SLOT_PROMPT.format(
        schema=_format_schema(metadata_field_info),
        current_state=_format_state(previous_state),
        history=_format_history(conversation_history or []),
        query=current_query,
    )

    try:
        response = llm.invoke(prompt)
        ops, semantic_query, is_followup, reset_all = parse_slot_response(
            response.content, allowed
        )
    except Exception as e:
        logger.warning(f"Slot resolution failed ({e}); caller should fall back")
        return SlotResolution(
            slot_state=previous_state,
            previous_state=previous_state,
            semantic_query=current_query,
            effective_query=current_query,
            used_fallback=True,
            fallback_reason=f"{type(e).__name__}: {e}",
        )

    base_state = {} if reset_all else previous_state
    new_state, corrections = apply_ops(base_state, ops, turn_index=turn_index)

    # Derive the legacy filter_action so downstream nodes keep working unchanged.
    if reset_all or not previous_state:
        filter_action = "fresh"
    elif not ops:
        filter_action = "keep_all"
    else:
        filter_action = "modify"

    if reset_all:
        is_followup = False

    # With no slots there is nothing to render, so fall back to the raw message
    # rather than emitting an empty query.
    core = semantic_query if new_state else (semantic_query or current_query)
    effective_query = render_slots_as_query(core, new_state)
    if not effective_query.strip():
        effective_query = current_query

    resolution = SlotResolution(
        ops=ops,
        slot_state=new_state,
        previous_state=previous_state,
        semantic_query=semantic_query,
        effective_query=effective_query,
        is_followup=is_followup,
        filter_action=filter_action,
        carried_filters=compile_slots_to_filter(new_state),
        corrections=corrections,
        used_fallback=False,
    )

    logger.info(
        f"Slots resolved — action={filter_action}, ops={[o.op + ':' + o.field_name for o in ops]}, "
        f"corrections={len(corrections)}, state={sorted(new_state)}"
    )
    return resolution
