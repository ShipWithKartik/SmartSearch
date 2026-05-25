"""
query_resolver.py — Multi-Turn Query Resolution
=================================================
Detects follow-up queries and rewrites them into self-contained versions
so the existing pipeline can process them without modification.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResolvedQuery:
    """Result of query resolution — tells the pipeline how to proceed."""

    is_followup: bool = False
    resolved_query: str = ""
    filter_action: str = "fresh"  # "fresh" | "keep_all" | "modify"
    carried_filters: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Prompt for follow-up detection and query rewriting
# ──────────────────────────────────────────────────────────────────────────────
_RESOLVER_PROMPT = """\
You are a conversation analyst for a search system. Your job is to determine \
whether the user's CURRENT query is a follow-up to their previous queries, \
or a completely fresh/new query.

## Conversation History (last turns)
{conversation_history}

## Current Filters Active
{active_filters}

## Current User Query
"{current_query}"

## Instructions

Analyze the current query and determine:

1. **is_followup**: Is this query referencing or building upon the previous \
conversation? Look for:
   - Pronouns referring to prior results ("those", "them", "it", "these")
   - Comparative/modifier phrases ("but cheaper", "same but", "more like that")
   - Partial constraints that only make sense with prior context \
("what about before 1990", "in English", "by the same director")
   - Requests for more results ("show me more", "any others", "what else")

2. **filter_action**: One of:
   - "fresh" — completely new topic, discard all previous filters
   - "keep_all" — keep all existing filters (e.g., "show me more")
   - "modify" — keep some filters but change/add others

3. **resolved_query**: Rewrite the current query into a FULLY SELF-CONTAINED \
version that includes ALL relevant context from the conversation. This must be \
understandable without any conversation history.
   - For "fresh": just return the original query unchanged
   - For "keep_all": return the PREVIOUS query (since user wants more of the same)
   - For "modify": combine previous context + new constraints into one query

## Few-Shot Examples

### Example 1
History: [{{"query": "show me anime movies", "filters": {{"genre": {{"$in": ["anime"]}}}}}}]
Current: "show me more"
Response: {{"is_followup": true, "filter_action": "keep_all", \
"resolved_query": "anime movies"}}

### Example 2
History: [{{"query": "show me anime movies", "filters": {{"genre": {{"$in": ["anime"]}}}}}}]
Current: "what about before 1990"
Response: {{"is_followup": true, "filter_action": "modify", \
"resolved_query": "anime movies released before 1990"}}

### Example 3
History: [{{"query": "anime movies", "filters": {{"genre": {{"$in": ["anime"]}}}}}}]
Current: "recommend a comedy"
Response: {{"is_followup": false, "filter_action": "fresh", \
"resolved_query": "recommend a comedy"}}

### Example 4
History: [{{"query": "electronics under $500", \
"filters": {{"$and": [{{"category": "Electronics"}}, {{"price": {{"$lt": 500}}}}]}}}}]
Current: "same but from Apple"
Response: {{"is_followup": true, "filter_action": "modify", \
"resolved_query": "Apple electronics under $500"}}

### Example 5
History: [{{"query": "thriller movies by Nolan", \
"filters": {{"$and": [{{"genre": {{"$in": ["thriller"]}}}}, \
{{"director": "Christopher Nolan"}}]}}}}]
Current: "same but in anime genre"
Response: {{"is_followup": true, "filter_action": "modify", \
"resolved_query": "anime movies by Christopher Nolan"}}

### Example 6
History: []
Current: "show me action movies"
Response: {{"is_followup": false, "filter_action": "fresh", \
"resolved_query": "show me action movies"}}

## Rules
- Return ONLY valid JSON, no markdown, no explanation
- If conversation history is empty, it is ALWAYS a fresh query
- When in doubt, classify as fresh (safer than incorrectly carrying filters)
- The resolved_query must be a natural language search query, NOT a filter expression

Return JSON:
{{"is_followup": bool, "filter_action": "fresh|keep_all|modify", \
"resolved_query": "the rewritten query"}}
"""


def resolve_query(
    current_query: str,
    conversation_history: List[Dict],
    active_filters: Dict,
    llm: BaseChatModel,
) -> ResolvedQuery:
    """Determine if the current query is a follow-up and resolve it.

    Args:
        current_query: The user's new query string.
        conversation_history: Last N turns as [{query, filter_used}].
        active_filters: The last merged_filter dict from the previous turn.
        llm: The configured LLM.

    Returns:
        ResolvedQuery with classification and rewritten query.
    """
    # ── No history → always fresh ────────────────────────────────────────
    if not conversation_history:
        logger.info("No conversation history — treating as fresh query")
        return ResolvedQuery(
            is_followup=False,
            resolved_query=current_query,
            filter_action="fresh",
            carried_filters={},
        )

    # ── Build prompt ─────────────────────────────────────────────────────
    # Use last 3 turns max to keep prompt small
    recent_history = conversation_history[-3:]
    history_str = json.dumps(recent_history, indent=2, default=str)
    filters_str = json.dumps(active_filters, indent=2, default=str) if active_filters else "{}"

    prompt = _RESOLVER_PROMPT.format(
        conversation_history=history_str,
        active_filters=filters_str,
        current_query=current_query,
    )

    logger.info(f"Resolving query: '{current_query}' with {len(recent_history)} turns of history")

    try:
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines)

        parsed = json.loads(raw_text)

        result = ResolvedQuery(
            is_followup=parsed.get("is_followup", False),
            resolved_query=parsed.get("resolved_query", current_query),
            filter_action=parsed.get("filter_action", "fresh"),
            carried_filters=active_filters if parsed.get("is_followup", False) else {},
        )

        # Validate filter_action
        if result.filter_action not in ("fresh", "keep_all", "modify"):
            result.filter_action = "fresh"
            result.is_followup = False

        logger.info(
            f"Query resolved — followup={result.is_followup}, "
            f"action={result.filter_action}, "
            f"resolved='{result.resolved_query}'"
        )
        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Query resolution failed ({e}), treating as fresh query")
        return ResolvedQuery(
            is_followup=False,
            resolved_query=current_query,
            filter_action="fresh",
            carried_filters={},
        )
    except Exception as e:
        logger.error(f"Unexpected error in query resolution: {e}")
        return ResolvedQuery(
            is_followup=False,
            resolved_query=current_query,
            filter_action="fresh",
            carried_filters={},
        )
