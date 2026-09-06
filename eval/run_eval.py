"""
run_eval.py — Evaluation Harness
=================================
Runs every case in eval/test_queries.json through the REAL pipeline — the same
`rag.graph.run_graph` the Streamlit app calls — and scores it. Nothing here
reimplements retrieval, filtering or generation, and the pipeline is not
modified to accommodate the harness.

Metrics
-------
  filter_exact     — merged pre-filter equals the expected filter (normalised)
  filter_partial   — F1 over (field, operator, value) triples in the filter
  slot_accuracy    — expected slots present with the right operator/value
  precision@k      — retrieved documents that are in the expected set
  recall@k         — expected documents that made it into the top k
  faithfulness     — verdict from rag/answer_synthesis.py (real Gemini auditor)
  answer_relevance — fraction of expected key facts present in the answer

Multi-turn cases keep slot state and conversation history between turns, so
follow-ups and corrections are exercised exactly as they are in the app.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --categories correction,followup
    python -m eval.run_eval --limit 5 --out eval/report
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("rag").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUERIES = os.path.join(HERE, "test_queries.json")


# ──────────────────────────────────────────────────────────────────────────────
# Filter comparison
# ──────────────────────────────────────────────────────────────────────────────
def _normalise_value(value: Any) -> Any:
    """Make values comparable across the small variations models produce."""
    if isinstance(value, list):
        return tuple(sorted(_normalise_value(v) for v in value))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        # A bare year and a full date mean the same bound here.
        if len(s) == 4 and s.isdigit():
            return f"{s}-01-01"
        try:
            return float(s)
        except ValueError:
            return s.lower()
    return value


def filter_triples(filter_dict: Any) -> set:
    """Flatten a MongoDB filter into comparable (field, op, value) triples."""
    triples = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in ("$and", "$or", "$nor"):
                if isinstance(value, list):
                    for item in value:
                        walk(item)
            elif key.startswith("$"):
                continue
            elif isinstance(value, dict):
                ops = {k: v for k, v in value.items() if k.startswith("$")}
                if ops:
                    for op, operand in ops.items():
                        triples.add((key, op, _normalise_value(operand)))
                else:
                    walk(value)
            else:
                triples.add((key, "$eq", _normalise_value(value)))

    walk(filter_dict)
    return triples


def score_filter(actual: Any, expected: Any) -> Tuple[bool, float]:
    """(exact_match, partial_F1) between the produced and expected filter."""
    a, e = filter_triples(actual), filter_triples(expected)

    if not a and not e:
        return True, 1.0
    if not a or not e:
        return False, 0.0

    overlap = len(a & e)
    precision = overlap / len(a)
    recall = overlap / len(e)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return a == e, f1


def score_slots(actual_slots: Dict, expected_slots: Optional[Dict]) -> Optional[float]:
    """Fraction of expected slots present with the right operator and value."""
    if expected_slots is None:
        return None
    if not expected_slots:
        return 1.0 if not actual_slots else 0.0

    hits = 0
    for field_name, spec in expected_slots.items():
        got = (actual_slots or {}).get(field_name)
        if not got:
            continue
        op_ok = str(got.get("operator", "")).lower() == str(spec.get("operator", "")).lower()
        val_ok = _normalise_value(got.get("value")) == _normalise_value(spec.get("value"))
        if op_ok and val_ok:
            hits += 1
    return hits / len(expected_slots)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval + answer scoring
# ──────────────────────────────────────────────────────────────────────────────
def doc_matches(document, key: str) -> bool:
    """A retrieved document counts as `key` if the key appears in its text."""
    text = (getattr(document, "page_content", "") or "").lower()
    return key.lower() in text


def score_retrieval(documents: List, expected_docs: Optional[List[str]]) -> Tuple[Optional[float], Optional[float]]:
    """(precision@k, recall@k) against the expected document keys."""
    if expected_docs is None:
        return None, None
    if not expected_docs:
        return None, None
    if not documents:
        return 0.0, 0.0

    matched_docs = sum(1 for d in documents if any(doc_matches(d, k) for k in expected_docs))
    found_keys = sum(1 for k in expected_docs if any(doc_matches(d, k) for d in documents))

    precision = matched_docs / len(documents)
    recall = found_keys / len(expected_docs)
    return precision, recall


_REFUSAL_MARKERS = (
    "do not contain", "does not contain", "no information", "not available",
    "cannot answer", "can't answer", "not provided", "no documents",
    "not mentioned", "does not mention", "do not mention", "unable to",
    "not specified", "no mention",
)


def score_answer_relevance(answer: str, expected_contains: Optional[List[str]]) -> Optional[float]:
    """Fraction of the expected key facts that appear in the answer."""
    if not expected_contains:
        return None
    text = (answer or "").lower()
    if not text:
        return 0.0
    hits = sum(1 for key in expected_contains if key.lower() in text)
    return hits / len(expected_contains)


def looks_like_refusal(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in _REFUSAL_MARKERS)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline wiring — identical to what app.py builds
# ──────────────────────────────────────────────────────────────────────────────
def build_context(mode: str, collection_name: str, llm, embeddings, faith_llm,
                  metadata_field_info, content_description, conversation_history,
                  slot_state, turn_index):
    from rag.config_loader import config
    from rag.graph import MODE_DOCUMENTS, PipelineContext
    from rag.metadata_filter import MetadataFilter
    from rag.utils.mongodb_helper import get_mongo_collection

    collection = get_mongo_collection(
        db_name=config["database_name"], collection_name=collection_name
    )
    mf = MetadataFilter(
        collection=collection,
        llm=llm,
        metadata_field_info=metadata_field_info,
        document_content_description=content_description,
    )
    return PipelineContext(
        llm=llm,
        embeddings=embeddings,
        collection=collection,
        metadata_field_info=metadata_field_info,
        document_content_description=content_description,
        metadata_filter=mf,
        faithfulness_llm=faith_llm,
        mode=mode,
        conversation_history=conversation_history,
        active_filters={},
        max_relaxations=int(config.get("max_filter_relaxations", 2)),
        enable_synthesis=bool(config.get("enable_answer_synthesis", True)),
        enable_faithfulness=bool(config.get("enable_faithfulness_check", True)),
        enable_clarification=bool(config.get("enable_clarification", True)),
        enable_slot_memory=bool(config.get("enable_slot_memory", True)),
        slot_state=slot_state,
        turn_index=turn_index,
    )


def run_case(case: Dict, collections: Dict, shared: Dict) -> Dict:
    """Run every turn of one case, carrying slot state between them."""
    from rag.graph import MODE_DOCUMENTS, MODE_STRUCTURED, run_graph

    mode = case.get("mode", MODE_STRUCTURED)
    collection_name = collections.get(
        "documents" if mode == MODE_DOCUMENTS else "structured"
    )

    if mode == MODE_DOCUMENTS:
        metadata_field_info, content_description = [], "A chunk of text from an uploaded document"
    else:
        metadata_field_info = shared["eval_fields"]
        content_description = shared["eval_content_description"]

    conversation_history: List[Dict] = []
    slot_state: Dict = {}
    turn_results = []

    for turn_index, turn in enumerate(case.get("turns", [])):
        query = turn["query"]
        ctx = build_context(
            mode=mode,
            collection_name=collection_name,
            llm=shared["llm"],
            embeddings=shared["embeddings"],
            faith_llm=shared["faith_llm"],
            metadata_field_info=metadata_field_info,
            content_description=content_description,
            conversation_history=conversation_history,
            slot_state=slot_state,
            turn_index=turn_index,
        )

        started = time.perf_counter()
        error = ""
        try:
            state = run_graph(query, ctx, mode=mode)
        except Exception as e:                                # keep the suite going
            logger.error(f"[{case['id']} turn {turn_index}] pipeline error: {e}")
            state, error = {}, f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - started

        documents = state.get("documents", []) or []
        answer = state.get("answer", "") or ""
        faith = state.get("faithfulness", {}) or {}
        actual_filter = state.get("merged_filter", {}) or {}
        actual_slots = state.get("slot_state", {}) or {}

        exact, partial = score_filter(actual_filter, turn.get("expected_filter"))
        slot_score = score_slots(actual_slots, turn.get("expected_slots"))
        precision, recall = score_retrieval(documents, turn.get("expected_docs"))
        relevance = score_answer_relevance(answer, turn.get("expected_answer_contains"))

        # Correction / clearing expectations (feature 1)
        corrected_fields = [c.get("field") for c in (state.get("slot_corrections") or [])]
        expect_correction = turn.get("expect_correction_on")
        correction_ok = (
            None if not expect_correction
            else all(f in corrected_fields for f in expect_correction)
        )
        expect_cleared = turn.get("expect_cleared")
        cleared_ok = (
            None if not expect_cleared
            else all(f not in actual_slots for f in expect_cleared)
        )
        expect_absent = turn.get("expect_absent_slots")
        absent_ok = (
            None if not expect_absent
            else all(f not in actual_slots for f in expect_absent)
        )
        refusal_ok = (
            None if not turn.get("expect_refusal")
            else looks_like_refusal(answer)
        )

        turn_results.append(
            {
                "turn_index": turn_index,
                "query": query,
                "error": error,
                "actual_filter": actual_filter,
                "expected_filter": turn.get("expected_filter"),
                "actual_slots": {
                    k: {"operator": v.get("operator"), "value": v.get("value")}
                    for k, v in actual_slots.items()
                },
                "expected_slots": turn.get("expected_slots"),
                "resolution_mode": state.get("resolution_mode", ""),
                "corrections": corrected_fields,
                "retrieved": [
                    (getattr(d, "page_content", "") or "")[:70] for d in documents
                ],
                "answer": answer,
                "filter_exact": exact,
                "filter_partial": partial,
                "slot_accuracy": slot_score,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "answer_relevance": relevance,
                "faithfulness_verdict": faith.get("verdict", "unknown"),
                "faithfulness_score": faith.get("score", 0.0),
                "faithfulness_method": faith.get("method", ""),
                "correction_ok": correction_ok,
                "cleared_ok": cleared_ok,
                "absent_slots_ok": absent_ok,
                "refusal_ok": refusal_ok,
                "nodes": [t.get("node") for t in (state.get("trace") or [])],
                "elapsed_s": round(elapsed, 2),
            }
        )

        # Carry state forward exactly as the app does.
        if not state.get("clarification_question"):
            conversation_history.append(
                {"query": query, "filter_used": actual_filter}
            )
            if state.get("resolution_mode") == "slots":
                slot_state = actual_slots

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorised"),
        "mode": mode,
        "description": case.get("description", ""),
        "turns": turn_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation + reporting
# ──────────────────────────────────────────────────────────────────────────────
def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def aggregate(results: List[Dict]) -> Dict:
    per_category: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    overall: Dict[str, List] = defaultdict(list)
    errors = []

    for case in results:
        cat = case["category"]
        for turn in case["turns"]:
            if turn["error"]:
                errors.append({"id": case["id"], "query": turn["query"], "error": turn["error"]})
                continue
            for metric in ("filter_exact", "filter_partial", "slot_accuracy",
                           "precision_at_k", "recall_at_k", "answer_relevance",
                           "correction_ok", "cleared_ok", "absent_slots_ok", "refusal_ok"):
                value = turn.get(metric)
                if value is None:
                    continue
                value = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
                per_category[cat][metric].append(value)
                overall[metric].append(value)

            grounded = 1.0 if turn["faithfulness_verdict"] == "grounded" else 0.0
            per_category[cat]["faithful_rate"].append(grounded)
            overall["faithful_rate"].append(grounded)
            per_category[cat]["faithfulness_score"].append(float(turn["faithfulness_score"] or 0.0))
            overall["faithfulness_score"].append(float(turn["faithfulness_score"] or 0.0))
            per_category[cat]["latency_s"].append(turn["elapsed_s"])
            overall["latency_s"].append(turn["elapsed_s"])

    return {
        "per_category": {
            cat: {m: _mean(v) for m, v in metrics.items()}
            for cat, metrics in per_category.items()
        },
        "overall": {m: _mean(v) for m, v in overall.items()},
        "turn_count": sum(len(c["turns"]) for c in results),
        "case_count": len(results),
        "errors": errors,
    }


def _fmt(value: Optional[float], pct: bool = True) -> str:
    if value is None:
        return "   —  "
    return f"{value:6.1%}" if pct else f"{value:6.2f}"


def print_report(summary: Dict, results: List[Dict]):
    o = summary["overall"]
    print()
    print("=" * 96)
    print("  SmartFilteringRAG — EVALUATION REPORT")
    print("=" * 96)
    print(f"  {summary['case_count']} cases / {summary['turn_count']} turns"
          f"   ·   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    header = (f"  {'CATEGORY':<18}{'n':>4}{'filt=':>8}{'filtF1':>8}{'slots':>8}"
              f"{'P@k':>8}{'R@k':>8}{'faith':>8}{'relev':>8}{'sec':>7}")
    print(header)
    print("  " + "-" * 93)

    counts = defaultdict(int)
    for case in results:
        counts[case["category"]] += len(case["turns"])

    for cat in sorted(summary["per_category"]):
        m = summary["per_category"][cat]
        print(f"  {cat:<18}{counts[cat]:>4}"
              f"{_fmt(m.get('filter_exact')):>8}{_fmt(m.get('filter_partial')):>8}"
              f"{_fmt(m.get('slot_accuracy')):>8}{_fmt(m.get('precision_at_k')):>8}"
              f"{_fmt(m.get('recall_at_k')):>8}{_fmt(m.get('faithful_rate')):>8}"
              f"{_fmt(m.get('answer_relevance')):>8}"
              f"{_fmt(m.get('latency_s'), pct=False):>7}")

    print("  " + "-" * 93)
    print(f"  {'OVERALL':<18}{summary['turn_count']:>4}"
          f"{_fmt(o.get('filter_exact')):>8}{_fmt(o.get('filter_partial')):>8}"
          f"{_fmt(o.get('slot_accuracy')):>8}{_fmt(o.get('precision_at_k')):>8}"
          f"{_fmt(o.get('recall_at_k')):>8}{_fmt(o.get('faithful_rate')):>8}"
          f"{_fmt(o.get('answer_relevance')):>8}"
          f"{_fmt(o.get('latency_s'), pct=False):>7}")
    print()

    # Slot-memory specific checks (feature 1)
    for label, key in (("Corrections applied to the right slot", "correction_ok"),
                       ("Slots cleared on request", "cleared_ok"),
                       ("Stale slots dropped on topic change", "absent_slots_ok"),
                       ("Declined when the answer is absent", "refusal_ok")):
        if o.get(key) is not None:
            print(f"  {label:<44}{_fmt(o[key])}")
    print(f"  {'Mean faithfulness score (Gemini auditor)':<44}{_fmt(o.get('faithfulness_score'))}")
    print()

    if summary["errors"]:
        print(f"  ⚠ {len(summary['errors'])} pipeline error(s):")
        for e in summary["errors"][:10]:
            print(f"     [{e['id']}] {e['query'][:48]} -> {e['error'][:70]}")
        print()

    # Anything that missed an exact filter match, for quick inspection
    misses = [
        (c["id"], t["query"], t["actual_filter"], t["expected_filter"])
        for c in results for t in c["turns"]
        if t.get("expected_filter") is not None and not t["filter_exact"] and not t["error"]
    ]
    if misses:
        print(f"  Filter mismatches ({len(misses)}):")
        for cid, q, actual, expected in misses[:12]:
            print(f"     [{cid}] {q[:44]}")
            print(f"          expected {json.dumps(expected)}")
            print(f"          actual   {json.dumps(actual)}")
        print()
    print("=" * 96)


def write_reports(summary: Dict, results: List[Dict], out_prefix: str, meta: Dict):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "summary": summary,
        "results": results,
    }
    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    o = summary["overall"]
    lines = [
        "# SmartFilteringRAG — Evaluation Report",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{summary['case_count']} cases / {summary['turn_count']} turns_",
        "",
        f"- Generator model: `{meta.get('model')}`",
        f"- Faithfulness auditor: `{meta.get('faithfulness_model')}`",
        f"- Structured corpus: `{meta.get('structured_collection')}`",
        f"- Document corpus: `{meta.get('documents_collection')}`",
        "",
        "## Overall",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Filter exact match | {_fmt(o.get('filter_exact')).strip()} |",
        f"| Filter partial (F1) | {_fmt(o.get('filter_partial')).strip()} |",
        f"| Slot accuracy | {_fmt(o.get('slot_accuracy')).strip()} |",
        f"| Precision@k | {_fmt(o.get('precision_at_k')).strip()} |",
        f"| Recall@k | {_fmt(o.get('recall_at_k')).strip()} |",
        f"| Answers judged grounded | {_fmt(o.get('faithful_rate')).strip()} |",
        f"| Mean faithfulness score | {_fmt(o.get('faithfulness_score')).strip()} |",
        f"| Answer relevance | {_fmt(o.get('answer_relevance')).strip()} |",
        f"| Mean latency | {_fmt(o.get('latency_s'), pct=False).strip()}s |",
        "",
        "## Per category",
        "",
        "| Category | Turns | Filter= | Filter F1 | Slots | P@k | R@k | Grounded | Relevance | Sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    counts = defaultdict(int)
    for case in results:
        counts[case["category"]] += len(case["turns"])
    for cat in sorted(summary["per_category"]):
        m = summary["per_category"][cat]
        lines.append(
            f"| {cat} | {counts[cat]} | {_fmt(m.get('filter_exact')).strip()} "
            f"| {_fmt(m.get('filter_partial')).strip()} | {_fmt(m.get('slot_accuracy')).strip()} "
            f"| {_fmt(m.get('precision_at_k')).strip()} | {_fmt(m.get('recall_at_k')).strip()} "
            f"| {_fmt(m.get('faithful_rate')).strip()} | {_fmt(m.get('answer_relevance')).strip()} "
            f"| {_fmt(m.get('latency_s'), pct=False).strip()} |"
        )

    extras = [
        ("Corrections applied to the right slot", "correction_ok"),
        ("Slots cleared on request", "cleared_ok"),
        ("Stale slots dropped on topic change", "absent_slots_ok"),
        ("Declined when the answer is absent", "refusal_ok"),
    ]
    present = [(label, key) for label, key in extras if o.get(key) is not None]
    if present:
        lines += ["", "## Slot-memory behaviour", "", "| Check | Score |", "| --- | --- |"]
        lines += [f"| {label} | {_fmt(o[key]).strip()} |" for label, key in present]

    if summary["errors"]:
        lines += ["", "## Pipeline errors", ""]
        lines += [f"- `{e['id']}` {e['query']} — {e['error']}" for e in summary["errors"]]

    md_path = f"{out_prefix}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return json_path, md_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run the SmartFilteringRAG eval suite.")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--categories", default="", help="comma-separated filter")
    parser.add_argument("--ids", default="", help="comma-separated case ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(HERE, "report"))
    args = parser.parse_args()

    with open(args.queries, encoding="utf-8") as fh:
        spec = json.load(fh)

    cases = spec["cases"]
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        cases = [c for c in cases if c.get("category") in wanted]
    if args.ids:
        wanted_ids = {c.strip() for c in args.ids.split(",") if c.strip()}
        cases = [c for c in cases if c["id"] in wanted_ids]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No cases selected.")
        return 1

    # Build the models once and share them across cases.
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    from eval.seed_eval_data import EVAL_CONTENT_DESCRIPTION, EVAL_METADATA_FIELDS
    from rag.config_loader import config
    from rag.graph import make_faithfulness_llm

    llm = ChatOpenAI(
        model=config["model"],
        max_tokens=1024,
        openai_api_key=os.getenv("OPEN_AI_API_KEY"),
        openai_api_base=os.getenv("OPEN_API_BASE"),
    )
    faith_llm = make_faithfulness_llm()
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])

    shared = {
        "llm": llm,
        "faith_llm": faith_llm,
        "embeddings": embeddings,
        "eval_fields": EVAL_METADATA_FIELDS,
        "eval_content_description": EVAL_CONTENT_DESCRIPTION,
    }
    collections = spec.get("collections", {})

    meta = {
        "model": config["model"],
        "faithfulness_model": getattr(faith_llm, "model_name", "?"),
        "faithfulness_endpoint": str(getattr(faith_llm, "openai_api_base", "")),
        "embedding_model": config["embedding_model"],
        "structured_collection": collections.get("structured"),
        "documents_collection": collections.get("documents"),
        "database": config["database_name"],
    }

    print(f"Running {len(cases)} case(s) against the live stack…")
    print(f"  generator : {meta['model']}")
    print(f"  auditor   : {meta['faithfulness_model']} @ {meta['faithfulness_endpoint']}")
    print(f"  corpora   : {meta['structured_collection']} / {meta['documents_collection']}")
    print()

    results = []
    started = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        turns = len(case.get("turns", []))
        print(f"  [{i:>2}/{len(cases)}] {case['id']:<14} ({case.get('category','')}, {turns} turn(s))…",
              end="", flush=True)
        case_started = time.perf_counter()
        result = run_case(case, collections, shared)
        results.append(result)
        exact = sum(1 for t in result["turns"] if t["filter_exact"])
        print(f" {time.perf_counter() - case_started:5.1f}s  filter={exact}/{turns}")

    summary = aggregate(results)
    print_report(summary, results)

    json_path, md_path = write_reports(summary, results, args.out, meta)
    print(f"  Total wall time: {time.perf_counter() - started:.0f}s")
    print(f"  Wrote {json_path}")
    print(f"  Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
