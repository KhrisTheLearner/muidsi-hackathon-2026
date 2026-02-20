"""Response evaluation tool — heuristic scoring, zero LLM token cost.

Logs every query+response to data/evaluations.jsonl.
Provides summary + improvement suggestions for the agriflow-evaluator Archia agent.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_EVAL_LOG  = Path("data/evaluations.jsonl")
_SUGG_LOG  = Path("data/suggestions.jsonl")
_SOURCES   = ["USDA", "NASS", "FEMA", "Census", "Atlas", "weather", "Open-Meteo"]


def _score(query: str, answer: str, tool_calls: list) -> dict[str, float]:
    words        = len(answer.split())
    numbers      = len(re.findall(r"\b\d+\.?\d*%?\b", answer))
    citations    = sum(1 for s in _SOURCES if s.lower() in answer.lower())
    n_tools      = len(tool_calls)
    has_specific = any(c.isdigit() for c in answer)

    return {
        "completeness": min(5.0, max(1.0, words / 60)),
        "data_quality":  min(5.0, max(1.0, 1.0 + numbers * 1.5)),
        "tool_usage":    min(5.0, max(1.0, n_tools)),
        "citation":      min(5.0, max(0.0, citations)),
        "specificity":   4.0 if has_specific else 1.0,
    }


_DEFLECTION_PHRASES = [
    "insufficient data", "need more data", "data gap", "resubmit",
    "acquire data", "data acquisition", "backfill", "data limitation",
]


def _suggest(scores: dict, tool_calls: list, answer: str) -> list[str]:
    out = []
    answer_lower = answer.lower()

    if scores["completeness"] < 2:
        out.append("ui: Response too brief — synthesizer may need a minimum-length prompt update")
    if scores["data_quality"] < 2:
        out.append("data: Response lacks specific numbers — check data source connectivity")
    if scores["tool_usage"] < 2:
        out.append("routing: Low tool usage — planner may be over-simplifying the query")
    if scores["citation"] < 1:
        out.append("prompt: No source citations — add citation requirement to SYNTHESIZER_PROMPT")
    errors = [tc for tc in tool_calls if isinstance(tc.get("args"), dict) and "error" in str(tc.get("args", ""))]
    if errors:
        out.append(f"model: Tool errors detected: {[e['tool'] for e in errors]}")

    # Detect deflection — agent said "need more data" instead of answering
    deflections = [p for p in _DEFLECTION_PHRASES if p in answer_lower]
    if deflections:
        out.append(f"prompt: DEFLECTION detected ({', '.join(deflections)}) — agent must answer with knowledge, not defer")

    return out


@tool
def evaluate_response(
    query: str,
    answer: str,
    tool_calls_json: str = "[]",
    duration_ms: int = 0,
) -> dict[str, Any]:
    """Evaluate an AgriFlow response for quality and log it to data/evaluations.jsonl.

    Uses heuristic scoring (no LLM call) to keep token cost at zero.
    Returns quality scores and improvement suggestions.
    """
    try:
        tool_calls = json.loads(tool_calls_json)
    except (json.JSONDecodeError, ValueError):
        tool_calls = []

    scores  = _score(query, answer, tool_calls)
    overall = round(sum(scores.values()) / len(scores), 2)
    suggestions = _suggest(scores, tool_calls, answer)

    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "query":       query[:300],
        "answer_len":  len(answer.split()),
        "tool_count":  len(tool_calls),
        "duration_ms": duration_ms,
        "scores":      scores,
        "overall":     overall,
        "suggestions": suggestions,
    }

    _EVAL_LOG.parent.mkdir(exist_ok=True)
    with _EVAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"overall_score": overall, "scores": scores, "suggestions": suggestions}


@tool
def get_evaluation_summary(last_n: int = 30) -> dict[str, Any]:
    """Return a summary of recent AgriFlow response evaluations and top improvement areas."""
    if not _EVAL_LOG.exists():
        return {"total": 0, "message": "No evaluations logged yet."}

    logs = []
    for line in _EVAL_LOG.read_text(encoding="utf-8").splitlines():
        try:
            logs.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            pass

    recent = logs[-last_n:]
    if not recent:
        return {"total": 0, "message": "No evaluations."}

    avg    = lambda key: round(sum(r[key] for r in recent) / len(recent), 2)
    issues = [s for r in recent for s in r.get("suggestions", [])]
    top    = [{"issue": k, "count": v} for k, v in Counter(issues).most_common(5)]
    low    = [r["query"] for r in recent if r["overall"] < 2.5][:5]

    return {
        "total":            len(logs),
        "recent":           len(recent),
        "avg_score":        avg("overall"),
        "avg_tool_calls":   avg("tool_count"),
        "avg_answer_words": avg("answer_len"),
        "top_issues":       top,
        "low_score_queries": low,
    }


@tool
def suggest_improvement(
    category: str,
    suggestion: str,
    priority: str = "medium",
) -> dict[str, Any]:
    """Log an improvement suggestion for AgriFlow's data model, UI, or prompts.

    category: 'ui' | 'model' | 'data' | 'prompt' | 'routing'
    priority: 'high' | 'medium' | 'low'
    """
    record = {
        "ts":         datetime.now(timezone.utc).isoformat(),
        "category":   category,
        "suggestion": suggestion,
        "priority":   priority,
    }
    _SUGG_LOG.parent.mkdir(exist_ok=True)
    with _SUGG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"logged": True, "category": category, "priority": priority}
