"""Planner node - decomposes user queries into tool-calling sub-tasks."""

from __future__ import annotations

import json
import queue
import re
import threading

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.llm import PLANNER_MODEL, get_llm
from src.agent.prompts.system import PLANNER_PROMPT
from src.agent.state import AgriFlowState

_PLANNER_TIMEOUT = 45  # seconds


def _fallback_plan(query: str) -> list[str]:
    """Build a sensible fallback plan when the LLM planner times out."""
    q = query.lower()
    plan = ["[data] " + query]
    if any(kw in q for kw in ["risk", "at risk", "vulnerable", "impact", "affect"]):
        plan.append("[analytics] Assess community vulnerability using collected data")
    plan.append("[viz] Visualize findings for the most at-risk communities")
    return plan[:4]


def _invoke_with_timeout(fn, args, timeout: float):
    """Run fn(*args) in a daemon thread with a timeout.

    Returns (result, None) on success or (None, exc) on timeout/error.
    Daemon threads don't block Python exit if abandoned.
    """
    result_queue: queue.Queue = queue.Queue()

    def _worker():
        try:
            result_queue.put(("ok", fn(*args)))
        except Exception as exc:
            result_queue.put(("err", exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        status, value = result_queue.get(timeout=timeout)
        if status == "ok":
            return value, None
        return None, value
    except queue.Empty:
        return None, TimeoutError(f"LLM call timed out after {timeout}s")


def planner_node(state: AgriFlowState) -> dict:
    """Decompose the user's question into a plan of sub-tasks."""
    llm = get_llm(model=PLANNER_MODEL, temperature=0.0)

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        *state["messages"],
    ]

    user_query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
    )

    response, err = _invoke_with_timeout(llm.invoke, (messages,), _PLANNER_TIMEOUT)
    if err is not None:
        fallback = _fallback_plan(user_query)
        label = "timed out" if isinstance(err, TimeoutError) else f"error: {type(err).__name__}"
        reasoning = [f"Plan: planner {label} — fallback plan ({len(fallback)} steps)"]
        for i, s in enumerate(fallback, 1):
            reasoning.append(f"  {i}. {s}")
        return {
            "plan": fallback,
            "current_step": 0,
            "tool_results": {},
            "reasoning_trace": state.get("reasoning_trace", []) + reasoning,
        }

    # Parse the plan from the LLM response
    try:
        text = response.content
        # Try to find JSON array — LLM often wraps it in ```json ... ``` blocks
        # First try: extract from code block
        code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if code_match:
            plan_data = json.loads(code_match.group(1))
        else:
            # Second try: find the outermost JSON array by bracket matching
            # Start from the first `[\n` or `[{` to avoid matching [data] tags
            array_match = re.search(r"\[\s*\{", text)
            if not array_match:
                raise ValueError("No JSON array found")
            start = array_match.start()
            # Find the matching closing bracket
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            plan_data = json.loads(text[start:end])

        plan = [step.get("task", str(step)) for step in plan_data]
        if not plan:
            raise ValueError("Empty plan")
        # Hard cap: never more than 4 steps to avoid timeouts
        plan = plan[:4]
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        # Fallback: treat the whole response as a single-step plan
        plan = [response.content]

    reasoning = [f"Plan: decomposed query into {len(plan)} sub-tasks"]
    for i, step in enumerate(plan, 1):
        reasoning.append(f"  {i}. {step}")

    return {
        "plan": plan,
        "current_step": 0,
        "tool_results": {},
        "reasoning_trace": state.get("reasoning_trace", []) + reasoning,
    }
