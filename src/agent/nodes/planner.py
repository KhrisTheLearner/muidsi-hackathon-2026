"""Planner node - decomposes user queries into tool-calling sub-tasks."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, SystemMessage

from src.agent.llm import PLANNER_MODEL, get_llm
from src.agent.prompts.system import PLANNER_PROMPT
from src.agent.state import AgriFlowState


def planner_node(state: AgriFlowState) -> dict:
    """Decompose the user's question into a plan of sub-tasks.

    Reads the latest user message, sends it to the LLM with the planner
    prompt, and parses the returned JSON plan into state.
    """
    llm = get_llm(model=PLANNER_MODEL, temperature=0.0)

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        *state["messages"],
    ]

    response = llm.invoke(messages)

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
