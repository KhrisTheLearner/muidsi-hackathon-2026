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
        # Find JSON array using regex (handles text before/after the JSON)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found")
        plan_data = json.loads(match.group())
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
