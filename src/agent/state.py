"""AgriFlow agent state definition.

This is the shared state that flows through every node in the LangGraph.
Each node reads what it needs and writes its results back to state.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgriFlowState(MessagesState):
    """State schema for the AgriFlow agent graph.

    Extends MessagesState (which gives us `messages: list` with
    add-message reducer) and adds AgriFlow-specific fields.
    """

    # The agent's decomposed plan for the current query
    plan: list[str]

    # Which step of the plan we're currently executing (0-indexed)
    current_step: int

    # Accumulated results from tool calls, keyed by tool name
    tool_results: dict[str, Any]

    # Visible reasoning trace shown to the user (transparency)
    reasoning_trace: list[str]

    # Generated chart/map Plotly specs (for frontend rendering)
    charts: list[dict[str, Any]]

    # Final structured response (set by responder node)
    final_answer: str | None
