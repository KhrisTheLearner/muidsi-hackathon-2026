"""AgriFlow LangGraph agent definition.

This is the core orchestrator. It defines the graph:

    START -> router -> [planner | tool_caller] <-> tools -> synthesizer -> END

The router (pure Python, no LLM) fast-tracks simple queries directly to
the tool_caller, skipping the planner entirely. Complex queries still go
through the full planner decomposition.

The tool_caller asks the LLM which tools to call. The tool_executor runs
them. If the LLM decides more tools are needed, it loops back. Once it
responds without tool calls, we check if synthesis is needed.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes.planner import planner_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.tool_executor import ALL_TOOLS, tool_caller_node
from src.agent.state import AgriFlowState

# ---------------------------------------------------------------------------
# Smart router — skip planner for single-category queries (no LLM cost)
# ---------------------------------------------------------------------------
_DIRECT_PATTERNS: dict[str, re.Pattern] = {
    "data": re.compile(
        r"(food insecur|poverty|snap\b|food desert|food access|atlas|"
        r"census|fema|disaster|acs\b|demograph)", re.I,
    ),
    "sql": re.compile(r"(list tables|show tables|schema|run sql|query database|sqlite)", re.I),
    "viz": re.compile(r"(chart|graph|plot|heatmap|scatter.?map|visualiz|dashboard)", re.I),
    "route": re.compile(r"(route|deliver|schedule|logistics|transport|dispatch)", re.I),
    "weather": re.compile(r"(weather|forecast|temperature|\brain\b|drought|precipitation)", re.I),
    "ingest": re.compile(r"(load|import|ingest|profile|eda|add.*dataset|download.*data)", re.I),
    "analytics": re.compile(
        r"(train.*model|xgboost|random forest|shap|feature importan|"
        r"anomal|analytics|risk model|predict.*risk)", re.I,
    ),
    "ml": re.compile(r"(predict|scenario|evaluation|compare.*scenario)", re.I),
}


def _router_node(state: AgriFlowState) -> dict:
    """Classify the query and decide whether to skip the planner.

    If the query clearly maps to a single tool category, create a
    1-step plan and go direct. Otherwise, fall through to planner.
    """
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"plan": [], "current_step": 0, "tool_results": {}}

    # Check which categories match
    matches = []
    for cat, pattern in _DIRECT_PATTERNS.items():
        if pattern.search(query):
            matches.append(cat)

    # Single clear match -> skip planner, create 1-step plan
    if len(matches) == 1:
        cat = matches[0]
        plan = [f"[{cat}] {query}"]
        return {
            "plan": plan,
            "current_step": 0,
            "tool_results": {},
            "reasoning_trace": [f"Router: fast-track [{cat}] (skipped planner)"],
        }

    # No match or ambiguous -> flag for planner
    return {"plan": [], "current_step": 0, "tool_results": {}}


def _route_after_router(state: AgriFlowState) -> str:
    """After router: if we have a plan, go to tool_caller. Else, planner."""
    if state.get("plan"):
        return "tool_caller"
    return "planner"


def _should_continue(state: AgriFlowState) -> str:
    """Route after tool_caller: if the LLM made tool calls, execute them.
    Otherwise, check if we can skip synthesis.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If the tool_caller produced a substantive, structured answer,
    # skip the synthesizer entirely
    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content
        if len(content) > 200:
            has_structure = any(m in content for m in ["**", "##", "1.", "- "])
            if has_structure:
                return "done"

    return "synthesize"


def create_agent() -> StateGraph:
    """Build and compile the AgriFlow LangGraph agent.

    Returns:
        A compiled LangGraph that can be invoked with
        agent.invoke({"messages": [HumanMessage(content="...")]})
    """
    graph = StateGraph(AgriFlowState)

    # Node 0: Router - fast-track simple queries (no LLM cost)
    graph.add_node("router", _router_node)

    # Node 1: Planner - decomposes complex queries into sub-tasks
    graph.add_node("planner", planner_node)

    # Node 2: Tool Caller - LLM decides which tools to call
    graph.add_node("tool_caller", tool_caller_node)

    # Node 3: Tool Executor - actually runs the tools (LangGraph built-in)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Node 4: Synthesizer - combines results into final response
    graph.add_node("synthesizer", synthesizer_node)

    # Node 5: Finalizer - set final_answer from last AI message (no LLM)
    graph.add_node("finalizer", _finalizer_node)

    # --- Wire the edges ---

    # START -> router
    graph.set_entry_point("router")

    # router -> planner OR tool_caller
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"planner": "planner", "tool_caller": "tool_caller"},
    )

    # planner -> tool_caller
    graph.add_edge("planner", "tool_caller")

    # tool_caller -> tools OR synthesizer OR done (finalizer)
    graph.add_conditional_edges(
        "tool_caller",
        _should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesizer",
            "done": "finalizer",
        },
    )

    # tools -> tool_caller (loop)
    graph.add_edge("tools", "tool_caller")

    # synthesizer -> END
    graph.add_edge("synthesizer", END)

    # finalizer -> END
    graph.add_edge("finalizer", END)

    return graph.compile()


def _finalizer_node(state: AgriFlowState) -> dict:
    """Set final_answer from the last AI message (skip synthesis LLM call)."""
    last = state["messages"][-1] if state["messages"] else None
    answer = last.content if isinstance(last, AIMessage) and last.content else ""
    return {
        "final_answer": answer,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            "Finalizer: tool_caller answer was complete — skipped synthesizer"
        ],
    }


def run_agent(query: str, verbose: bool = True) -> dict[str, Any]:
    """Run AgriFlow on a single query.

    Args:
        query: The user's natural language question.
        verbose: If True, print reasoning trace as it runs.

    Returns:
        Full agent state including final_answer and reasoning_trace.
    """
    agent = create_agent()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "current_step": 0,
        "tool_results": {},
        "reasoning_trace": [],
        "charts": [],
        "final_answer": None,
    }

    # Stream events for visibility
    if verbose:
        print(f"\n{'='*60}")
        print(f"AgriFlow Query: {query}")
        print(f"{'='*60}\n")

    result = agent.invoke(initial_state)

    if verbose:
        print(f"\n{'-'*60}")
        print("Reasoning Trace:")
        for step in result.get("reasoning_trace", []):
            print(f"  {step}")
        print(f"{'-'*60}")
        print(f"\nFinal Answer:\n{result.get('final_answer', 'No answer generated.')}")

    return result
