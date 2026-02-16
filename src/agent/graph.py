"""AgriFlow LangGraph agent definition.

This is the core orchestrator. It defines the graph:

    START -> planner -> tool_caller <-> tool_executor -> synthesizer -> END
                            ^               |
                            +--- (loop) ----+

The tool_caller asks the LLM which tools to call. The tool_executor runs
them. If the LLM decides more tools are needed, it loops back. Once it
responds without tool calls, we move to synthesis.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes.planner import planner_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.tool_executor import ALL_TOOLS, tool_caller_node
from src.agent.state import AgriFlowState


def _should_continue(state: AgriFlowState) -> str:
    """Route after tool_caller: if the LLM made tool calls, execute them.
    Otherwise, go to synthesis.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesize"


def create_agent() -> StateGraph:
    """Build and compile the AgriFlow LangGraph agent.

    Returns:
        A compiled LangGraph that can be invoked with
        agent.invoke({"messages": [HumanMessage(content="...")]})
    """
    # --- Build the graph ---
    graph = StateGraph(AgriFlowState)

    # Node 1: Planner - decomposes query into sub-tasks
    graph.add_node("planner", planner_node)

    # Node 2: Tool Caller - LLM decides which tools to call
    graph.add_node("tool_caller", tool_caller_node)

    # Node 3: Tool Executor - actually runs the tools (LangGraph built-in)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Node 4: Synthesizer - combines results into final response
    graph.add_node("synthesizer", synthesizer_node)

    # --- Wire the edges ---

    # START -> planner
    graph.set_entry_point("planner")

    # planner -> tool_caller
    graph.add_edge("planner", "tool_caller")

    # tool_caller -> (conditional) -> tools OR synthesizer
    graph.add_conditional_edges(
        "tool_caller",
        _should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesizer",
        },
    )

    # tools -> tool_caller (loop back for more tool calls or final response)
    graph.add_edge("tools", "tool_caller")

    # synthesizer -> END
    graph.add_edge("synthesizer", END)

    return graph.compile()


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
