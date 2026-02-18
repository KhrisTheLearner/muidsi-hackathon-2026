"""AgriFlow LangGraph agent definition.

This is the core orchestrator. It defines the graph:

    START -> router -> [planner | tool_caller] <-> tools -> synthesizer -> END

The router (pure Python, no LLM) fast-tracks simple queries directly to
the tool_caller, skipping the planner entirely. Complex queries still go
through the full planner decomposition.

For [viz] plan steps, the graph uses a direct_viz node that programmatically
calls chart tools — no LLM needed. This is more reliable than asking the LLM
to generate tool calls for chart creation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes.planner import planner_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.tool_executor import ALL_TOOLS, tool_caller_node
from src.agent.state import AgriFlowState
from src.agent.tools.chart_generator import (
    create_bar_chart, create_line_chart, create_scatter_map, create_risk_heatmap,
)

# ---------------------------------------------------------------------------
# Smart router — skip planner for single-category queries (no LLM cost)
# ---------------------------------------------------------------------------
_DIRECT_PATTERNS: dict[str, re.Pattern] = {
    "data": re.compile(
        r"(food insecur|poverty|snap\b|food desert|food access|atlas|"
        r"census|fema|disaster|acs\b|demograph)", re.I,
    ),
    "sql": re.compile(r"(list tables|show tables|schema|run sql|query database|sqlite)", re.I),
    "viz": re.compile(r"(chart|graph|plot|heatmap|scatter.?map|\bmap\b|visualiz|dashboard)", re.I),
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
    """Classify the query and decide whether to skip the planner."""
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"plan": [], "current_step": 0, "tool_results": {}}

    matches = []
    for cat, pattern in _DIRECT_PATTERNS.items():
        if pattern.search(query):
            matches.append(cat)

    if len(matches) == 1:
        cat = matches[0]
        # Viz queries always need data fetched first — create a 2-step plan
        if cat == "viz":
            plan = [
                f"[data] Retrieve relevant data for: {query}",
                f"[viz] {query}",
            ]
            return {
                "plan": plan,
                "current_step": 0,
                "tool_results": {},
                "reasoning_trace": [f"Router: fast-track [data]+[viz] (skipped planner)"],
            }
        plan = [f"[{cat}] {query}"]
        return {
            "plan": plan,
            "current_step": 0,
            "tool_results": {},
            "reasoning_trace": [f"Router: fast-track [{cat}] (skipped planner)"],
        }

    return {"plan": [], "current_step": 0, "tool_results": {}}


def _route_after_router(state: AgriFlowState) -> str:
    """After router: if we have a plan, go to tool_caller. Else, planner."""
    if state.get("plan"):
        return "tool_caller"
    return "planner"


def _should_continue(state: AgriFlowState) -> str:
    """Route after tool_caller: tools -> advance_step -> synthesize -> done."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check if there are more plan steps to execute
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if current_step + 1 < len(plan):
        return "advance_step"

    # If the tool_caller produced a substantive, structured answer,
    # skip the synthesizer entirely
    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content
        if len(content) > 200:
            has_structure = any(m in content for m in ["**", "##", "1.", "- "])
            if has_structure:
                return "done"

    return "synthesize"


def _advance_step_node(state: AgriFlowState) -> dict:
    """Advance current_step to the next plan step."""
    next_step = state.get("current_step", 0) + 1
    plan = state.get("plan", [])
    return {
        "current_step": next_step,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            f"Advancing to plan step {next_step + 1}/{len(plan)}"
        ],
    }


def _should_continue_after_action(state: AgriFlowState) -> str:
    """After direct_viz: check if there are more plan steps."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if current_step + 1 < len(plan):
        return "advance_step"
    return "synthesize"


def _route_after_advance(state: AgriFlowState) -> str:
    """After advancing, route [viz] steps to direct_viz, others to tool_caller."""
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    if step < len(plan):
        step_text = plan[step].lower()
        if "[viz]" in step_text:
            return "direct_viz"
    return "tool_caller"


# ---------------------------------------------------------------------------
# Direct viz execution — bypasses LLM, calls chart tools programmatically
# ---------------------------------------------------------------------------

def _extract_data_from_messages(messages: list) -> list[dict]:
    """Extract data rows from ToolMessage results in the conversation."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            # Check for common result formats
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
            if isinstance(data, dict):
                # Some tools return {"results": [...]}
                for key in ("results", "data", "rows"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
        except (json.JSONDecodeError, TypeError):
            continue
    return []


def _detect_chart_type(step_text: str, query: str) -> str:
    """Detect which chart type to create from plan step + query text."""
    combined = (step_text + " " + query).lower()
    if any(w in combined for w in ["map", "scatter", "geographic", "geo"]):
        return "scatter_map"
    if any(w in combined for w in ["heatmap", "heat map", "matrix"]):
        return "heatmap"
    if any(w in combined for w in ["line", "trend", "time series", "over time"]):
        return "line"
    return "bar"


def _detect_columns(rows: list[dict], chart_type: str, step_text: str) -> dict:
    """Auto-detect which columns to use for the chart axes."""
    if not rows:
        return {}

    cols = list(rows[0].keys())

    # Classify columns by type
    str_cols = []
    num_cols = []
    lat_cols = []
    lon_cols = []
    for c in cols:
        val = rows[0].get(c)
        cl = c.lower()
        if cl in ("lat", "latitude"):
            lat_cols.append(c)
        elif cl in ("lon", "lng", "longitude"):
            lon_cols.append(c)
        elif isinstance(val, (int, float)):
            num_cols.append(c)
        else:
            str_cols.append(c)

    if chart_type == "scatter_map":
        lat = lat_cols[0] if lat_cols else None
        lon = lon_cols[0] if lon_cols else None
        color = num_cols[0] if num_cols else None
        text = str_cols[0] if str_cols else None
        return {"lat_col": lat, "lon_col": lon, "color_col": color or "", "text_col": text or ""}

    if chart_type == "heatmap":
        x = str_cols[0] if str_cols else cols[0]
        y = str_cols[1] if len(str_cols) > 1 else (num_cols[0] if num_cols else cols[1])
        z = num_cols[0] if num_cols else cols[-1]
        return {"x_col": x, "y_col": y, "z_col": z}

    if chart_type == "line":
        x = str_cols[0] if str_cols else cols[0]
        y = ",".join(num_cols[:3]) if num_cols else cols[1]
        return {"x_col": x, "y_cols": y}

    # bar chart (default)
    x = str_cols[0] if str_cols else cols[0]

    # Try to pick the most relevant numeric column based on step text
    best_y = num_cols[0] if num_cols else cols[1] if len(cols) > 1 else cols[0]
    step_lower = step_text.lower()
    for nc in num_cols:
        if nc.lower() in step_lower or any(w in nc.lower() for w in ["insecur", "rate", "pov", "risk", "score"]):
            best_y = nc
            break

    return {"x_col": x, "y_col": best_y}


def _direct_viz_node(state: AgriFlowState) -> dict:
    """Execute a [viz] plan step by directly calling chart tools.

    No LLM needed — extracts data from previous ToolMessages, detects
    chart type and columns, and calls the appropriate chart tool.
    """
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    step_text = plan[step] if step < len(plan) else ""

    # Get the original query
    query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    # Extract data from previous tool results
    rows = _extract_data_from_messages(state["messages"])
    if not rows:
        return {
            "reasoning_trace": state.get("reasoning_trace", []) + [
                "Direct viz: no data found in tool results — skipping chart"
            ],
        }

    chart_type = _detect_chart_type(step_text, query)
    col_args = _detect_columns(rows, chart_type, step_text)

    # Sort and limit for readability
    if chart_type == "bar" and "y_col" in col_args:
        y_col = col_args["y_col"]
        try:
            rows = sorted(rows, key=lambda r: float(r.get(y_col, 0)), reverse=True)[:10]
        except (ValueError, TypeError):
            rows = rows[:10]

    title = step_text.replace("[viz]", "").strip()
    if not title or title == query:
        y_label = col_args.get("y_col", col_args.get("color_col", ""))
        title = f"Top Counties by {y_label}" if y_label else "AgriFlow Visualization"

    data_json = json.dumps(rows)

    # Call the chart tool directly
    try:
        if chart_type == "scatter_map" and col_args.get("lat_col") and col_args.get("lon_col"):
            result = create_scatter_map.invoke({
                "title": title,
                "data_json": data_json,
                **col_args,
            })
        elif chart_type == "heatmap" and all(k in col_args for k in ("x_col", "y_col", "z_col")):
            result = create_risk_heatmap.invoke({
                "title": title,
                "data_json": data_json,
                **col_args,
            })
        elif chart_type == "line" and "x_col" in col_args and "y_cols" in col_args:
            result = create_line_chart.invoke({
                "title": title,
                "data_json": data_json,
                **col_args,
            })
        else:
            # Default: bar chart
            result = create_bar_chart.invoke({
                "title": title,
                "data_json": data_json,
                "x_col": col_args.get("x_col", list(rows[0].keys())[0]),
                "y_col": col_args.get("y_col", list(rows[0].keys())[1] if len(rows[0]) > 1 else list(rows[0].keys())[0]),
                "horizontal": len(rows) > 5,
            })
    except Exception as e:
        return {
            "reasoning_trace": state.get("reasoning_trace", []) + [
                f"Direct viz: chart creation failed — {e}"
            ],
        }

    # Add the chart result as a ToolMessage so the synthesizer and API can find it
    # Names must match _CHART_TOOLS in src/api/main.py for extraction
    _TOOL_NAMES = {
        "bar": "create_bar_chart",
        "line": "create_line_chart",
        "scatter_map": "create_scatter_map",
        "heatmap": "create_risk_heatmap",
    }
    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
    chart_msg = ToolMessage(
        content=result_str,
        name=_TOOL_NAMES.get(chart_type, "create_bar_chart"),
        tool_call_id=f"direct_viz_{step}",
    )

    chart_id = result.get("chart_id", "unknown") if isinstance(result, dict) else "unknown"
    reasoning = state.get("reasoning_trace", []) + [
        f"Direct viz: created {chart_type} chart ({chart_id}) with {len(rows)} data points"
    ]

    return {
        "messages": [chart_msg],
        "reasoning_trace": reasoning,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_agent() -> StateGraph:
    """Build and compile the AgriFlow LangGraph agent."""
    graph = StateGraph(AgriFlowState)

    graph.add_node("router", _router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_caller", tool_caller_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("finalizer", _finalizer_node)
    graph.add_node("advance_step", _advance_step_node)
    graph.add_node("direct_viz", _direct_viz_node)

    # --- Wire the edges ---

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"planner": "planner", "tool_caller": "tool_caller"},
    )

    graph.add_edge("planner", "tool_caller")

    graph.add_conditional_edges(
        "tool_caller",
        _should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesizer",
            "done": "finalizer",
            "advance_step": "advance_step",
        },
    )

    # After advancing, route [viz] steps to direct_viz, others to tool_caller
    graph.add_conditional_edges(
        "advance_step",
        _route_after_advance,
        {"direct_viz": "direct_viz", "tool_caller": "tool_caller"},
    )

    # direct_viz -> check if more steps or synthesize
    graph.add_conditional_edges(
        "direct_viz",
        _should_continue_after_action,
        {"advance_step": "advance_step", "synthesize": "synthesizer"},
    )

    graph.add_edge("tools", "tool_caller")
    graph.add_edge("synthesizer", END)
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
    """Run AgriFlow on a single query."""
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
