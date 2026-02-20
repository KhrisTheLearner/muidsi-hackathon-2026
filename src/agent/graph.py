"""AgriFlow LangGraph agent definition.

This is the core orchestrator. It defines the graph:

    START -> router -> [planner | tool_caller] <-> tools -> chart_validator -> synthesizer -> END

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

from src.agent.nodes.chart_validator import validate_chart_messages
from src.agent.nodes.planner import planner_node
from src.agent.nodes.synthesizer import synthesizer_node
from src.agent.nodes.tool_executor import ALL_TOOLS, tool_caller_node
from src.agent.state import AgriFlowState
from src.agent.tools.chart_generator import (
    create_bar_chart, create_line_chart, create_scatter_map, create_risk_heatmap,
    create_choropleth_map, create_chart,
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
    return "chart_validator"


def _chart_validator_node(state: AgriFlowState) -> dict:
    """Validate and fix all chart Plotly specs in the message history.

    Catches swapped axes, raw column names, empty traces, bad FIPS codes,
    reversed lat/lon, and missing labels — fixing them before the frontend
    ever sees them.
    """
    messages = state.get("messages", [])
    fixed_messages, fix_log = validate_chart_messages(messages)

    updates: dict = {}
    if fix_log:
        updates["messages"] = fixed_messages
        updates["reasoning_trace"] = state.get("reasoning_trace", []) + [
            f"Chart validator: applied {len(fix_log)} fix(es) — " + "; ".join(fix_log[:5])
        ]
    else:
        updates["reasoning_trace"] = state.get("reasoning_trace", []) + [
            "Chart validator: all charts passed validation"
        ]
    return updates


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

def _extract_data_from_messages(messages: list, step_text: str = "") -> list[dict]:
    """Extract the best data rows from ToolMessage results for visualization.

    Collects ALL tool result datasets, then picks the one most relevant to
    the viz step being executed. Prefers datasets with:
    1. Columns that match keywords in the step text
    2. More columns (richer data)
    3. County/geographic columns (for maps)
    """
    datasets: list[tuple[str, list[dict]]] = []  # (tool_name, rows)

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        # Skip chart tool results (they contain plotly specs, not data)
        if hasattr(msg, "name") and msg.name in (
            "create_bar_chart", "create_line_chart", "create_scatter_map",
            "create_risk_heatmap", "create_choropleth_map", "create_chart",
        ):
            continue
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            rows = None
            if isinstance(data, list) and data and isinstance(data[0], dict):
                rows = data
            elif isinstance(data, dict):
                for key in ("results", "data", "rows"):
                    if key in data and isinstance(data[key], list) and data[key]:
                        if isinstance(data[key][0], dict):
                            rows = data[key]
                            break
            if rows:
                tool_name = getattr(msg, "name", "unknown")
                datasets.append((tool_name, rows))
        except (json.JSONDecodeError, TypeError):
            continue

    if not datasets:
        return []

    # If only one dataset, return it
    if len(datasets) == 1:
        return datasets[0][1]

    # Score each dataset for relevance to the step text
    step_lower = step_text.lower()
    best_score = -1
    best_rows = datasets[-1][1]  # default: most recent

    for tool_name, rows in datasets:
        score = 0
        cols = {k.lower() for k in rows[0].keys()} if rows else set()

        # Prefer datasets with county/geographic info (needed for maps)
        if cols & {"county", "county_name", "name", "fips", "fipscode", "fips_code"}:
            score += 10

        # Prefer datasets with risk/insecurity columns (often the target for viz)
        if cols & {"foodinsec_13_15", "food_insecurity", "risk_score", "povrate15",
                   "poverty_rate", "predicted_risk", "composite_risk"}:
            score += 8

        # Prefer datasets with more columns (richer)
        score += min(len(cols), 10)

        # Prefer datasets with more rows (more data points)
        score += min(len(rows), 5)

        # Keyword matching: if step mentions "insecurity" and data has it, boost
        for kw in ["insecur", "poverty", "risk", "snap", "desert", "yield",
                    "corn", "weather", "disaster", "census", "demograph"]:
            if kw in step_lower:
                if any(kw in c for c in cols):
                    score += 5
                if kw in tool_name.lower():
                    score += 3

        # Penalize datasets that are mostly time-series without county data
        # (e.g., NASS yearly data that just has year + Value)
        if cols == {"year", "value"} or (len(cols) <= 3 and "year" in cols):
            score -= 5

        if score > best_score:
            best_score = score
            best_rows = rows

    return best_rows


def _detect_chart_type(step_text: str, query: str, has_fips: bool = False, has_latlon: bool = False) -> str:
    """Detect which chart type to create from plan step + query text."""
    combined = (step_text + " " + query).lower()
    # Check heatmap/choropleth BEFORE generic "map" to avoid false positive
    if any(w in combined for w in ["heatmap", "heat map", "choropleth"]):
        # Geographic heatmap if we have FIPS data, otherwise matrix heatmap
        if has_fips:
            return "choropleth"
        return "heatmap"
    if any(w in combined for w in ["map", "scatter", "geographic", "geo", "overlay"]):
        # If data has FIPS but no lat/lon, use choropleth
        if has_fips and not has_latlon:
            return "choropleth"
        return "scatter_map"
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
    fips_cols = []
    skip_cols = set()  # Columns that should NOT be used as value axes

    for c in cols:
        val = rows[0].get(c)
        cl = c.lower()
        if cl in ("lat", "latitude"):
            lat_cols.append(c)
            skip_cols.add(c)
        elif cl in ("lon", "lng", "longitude"):
            lon_cols.append(c)
            skip_cols.add(c)
        elif cl in ("fips", "fipscode", "fips_code", "county_fips", "geoid"):
            fips_cols.append(c)
            skip_cols.add(c)
        elif cl in ("year", "state_fips", "state_code", "id", "index"):
            # These are identifiers, not values — exclude from value columns
            skip_cols.add(c)
            if isinstance(val, (int, float)):
                pass  # Don't add to num_cols
            else:
                str_cols.append(c)
        elif isinstance(val, (int, float)):
            num_cols.append(c)
        else:
            str_cols.append(c)

    # Rank numeric columns by relevance to the step text
    def _rank_num_col(col_name: str) -> int:
        """Higher = more relevant as a chart value column."""
        cl = col_name.lower()
        score = 0
        step_lower = step_text.lower()

        # Direct mention in step text
        if cl in step_lower:
            score += 20

        # Priority columns for food security analysis
        priority_words = {
            "insec": 15, "insecur": 15, "foodinsec": 15,
            "risk": 14, "poverty": 13, "pov": 13,
            "score": 12, "rate": 11, "snap": 10, "desert": 10,
            "access": 9, "vulnerab": 9, "predict": 8, "compos": 8,
            "importance": 7, "shap": 7, "anomal": 7,
        }
        for word, pts in priority_words.items():
            if word in cl:
                score += pts
                break

        # Keyword match between step and column
        for kw in ["insec", "insecur", "risk", "poverty", "pov", "yield", "corn", "food", "vulnerab"]:
            if kw in step_lower and kw in cl:
                score += 10
                break

        return score

    # Deprioritize columns with no variance (same value for all rows)
    def _has_variance(col_name: str) -> bool:
        vals = {r.get(col_name) for r in rows[:20] if r.get(col_name) is not None}
        return len(vals) > 1

    def _final_rank(col_name: str) -> int:
        score = _rank_num_col(col_name)
        if not _has_variance(col_name):
            score -= 20  # Heavy penalty for uniform columns (e.g., state-level aggregates)
        return score

    num_cols.sort(key=_final_rank, reverse=True)

    if chart_type == "choropleth":
        fips = fips_cols[0] if fips_cols else None
        value = num_cols[0] if num_cols else None
        text = next((c for c in str_cols if c.lower() not in ("state", "fips")), str_cols[0] if str_cols else "")
        return {"fips_col": fips, "value_col": value, "text_col": text}

    if chart_type == "scatter_map":
        lat = lat_cols[0] if lat_cols else None
        lon = lon_cols[0] if lon_cols else None
        color = num_cols[0] if num_cols else None
        text = str_cols[0] if str_cols else None
        return {"lat_col": lat, "lon_col": lon, "color_col": color or "", "text_col": text or ""}

    if chart_type == "heatmap":
        # For heatmap, we need two categorical axes and a numeric z value
        x = str_cols[0] if str_cols else cols[0]
        y = str_cols[1] if len(str_cols) > 1 else cols[1] if len(cols) > 1 else cols[0]
        z = num_cols[0] if num_cols else cols[-1]
        # Avoid using the same column for x and y
        if x == y and len(cols) > 2:
            for c in cols:
                if c != x and c != z and c not in skip_cols:
                    y = c
                    break
        return {"x_col": x, "y_col": y, "z_col": z}

    if chart_type == "line":
        x = str_cols[0] if str_cols else cols[0]
        y = ",".join(num_cols[:3]) if num_cols else cols[1]
        return {"x_col": x, "y_cols": y}

    # bar chart (default)
    # Pick the best label column (county/name preferred)
    label_col = str_cols[0] if str_cols else cols[0]
    for c in str_cols:
        if c.lower() in ("county", "county_name", "name", "community"):
            label_col = c
            break

    # Best numeric column is already sorted by relevance
    best_y = num_cols[0] if num_cols else cols[1] if len(cols) > 1 else cols[0]

    return {"x_col": label_col, "y_col": best_y}


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

    # Extract data from previous tool results — picks the most relevant dataset
    rows = _extract_data_from_messages(state["messages"], step_text=step_text + " " + query)
    if not rows:
        # Fallback: query food atlas directly to get data for the chart
        try:
            from src.agent.tools.food_atlas import query_food_atlas
            # Extract state from query (default to MO)
            state_match = re.search(r"\b(Missouri|MO|Illinois|IL|Kansas|KS|Iowa|IA)\b", query, re.I)
            state_val = "MO"
            if state_match:
                abbrevs = {"missouri": "MO", "illinois": "IL", "kansas": "KS", "iowa": "IA"}
                state_val = abbrevs.get(state_match.group(1).lower(), state_match.group(1).upper())

            fallback_result = query_food_atlas.invoke({
                "state": state_val,
                "columns": "FIPS,County,FOODINSEC_21_23,POVRATE21,PCT_SNAP17,PCT_LACCESS_POP15",
                "limit": 200,
            })
            if isinstance(fallback_result, list) and fallback_result:
                rows = fallback_result
                # Add fallback data as a ToolMessage for downstream use
                fallback_msg = ToolMessage(
                    content=json.dumps(rows),
                    name="query_food_atlas",
                    tool_call_id=f"direct_viz_fallback_{step}",
                )
                state["messages"].append(fallback_msg)
        except Exception:
            pass

    if not rows:
        return {
            "reasoning_trace": state.get("reasoning_trace", []) + [
                "Direct viz: no data found in tool results — skipping chart"
            ],
        }

    # Detect data capabilities for smart chart type selection
    sample_keys = {k.lower() for k in rows[0].keys()} if rows else set()
    has_fips = bool(sample_keys & {"fips", "fipscode", "fips_code", "county_fips", "geoid"})
    has_latlon = bool(sample_keys & {"lat", "latitude"}) and bool(sample_keys & {"lon", "lng", "longitude"})

    # If user wants a geographic viz but data lacks FIPS, enrich from food_atlas
    combined_text = (step_text + " " + query).lower()
    wants_geo = any(w in combined_text for w in ["map", "geographic", "geo", "choropleth", "overlay"])
    if wants_geo and not has_fips and not has_latlon:
        # Try to match county names and add FIPS codes
        county_col = next((c for c in rows[0].keys() if c.lower() in ("county", "name", "county_name")), None)
        state_col = next((c for c in rows[0].keys() if c.lower() in ("state", "state_name")), None)
        if county_col:
            try:
                from src.agent.tools.food_atlas import query_food_atlas
                state_val = rows[0].get(state_col, "MO") if state_col else "MO"
                fips_lookup = query_food_atlas.invoke({"state": state_val, "columns": "FIPS,County", "limit": 500})
                fips_map = {r["County"].lower(): r["FIPS"] for r in fips_lookup if "County" in r and "FIPS" in r}
                for row in rows:
                    county = row.get(county_col, "").lower().replace(" county", "")
                    if county in fips_map:
                        row["FIPS"] = fips_map[county]
                has_fips = any("FIPS" in r for r in rows)
            except Exception:
                pass  # Best effort — fall back to non-geographic chart

    chart_type = _detect_chart_type(step_text, query, has_fips=has_fips, has_latlon=has_latlon)
    col_args = _detect_columns(rows, chart_type, step_text)

    # Sort and limit for readability
    sort_col = col_args.get("y_col") or col_args.get("value_col")
    if chart_type in ("bar",) and sort_col:
        try:
            rows = sorted(rows, key=lambda r: float(r.get(sort_col, 0)), reverse=True)[:15]
        except (ValueError, TypeError):
            rows = rows[:15]
    elif chart_type == "choropleth":
        pass  # Keep all rows for geographic maps

    # Generate a human-readable title
    from src.agent.nodes.chart_validator import _humanize_column
    raw_title = step_text.replace("[viz]", "").strip()
    y_label = col_args.get("y_col", col_args.get("value_col", col_args.get("color_col", "")))
    y_human = _humanize_column(y_label) if y_label else ""

    if not raw_title or raw_title == query:
        title = f"Top Counties by {y_human}" if y_human else "AgriFlow Visualization"
    else:
        # Check if raw title has useful info beyond just "create bar chart of..."
        generic_prefixes = ["create ", "show ", "generate ", "make ", "build "]
        cleaned = raw_title
        for prefix in generic_prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
        # Capitalize first letter
        title = cleaned[0].upper() + cleaned[1:] if cleaned else f"Top Counties by {y_human}"

    data_json = json.dumps(rows)

    # Call the chart tool directly
    try:
        if chart_type == "choropleth" and col_args.get("fips_col") and col_args.get("value_col"):
            result = create_choropleth_map.invoke({
                "title": title,
                "data_json": data_json,
                "fips_col": col_args["fips_col"],
                "value_col": col_args["value_col"],
                "text_col": col_args.get("text_col", ""),
            })
        elif chart_type == "scatter_map" and col_args.get("lat_col") and col_args.get("lon_col"):
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
        "choropleth": "create_choropleth_map",
        "scatter": "create_chart",
        "pie": "create_chart",
        "histogram": "create_chart",
        "box": "create_chart",
        "violin": "create_chart",
        "area": "create_chart",
        "funnel": "create_chart",
        "treemap": "create_chart",
        "sunburst": "create_chart",
        "waterfall": "create_chart",
        "indicator": "create_chart",
        "bubble": "create_chart",
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
    graph.add_node("chart_validator", _chart_validator_node)
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
            "synthesize": "chart_validator",
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

    # direct_viz -> check if more steps or validate charts
    graph.add_conditional_edges(
        "direct_viz",
        _should_continue_after_action,
        {"advance_step": "advance_step", "chart_validator": "chart_validator"},
    )

    graph.add_edge("tools", "tool_caller")
    graph.add_edge("chart_validator", "synthesizer")
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
