"""Tool executor node with multi-agent routing.

Routes plan steps to specialized agents with minimal tool sets.
Each agent only sees the tools it needs = fewer prompt tokens + faster responses.

Routing table:
  data      -> FAST_MODEL  + 7 data tools
  viz       -> FAST_MODEL  + 4 chart tools
  route     -> FAST_MODEL  + 4 route tools
  ml        -> ML_MODEL    + 3 prediction/eval tools
  analytics -> ML_MODEL    + 12 analytics tools (ML engine + eval + pipeline)
  sql       -> SQL_MODEL   + 2 SQL tools (+ Archia ODBC if agent)
  general   -> SQL_MODEL   + all tools (fallback)
"""

from __future__ import annotations

import queue
import re
import threading
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.llm import (
    DATA_MODEL, LOGISTICS_MODEL, ML_MODEL, SQL_MODEL, VIZ_MODEL, get_llm,
)
from src.agent.prompts.system import AGRIFLOW_SYSTEM
from src.agent.state import AgriFlowState

# ---------------------------------------------------------------------------
# Tool imports
# ---------------------------------------------------------------------------
from src.agent.tools.chart_generator import (
    create_bar_chart, create_line_chart, create_risk_heatmap, create_scatter_map,
    create_choropleth_map, create_chart,
)
from src.agent.tools.food_atlas import query_food_access, query_food_atlas
from src.agent.tools.nass_api import query_nass
from src.agent.tools.prediction import run_prediction
from src.agent.tools.route_optimizer import (
    calculate_distance, create_route_map, optimize_delivery_route, schedule_deliveries,
)
from src.agent.tools.census_acs import query_census_acs
from src.agent.tools.evaluation import (
    compare_scenarios, compute_ccc, compute_evaluation_metrics, explain_with_shap,
)
from src.agent.tools.fema_disasters import query_fema_disasters
from src.agent.tools.ml_engine import (
    build_feature_matrix, build_tract_feature_matrix, compute_food_insecurity_risk,
    detect_anomalies, get_feature_importance,
    predict_crop_yield, predict_risk, train_crop_model, train_risk_model,
    web_search_risks, run_eda_pipeline,
)
from src.agent.nodes.analytics_supervisor import run_analytics_pipeline
from src.agent.tools.sql_query import list_tables, run_sql_query
from src.agent.tools.weather import query_weather
from src.agent.tools.ingest import (
    list_db_tables, fetch_and_profile_csv, load_dataset,
    run_eda_query, drop_table,
)
from src.agent.tools.web_search import search_web, search_agricultural_news

# ---------------------------------------------------------------------------
# Specialized tool groups — each agent only sees what it needs
# ---------------------------------------------------------------------------
DATA_TOOLS = [
    query_food_atlas, query_food_access, query_nass, query_weather,
    query_fema_disasters, query_census_acs, run_prediction,
    search_web, search_agricultural_news,
]

VIZ_TOOLS = [
    create_bar_chart, create_line_chart, create_scatter_map, create_risk_heatmap, create_choropleth_map, create_chart,
    # Include data tools so viz agent can query data if not already in context
    query_food_atlas, query_food_access, query_nass, query_weather,
    query_fema_disasters, query_census_acs, run_prediction,
    search_web, search_agricultural_news,
]

ROUTE_TOOLS = [
    optimize_delivery_route, calculate_distance, create_route_map, schedule_deliveries,
]

ML_TOOLS = [
    run_prediction, compute_evaluation_metrics, compare_scenarios, compute_ccc,
]

ANALYTICS_TOOLS = [
    train_risk_model, predict_risk, train_crop_model, predict_crop_yield,
    get_feature_importance, detect_anomalies, web_search_risks,
    build_feature_matrix, build_tract_feature_matrix, compute_food_insecurity_risk,
    run_analytics_pipeline, run_eda_pipeline,
    compute_evaluation_metrics, compare_scenarios, compute_ccc, explain_with_shap,
]

SQL_TOOLS = [
    list_tables, run_sql_query,
]

INGEST_TOOLS = [
    list_db_tables, fetch_and_profile_csv, load_dataset,
    run_eda_query, drop_table,
]

# Full set (fallback for unclassified steps)
ALL_TOOLS = (
    DATA_TOOLS + SQL_TOOLS + ML_TOOLS + ANALYTICS_TOOLS + VIZ_TOOLS + ROUTE_TOOLS
    + INGEST_TOOLS
)
# Deduplicate (some tools appear in multiple groups)
ALL_TOOLS = list({t.name: t for t in ALL_TOOLS}.values())

# ---------------------------------------------------------------------------
# Category -> (model, tools) routing map
# ---------------------------------------------------------------------------
_ROUTE_MAP: dict[str, tuple[str, list]] = {
    "data":      (DATA_MODEL,      DATA_TOOLS),
    "viz":       (VIZ_MODEL,       VIZ_TOOLS),
    "route":     (LOGISTICS_MODEL, ROUTE_TOOLS),
    "ml":        (ML_MODEL,        ML_TOOLS),
    "analytics": (ML_MODEL,        ANALYTICS_TOOLS),
    "sql":       (SQL_MODEL,       SQL_TOOLS),
    "ingest":    (DATA_MODEL,      INGEST_TOOLS),  # Haiku — fast, no heavy reasoning
}

# Tool name -> category (for auto-detection from plan)
_TOOL_CATEGORY: dict[str, str] = {}
for _cat, (_, _tools) in _ROUTE_MAP.items():
    for _t in _tools:
        _TOOL_CATEGORY[_t.name] = _cat


def _detect_category(plan: list[str], step: int) -> str:
    """Detect the category for the current plan step.

    Checks for explicit category tags (e.g. "[data]") or tool name matches.
    Falls back to 'general' which uses all tools.
    """
    if step >= len(plan):
        return "general"

    step_text = plan[step].lower()

    # Check for explicit category tags from planner: [data], [viz], etc.
    for cat in _ROUTE_MAP:
        if f"[{cat}]" in step_text:
            return cat

    # Check for tool name mentions
    for tool_name, cat in _TOOL_CATEGORY.items():
        if tool_name in step_text:
            return cat

    # Keyword heuristics
    keywords: dict[str, list[str]] = {
        "analytics": ["train", "xgboost", "random forest", "feature importan",
                      "anomal", "shap", "analytics", "deep analysis", "pipeline",
                      "build feature", "web search risk", "ccc", "concordance",
                      "eda", "exploratory", "distribution", "correlation",
                      "gradient boost", "composite risk", "vulnerability"],
        "data": ["food", "insecur", "atlas", "census", "fema", "disaster",
                 "weather", "nass", "crop", "yield", "poverty", "snap",
                 "disease", "pest", "outbreak", "news", "alert", "search",
                 "tar spot", "aphid", "rootworm", "avian flu", "corn yield",
                 "soybean yield"],
        "viz":  ["chart", "bar", "line", "heatmap", "scatter", "map", "visual",
                 "graph", "plot", "dashboard", "pie", "histogram", "box plot",
                 "violin", "area chart", "funnel", "treemap", "sunburst",
                 "waterfall", "gauge", "bubble"],
        "route": ["route", "deliver", "schedule", "distance", "logistics",
                  "transport", "dispatch"],
        "ml":   ["predict", "risk", "scenario", "evaluat", "metric", "compare",
                 "rmse", "f1", "accuracy"],
        "sql":  ["sql", "query", "table", "database", "schema"],
        "ingest": ["ingest", "load dataset", "fetch dataset", "download data",
                   "profile", "eda", "new dataset", "add to database", "import csv",
                   "load csv", "load excel", "list tables", "list db"],
    }
    for cat, kws in keywords.items():
        if any(kw in step_text for kw in kws):
            return cat

    return "general"


def _get_routed_llm(category: str):
    """Get an LLM bound with only the tools needed for this category."""
    if category in _ROUTE_MAP:
        model, tools = _ROUTE_MAP[category]
    else:
        model, tools = SQL_MODEL, ALL_TOOLS

    llm = get_llm(model=model, temperature=0.0)
    return llm.bind_tools(tools), category, len(tools)


def get_tool_executor_llm():
    """Get an LLM with all tools bound (legacy/fallback)."""
    llm = get_llm(model=SQL_MODEL, temperature=0.0)
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Synthetic tool call builder — when LLM refuses, we construct calls ourselves
# ---------------------------------------------------------------------------
_STATE_ABBREVS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def _extract_state_from_text(text: str) -> str | None:
    """Extract a US state abbreviation from step/query text."""
    # Check for 2-letter codes already
    m = re.search(r'\b([A-Z]{2})\b', text)
    if m and m.group(1) in _STATE_ABBREVS.values():
        return m.group(1)
    # Check for full state names
    text_lower = text.lower()
    for name, abbrev in _STATE_ABBREVS.items():
        if name in text_lower:
            return abbrev
    return None


def _build_synthetic_tool_calls(step_text: str, category: str, query: str) -> list[dict]:
    """Build synthetic tool calls from a plan step when the LLM refuses.

    Returns a list of tool_call dicts (name, args, id) based on keyword analysis.
    Prioritizes step_text keywords; falls back to query keywords only for state
    extraction and when step_text has no specific keywords.
    """
    calls = []
    step_lower = step_text.lower()
    query_lower = query.lower()
    state = _extract_state_from_text(step_text) or _extract_state_from_text(query)

    if category == "data" or "[data]" in step_lower:
        # Match keywords from the STEP text first (more specific), then query
        def _step_has(*kws: str) -> bool:
            return any(kw in step_lower for kw in kws)

        def _any_has(*kws: str) -> bool:
            return any(kw in step_lower or kw in query_lower for kw in kws)

        if _step_has("food insecur", "food access", "poverty", "snap",
                     "food desert", "atlas", "laccess", "hunger",
                     "food environment", "risk"):
            args: dict = {"limit": 200}
            if state:
                args["state"] = state
            calls.append({
                "name": "query_food_atlas",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("census", "demograph", "population", "income", "median"):
            args = {}
            if state:
                args["state_fips"] = state
            args["variables"] = "B01003_001E,B19013_001E,B17001_002E"
            calls.append({
                "name": "query_census_acs",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("weather", "drought", "rain", "temperature", "precipitation"):
            args = {}
            if state:
                args["state"] = state
            calls.append({
                "name": "query_weather",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("crop", "yield", "corn", "soybean", "wheat", "nass",
                     "harvest", "production"):
            args = {"source": "SURVEY"}
            if state:
                args["state_alpha"] = state
            # Detect specific crop from step or query
            for crop in ["CORN", "SOYBEANS", "WHEAT", "RICE", "COTTON"]:
                if crop.lower() in step_lower or crop.lower() in query_lower:
                    args["commodity_desc"] = crop
                    break
            calls.append({
                "name": "query_nass",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("fema", "disaster", "flood", "tornado", "hurricane", "storm"):
            args = {}
            if state:
                args["state"] = state
            calls.append({
                "name": "query_fema_disasters",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("disease", "pest", "outbreak", "news", "alert", "threat",
                     "tar spot", "aphid", "rootworm", "avian flu", "search web",
                     "web search", "search agricultural"):
            # Extract the most specific topic from step/query text
            topic = "crop disease pest outbreak"
            for kw in ["tar spot", "gray leaf spot", "soybean rust", "sudden death",
                        "aphid", "rootworm", "fall armyworm", "avian flu", "swine fever",
                        "drought", "disease", "pest", "outbreak"]:
                if kw in step_lower or kw in query_lower:
                    topic = kw
                    break
            args = {"topic": topic}
            if state:
                args["region"] = state
            calls.append({
                "name": "search_agricultural_news",
                "args": args,
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        if _step_has("search web", "web search") and not _step_has(
                "disease", "pest", "outbreak", "news", "alert"):
            # Plain web search (not agriculture-specific)
            topic = query[:120] if query else step_text[:120]
            calls.append({
                "name": "search_web",
                "args": {"query": topic},
                "id": f"synth_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        # Fallback: if no step-specific keywords matched, use query keywords
        if not calls:
            if _any_has("food insecur", "poverty", "risk", "hunger"):
                args = {"limit": 200}
                if state:
                    args["state"] = state
                calls.append({
                    "name": "query_food_atlas",
                    "args": args,
                    "id": f"synth_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })
            else:
                # Ultimate fallback: food atlas
                args = {"limit": 200}
                if state:
                    args["state"] = state
                calls.append({
                    "name": "query_food_atlas",
                    "args": args,
                    "id": f"synth_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })

    elif category == "analytics" or "[analytics]" in step_text.lower():
        args: dict = {}
        if state:
            args["state"] = state
        # Extract model_type from step text (check multi-word names first)
        for _mt in ("gradient_boosting", "random_forest", "linear_regression", "xgboost", "svr"):
            if _mt.replace("_", " ") in step_text.lower() or _mt in step_text.lower():
                args["model_type"] = _mt
                break
        # Extract scenario
        for _sc in ("drought", "price_shock", "baseline"):
            if _sc in step_text.lower():
                args["scenario"] = _sc
                break
        calls.append({
            "name": "run_analytics_pipeline",
            "args": args,
            "id": f"synth_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })

    elif category == "route" or "[route]" in step_text.lower():
        # Extract origin and destinations from step/query text.
        # step_text arrives lowercased; use query for original casing where needed.
        origin = "Cape Girardeau Hub"  # sensible MO default
        destinations = ""

        # Try to find "from <place>" pattern (case-insensitive on lowercase text)
        from_match = re.search(r'\bfrom\s+([\w\s]+?)(?:\s+to\b|\s+and\b|$)', step_text, re.I)
        if from_match:
            origin = from_match.group(1).strip().title()

        # Collect destination mentions: "to X, Y, and Z" → comma-separated list
        to_match = re.search(r'\bto\s+(.+?)(?:\.|$)', step_text, re.I)
        if to_match:
            raw_dest = to_match.group(1).strip()
            # Remove Oxford comma before "and", then replace " and " with ", "
            raw_dest = re.sub(r',\s+and\s+', ', ', raw_dest, flags=re.I)
            raw_dest = re.sub(r'\s+and\s+', ', ', raw_dest, flags=re.I)
            # Collapse any double-commas and strip trailing punctuation
            raw_dest = re.sub(r',\s*,+', ',', raw_dest).strip(", ")
            # Title-case each destination to match route tool's location registry
            destinations = ", ".join(p.strip().title() for p in raw_dest.split(",") if p.strip())

        args: dict = {"origin": origin}
        if destinations:
            args["destinations"] = destinations

        calls.append({
            "name": "optimize_delivery_route",
            "args": args,
            "id": f"synth_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })

    elif category == "sql" or "[sql]" in step_text.lower():
        # For SQL steps, first list tables to orient the LLM, then query
        calls.append({
            "name": "list_tables",
            "args": {},
            "id": f"synth_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })

    return calls


def tool_caller_node(state: AgriFlowState) -> dict:
    """Route the current plan step to the appropriate specialized agent.

    Each category uses a different model + tool subset for efficiency.
    """
    plan = state.get("plan", [])
    step = state.get("current_step", 0)

    # Detect category and get the right LLM + tools
    category = _detect_category(plan, step)
    llm_with_tools, cat_name, n_tools = _get_routed_llm(category)

    # Build context
    plan_text = "\n".join(
        f"{'>' if i == step else ' '} {i+1}. {s}"
        for i, s in enumerate(plan)
    )

    # Count how many real ToolMessages we have (to detect data-starved plans)
    tool_msg_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))

    # Force tool calls on ALL [data]/[analytics] tagged steps.
    # The synthetic fallback will construct tool calls if the LLM ignores this.
    force_tools = ""
    step_text_lower = plan[step].lower() if step < len(plan) else ""
    if "[data]" in step_text_lower:
        force_tools = (
            "\n\nCRITICAL: This is a [data] step. You MUST call at least one data "
            "tool (query_food_atlas, query_food_access, query_nass, query_weather, "
            "query_fema_disasters, query_census_acs, search_web, "
            "search_agricultural_news). Do NOT respond with text only. "
            "The downstream visualization steps DEPEND on real tool data being in "
            "the message history. If you skip this, the charts will have no data. "
            "For disease/pest/outbreak/news queries, call search_agricultural_news."
        )
    elif "[analytics]" in step_text_lower:
        force_tools = (
            "\n\nCRITICAL: This is an [analytics] step. You MUST call "
            "run_analytics_pipeline (preferred — returns R², RMSE, CCC, AUC, charts) "
            "or train_risk_model + predict_risk. Do NOT skip with a text-only response."
        )
    elif "[ml]" in step_text_lower:
        force_tools = (
            "\n\nCRITICAL: This is an [ml] step. You MUST call prediction/evaluation "
            "tools (run_prediction, compute_evaluation_metrics, compare_scenarios). "
            "Do NOT skip with a text-only response."
        )
    elif "[route]" in step_text_lower:
        force_tools = (
            "\n\nCRITICAL: This is a [route] step. You MUST call routing tools "
            "(optimize_delivery_route, calculate_distance, create_route_map, "
            "schedule_deliveries). Do NOT skip with a text-only response."
        )
    elif "[sql]" in step_text_lower:
        force_tools = (
            "\n\nCRITICAL: This is a [sql] step. You MUST call SQL tools "
            "(list_tables, run_sql_query). Do NOT skip with a text-only response."
        )

    plan_context = f"""
Current execution plan:
{plan_text}

Previously collected data:
{_summarize_tool_results(state.get('tool_results', {}))}
Tool messages collected so far: {tool_msg_count}

INSTRUCTIONS:
1. Execute the tool call(s) specified in the CURRENT plan step (marked with >).
2. You MUST call tools for [data], [viz], [route], [analytics], and [ml] steps.
   NEVER skip these with a text-only response. The data you collect here is used
   by downstream visualization and analysis steps.
3. For [data] steps: ALWAYS call the appropriate query tool, even if you think
   you know the answer. The visualization steps need REAL data from tool results,
   not your training knowledge. Call query_food_atlas for food insecurity,
   query_census_acs for demographics, query_nass for crop data, etc.
4. CRITICAL: If a previous tool returned an error, you MUST self-correct and retry:
   - If a location/county was not found, try alternate names (city->county, abbreviations).
   - If an API returned no data, try different parameters (broader date range, different format).
   - If a column was not found, call list_tables or query with fewer columns.
   - NEVER report a tool error back to the user as your final answer.
   - ALWAYS attempt at least one retry with corrected parameters before giving up.
5. Missouri city-to-county examples: Columbia=Boone, Springfield=Greene, Jefferson City=Cole,
   Joplin=Jasper, St. Joseph=Buchanan, Rolla=Phelps, Sedalia=Pettis.
6. DIRECT ANSWERS: Only for steps that are explicitly about general knowledge
   (NOT tagged [data], [viz], [analytics], [ml], [route]). If the step has a
   category tag, you MUST call tools.
7. NEVER DEFLECT: If tools return limited or no data, combine whatever data
   you have with your agricultural expertise to give a COMPLETE answer.
{force_tools}"""

    reasoning = state.get("reasoning_trace", [])
    reasoning.append(f"Router: step {step+1} -> [{cat_name}] ({n_tools} tools)")

    # --- SYNTHETIC TOOL CALLS for tagged steps ---
    # The Archia LLM API hangs when tools are bound to the request.
    # For tagged steps ([data], [analytics], [ml]), we skip the LLM and build
    # tool calls directly from the step text keywords. This is faster and more
    # reliable than waiting for an LLM that may timeout.
    user_query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
    )

    # Check if we're returning from tool execution (last message is a ToolMessage).
    # If so, don't re-trigger synthetic tool calls — tools already ran for this step.
    last_msg = state["messages"][-1] if state["messages"] else None
    coming_from_tools = isinstance(last_msg, ToolMessage)

    needs_tools = bool(force_tools)
    if needs_tools and coming_from_tools:
        # Tool results are already in history — produce empty response to trigger
        # advance_step (if more plan steps remain) or synthesizer.
        response = AIMessage(content="")
        reasoning.append(f"Synthetic: [{cat_name}] step {step+1} complete — tool results collected")
    elif needs_tools:
        # Skip LLM entirely — Archia hangs with tool binding.
        # Build synthetic tool calls directly from step keywords.
        reasoning.append(
            f"Synthetic: [{cat_name}] step {step+1} — "
            "building tool calls from step keywords"
        )
        synthetic_calls = _build_synthetic_tool_calls(
            step_text_lower, category, user_query
        )
        if synthetic_calls:
            response = AIMessage(content="", tool_calls=synthetic_calls)
            for tc in synthetic_calls:
                reasoning.append(f"Synthetic tool call: {tc['name']}({tc['args']})")
        else:
            response = AIMessage(content="")
            reasoning.append("No synthetic calls could be built — proceeding with text")
    else:  # not needs_tools
        # Non-tagged step: use LLM text-only (no tools bound) for knowledge answers.
        # Use a no-tools LLM to avoid Archia tool binding timeouts.
        llm_no_tools = get_llm(model=SQL_MODEL, temperature=0.0)
        messages = [
            SystemMessage(content=AGRIFLOW_SYSTEM + "\n\n" + plan_context),
            *state["messages"],
        ]
        LLM_TIMEOUT = 30  # seconds
        _result_q: queue.Queue = queue.Queue()

        def _llm_worker():
            try:
                _result_q.put(("ok", llm_no_tools.invoke(messages)))
            except Exception as exc:
                _result_q.put(("err", exc))

        t = threading.Thread(target=_llm_worker, daemon=True, name="llm_text")
        t.start()
        try:
            status, value = _result_q.get(timeout=LLM_TIMEOUT)
            response = value if status == "ok" else AIMessage(content="")
        except queue.Empty:
            response = AIMessage(content="")

        if response.tool_calls:
            for tc in response.tool_calls:
                reasoning.append(f"Tool call: {tc['name']}({tc['args']})")
        else:
            reasoning.append("Analysis: LLM provided text answer")

    return {
        "messages": [response],
        "reasoning_trace": reasoning,
    }


def _summarize_tool_results(results: dict) -> str:
    """Create a brief summary of collected tool results for the LLM context."""
    if not results:
        return "None yet."

    lines = []
    for tool_name, data in results.items():
        if isinstance(data, list):
            lines.append(f"- {tool_name}: {len(data)} rows returned")
        elif isinstance(data, dict) and "error" in data:
            lines.append(f"- {tool_name}: ERROR - {data['error']}")
        else:
            lines.append(f"- {tool_name}: data collected")
    return "\n".join(lines)
