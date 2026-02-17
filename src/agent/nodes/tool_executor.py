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

from langchain_core.messages import AIMessage, SystemMessage

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
    build_feature_matrix, detect_anomalies, get_feature_importance,
    predict_crop_yield, predict_risk, train_crop_model, train_risk_model,
    web_search_risks,
)
from src.agent.nodes.analytics_supervisor import run_analytics_pipeline
from src.agent.tools.sql_query import list_tables, run_sql_query
from src.agent.tools.weather import query_weather
from src.agent.tools.ingest import (
    list_db_tables, fetch_and_profile_csv, load_dataset,
    run_eda_query, drop_table,
)

# ---------------------------------------------------------------------------
# Specialized tool groups — each agent only sees what it needs
# ---------------------------------------------------------------------------
DATA_TOOLS = [
    query_food_atlas, query_food_access, query_nass, query_weather,
    query_fema_disasters, query_census_acs, run_prediction,
]

VIZ_TOOLS = [
    create_bar_chart, create_line_chart, create_scatter_map, create_risk_heatmap,
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
    build_feature_matrix, run_analytics_pipeline,
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
                      "build feature", "web search risk", "ccc", "concordance"],
        "data": ["food", "insecur", "atlas", "census", "fema", "disaster",
                 "weather", "nass", "crop", "yield", "poverty", "snap"],
        "viz":  ["chart", "bar", "line", "heatmap", "scatter", "map", "visual",
                 "graph", "plot", "dashboard"],
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

    plan_context = f"""
Current execution plan:
{plan_text}

Previously collected data:
{_summarize_tool_results(state.get('tool_results', {}))}

Execute the next needed tool calls to gather data for the current plan step.
If you already have all needed data, respond with your analysis (no tool calls).
"""

    messages = [
        SystemMessage(content=AGRIFLOW_SYSTEM + "\n\n" + plan_context),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    reasoning = state.get("reasoning_trace", [])
    reasoning.append(f"Router: step {step+1} -> [{cat_name}] ({n_tools} tools)")
    if response.tool_calls:
        for tc in response.tool_calls:
            reasoning.append(f"Tool call: {tc['name']}({tc['args']})")
    else:
        reasoning.append("Analysis: LLM responded without tool calls (sufficient data)")

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
