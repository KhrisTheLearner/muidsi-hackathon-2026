"""AgriFlow FastAPI backend.

Serves the agent via a REST API that the React frontend can call.
Includes chart/map extraction from agent responses for frontend rendering.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

load_dotenv()

from src.agent.graph import create_agent  # noqa: E402

app = FastAPI(
    title="AgriFlow API",
    description="Food Supply Chain Intelligence Agent",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = create_agent()

# In-memory chart store for the session
_chart_store: dict[str, dict] = {}

# Tool names that produce chart/map Plotly specs
_CHART_TOOLS = {
    "create_bar_chart", "create_line_chart", "create_scatter_map",
    "create_risk_heatmap", "create_route_map",
}

# Tools that return analytics reports with embedded charts
_ANALYTICS_TOOLS = {"run_analytics_pipeline"}

# Session-scoped analytics report store
_analytics_store: dict[str, dict] = {}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    reasoning_trace: list[str]
    tool_calls: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    analytics_report: dict[str, Any] | None = None


def _extract_charts(messages: list) -> list[dict[str, Any]]:
    """Extract Plotly chart specs from ToolMessage results."""
    charts = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if not hasattr(msg, "name") or msg.name not in _CHART_TOOLS:
            continue
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            if isinstance(data, dict) and "plotly_spec" in data:
                cid = data.get("chart_id", f"chart_{uuid.uuid4().hex[:8]}")
                _chart_store[cid] = data
                charts.append(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return charts


def _extract_analytics(messages: list) -> dict[str, Any] | None:
    """Extract analytics pipeline report from ToolMessage results."""
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        if not hasattr(msg, "name") or msg.name not in _ANALYTICS_TOOLS:
            continue
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            if isinstance(data, dict):
                rid = data.get("pipeline", f"analytics_{uuid.uuid4().hex[:8]}")
                _analytics_store[rid] = data
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return None


@app.get("/api/analytics")
async def list_analytics():
    """List all analytics reports from the current session."""
    return {"reports": list(_analytics_store.values())}


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Send a natural language query to AgriFlow and get a structured response."""
    query = request.query.strip()
    if not query:
        return QueryResponse(
            answer="Please provide a question.",
            reasoning_trace=[], tool_calls=[], charts=[],
        )
    if len(query) > 2000:
        return QueryResponse(
            answer="Query too long. Please keep questions under 2000 characters.",
            reasoning_trace=[], tool_calls=[], charts=[],
        )

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "current_step": 0,
        "tool_results": {},
        "reasoning_trace": [],
        "final_answer": None,
        "charts": [],
    }

    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        return QueryResponse(
            answer=f"Agent error: {type(e).__name__}. Please try again or rephrase your question.",
            reasoning_trace=[f"Error: {e}"],
            tool_calls=[], charts=[],
        )

    tool_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    charts = _extract_charts(result.get("messages", []))
    analytics_report = _extract_analytics(result.get("messages", []))

    # If analytics report has embedded charts, merge them
    if analytics_report and "charts" in analytics_report:
        charts.extend(analytics_report.get("charts", []))

    return QueryResponse(
        answer=result.get("final_answer", "No answer generated."),
        reasoning_trace=result.get("reasoning_trace", []),
        tool_calls=tool_calls,
        charts=charts,
        analytics_report=analytics_report,
    )


@app.get("/api/charts")
async def list_charts():
    """List all generated charts in the current session."""
    return {"charts": list(_chart_store.values())}


@app.get("/api/charts/{chart_id}")
async def get_chart(chart_id: str):
    """Retrieve a previously generated chart by ID."""
    if chart_id in _chart_store:
        return _chart_store[chart_id]
    return {"error": "Chart not found"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": os.getenv("DEFAULT_MODEL", "priv-claude-sonnet-4-5-20250929"),
        "archia_configured": bool(os.getenv("ARCHIA_TOKEN")),
        "nass_configured": bool(os.getenv("NASS_API_KEY")),
        "tools_available": {
            "data": ["query_food_atlas", "query_food_access", "query_nass",
                     "query_weather", "query_fema_disasters", "query_census_acs",
                     "run_prediction"],
            "sql": ["list_tables", "run_sql_query"],
            "ml": ["compute_evaluation_metrics", "compare_scenarios",
                   "compute_ccc", "explain_with_shap"],
            "analytics": ["run_analytics_pipeline", "build_feature_matrix",
                          "train_risk_model", "predict_risk",
                          "train_crop_model", "predict_crop_yield",
                          "get_feature_importance", "detect_anomalies",
                          "web_search_risks"],
            "viz": ["create_bar_chart", "create_line_chart",
                    "create_scatter_map", "create_risk_heatmap"],
            "route": ["optimize_delivery_route", "calculate_distance",
                      "create_route_map", "schedule_deliveries"],
        },
        "total_tools": 32,
    }


@app.get("/api/examples")
async def examples():
    """Return example queries for the frontend."""
    return {
        "examples": [
            {
                "title": "Food Insecurity Hotspots",
                "query": "Which Missouri counties have the highest food insecurity rates and what factors contribute to this?",
            },
            {
                "title": "Corn Dependency Risk",
                "query": "What are the top corn-producing counties in Missouri and how do their food insecurity rates compare?",
            },
            {
                "title": "Drought Scenario",
                "query": "If corn yields drop 20% due to drought, which Missouri communities would be most at risk and what should we do?",
            },
            {
                "title": "Food Desert Map",
                "query": "Show me a map of food desert risk across southeastern Missouri counties with risk scores as colors.",
            },
            {
                "title": "Delivery Route Planning",
                "query": "Plan an optimal delivery route from Cape Girardeau Hub to Wayne County, Pemiscot County, Dunklin County, and New Madrid County. Show the route on a map with a delivery schedule.",
            },
            {
                "title": "Risk Dashboard",
                "query": "Create a bar chart comparing food insecurity rates across the top 10 Missouri counties and a risk heatmap showing how different factors contribute to vulnerability.",
            },
            {
                "title": "Weather Impact",
                "query": "What is the current weather forecast for Wayne County Missouri and how might it affect crop production?",
            },
            {
                "title": "Disaster Response",
                "query": "If flooding hits SE Missouri, plan emergency food delivery routes from St. Louis Food Bank and Cape Girardeau Hub to the 5 most vulnerable counties. Show routes on a map with schedules.",
            },
            {
                "title": "ML Risk Analysis",
                "query": "Train an XGBoost risk model on Missouri county data and predict which counties are most vulnerable to a 30% corn yield drop. Show feature importance.",
            },
            {
                "title": "Full Analytics Pipeline",
                "query": "Run a full analytics report on Missouri food supply chain risk under a drought scenario. Include model training, predictions, SHAP explanations, anomaly detection, and web research on emerging threats.",
            },
            {
                "title": "Anomaly Detection",
                "query": "Which Missouri counties have unusual food insecurity patterns compared to their demographics? Flag anomalies and explain what makes them outliers.",
            },
        ]
    }
