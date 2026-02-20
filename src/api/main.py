"""AgriFlow FastAPI backend.

Serves the agent via a REST API that the React frontend can call.
Includes chart/map extraction from agent responses for frontend rendering.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

load_dotenv()

from src.agent.graph import create_agent  # noqa: E402
from src.agent.tools.evaluator import evaluate_response, get_evaluation_summary  # noqa: E402
from src.agent.tools.route_optimizer import (  # noqa: E402
    optimize_delivery_route, create_route_map, LOCATIONS,
)

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

# ---------------------------------------------------------------------------
# Response cache — avoids re-running identical queries (10-min TTL)
# ---------------------------------------------------------------------------
_response_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 600  # 10 minutes


def _cache_key(query: str) -> str:
    """Normalize query and hash for cache lookup."""
    normalized = query.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def _get_cached_response(query: str) -> dict | None:
    key = _cache_key(query)
    if key in _response_cache:
        ts, data = _response_cache[key]
        if time.monotonic() - ts < _CACHE_TTL:
            return data
        del _response_cache[key]
    return None


def _set_cached_response(query: str, data: dict) -> None:
    _response_cache[_cache_key(query)] = (time.monotonic(), data)


# In-memory chart store for the session
_chart_store: dict[str, dict] = {}

# Tool names that produce chart/map Plotly specs
_CHART_TOOLS = {
    "create_bar_chart", "create_line_chart", "create_scatter_map",
    "create_risk_heatmap", "create_route_map", "create_choropleth_map",
    "create_chart",
}

# Tools that return analytics reports with embedded charts
_ANALYTICS_TOOLS = {"run_analytics_pipeline", "train_risk_model"}

# Session-scoped analytics report store
_analytics_store: dict[str, dict] = {}


class RouteRequest(BaseModel):
    origin: str
    destinations: str  # comma-separated location names


class RouteResponse(BaseModel):
    optimized_route: list[str]
    total_distance_miles: float
    est_total_drive_minutes: int
    total_stops: int
    legs: list[dict]
    method: str
    plotly_spec: dict | None = None
    directions: list[dict] = []
    error: str | None = None


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

    # Check response cache — instant return for identical queries
    cached = _get_cached_response(query)
    if cached:
        cached["reasoning_trace"] = ["Cache: returning cached response (< 10 min old)"] + cached.get("reasoning_trace", [])
        return QueryResponse(**cached)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "current_step": 0,
        "tool_results": {},
        "reasoning_trace": [],
        "final_answer": None,
        "charts": [],
    }

    t0 = time.monotonic()
    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        return QueryResponse(
            answer=f"Agent error: {type(e).__name__}. Please try again or rephrase your question.",
            reasoning_trace=[f"Error: {e}"],
            tool_calls=[], charts=[],
        )
    duration_ms = int((time.monotonic() - t0) * 1000)

    tool_calls = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    charts = _extract_charts(result.get("messages", []))
    analytics_report = _extract_analytics(result.get("messages", []))
    final_answer = result.get("final_answer", "No answer generated.")

    # Auto-evaluate in background (zero LLM token cost — pure heuristics)
    try:
        evaluate_response.invoke({
            "query":          query,
            "answer":         final_answer,
            "tool_calls_json": json.dumps(tool_calls),
            "duration_ms":    duration_ms,
        })
    except Exception:
        pass  # Never let eval failure block the response

    # If analytics report has embedded charts, merge them
    if analytics_report and "charts" in analytics_report:
        charts.extend(analytics_report.get("charts", []))

    response_data = {
        "answer": final_answer,
        "reasoning_trace": result.get("reasoning_trace", []),
        "tool_calls": tool_calls,
        "charts": charts,
        "analytics_report": analytics_report,
    }

    # Cache successful responses for instant replay
    if final_answer and "error" not in final_answer.lower():
        _set_cached_response(query, response_data)

    return QueryResponse(**response_data)


@app.post("/api/query/stream")
async def query_agent_stream(request: QueryRequest):
    """Stream query progress via Server-Sent Events.

    Emits events: status, tool_start, tool_end, reasoning, answer.
    The frontend can progressively render each step as it arrives.
    """
    query = request.query.strip()
    if not query or len(query) > 2000:
        async def _err():
            msg = "Please provide a question." if not query else "Query too long."
            yield f"data: {json.dumps({'type': 'answer', 'data': {'answer': msg, 'reasoning_trace': [], 'tool_calls': [], 'charts': []}})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "current_step": 0,
        "tool_results": {},
        "reasoning_trace": [],
        "final_answer": None,
        "charts": [],
    }

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing query...'})}\n\n"

        try:
            # Run agent synchronously in a thread — this is the only reliable
            # way to get the full final state (including chart ToolMessages).
            # astream() state merging breaks in FastAPI's async generator context.
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing query...'})}\n\n"

            # Run agent in background thread + send keepalive pings every 5s
            # so the SSE connection doesn't time out on long queries.
            loop = asyncio.get_event_loop()
            agent_task = loop.run_in_executor(None, agent.invoke, initial_state)
            while True:
                done, _ = await asyncio.wait({agent_task}, timeout=5)
                if done:
                    result = agent_task.result()
                    break
                # Send a keepalive comment (SSE comments start with ':')
                yield ": keepalive\n\n"

            # Extract tool calls
            tool_calls = []
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({"tool": tc["name"], "args": tc["args"]})

            charts = _extract_charts(result.get("messages", []))
            analytics_report = _extract_analytics(result.get("messages", []))
            final_answer = result.get("final_answer", "No answer generated.")

            if analytics_report and "charts" in analytics_report:
                charts.extend(analytics_report.get("charts", []))

            # Emit reasoning steps individually for progressive rendering
            for step in result.get("reasoning_trace", []):
                yield f"data: {json.dumps({'type': 'reasoning', 'step': step})}\n\n"

            # Emit tool call events
            for tc in tool_calls:
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': tc['tool']})}\n\n"

            response_data = {
                "answer": final_answer,
                "reasoning_trace": result.get("reasoning_trace", []),
                "tool_calls": tool_calls,
                "charts": charts,
                "analytics_report": analytics_report,
            }

            yield f"data: {json.dumps({'type': 'answer', 'data': response_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'answer', 'data': {'answer': f'Agent error: {type(e).__name__}. Please try again.', 'reasoning_trace': [str(e)], 'tool_calls': [], 'charts': []}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


@app.get("/api/eval/summary")
async def eval_summary():
    """Return heuristic quality summary across recent AgriFlow responses."""
    try:
        return get_evaluation_summary.invoke({"last_n": 30})
    except Exception as e:
        return {"error": str(e), "total": 0}


@app.post("/api/route")
async def route_direct(request: RouteRequest):
    """Direct route endpoint — bypasses LangGraph for ~3s response.
    Returns ORS road routing with haversine fallback."""
    _empty = {"optimized_route": [], "total_distance_miles": 0,
              "est_total_drive_minutes": 0, "total_stops": 0,
              "legs": [], "method": "", "directions": [], "plotly_spec": None}

    origin = request.origin.strip()
    destinations = request.destinations.strip()

    if not origin or not destinations:
        return {**_empty, "error": "Origin and destinations required."}

    unknown = [loc for loc in [origin] + [d.strip() for d in destinations.split(",")]
               if loc not in LOCATIONS]
    if unknown:
        return {**_empty, "error": f"Unknown locations: {unknown}"}

    loop = asyncio.get_event_loop()
    route_data = await loop.run_in_executor(
        None,
        lambda: optimize_delivery_route.invoke({"origin": origin, "destinations": destinations})
    )
    if "error" in route_data:
        return {**_empty, "error": route_data["error"]}

    map_data = await loop.run_in_executor(
        None,
        lambda: create_route_map.invoke({"route_json": json.dumps(route_data)})
    )

    return {
        "optimized_route": route_data["optimized_route"],
        "total_distance_miles": route_data["total_distance_miles"],
        "est_total_drive_minutes": route_data["est_total_drive_minutes"],
        "total_stops": route_data["total_stops"],
        "legs": route_data["legs"],
        "method": route_data["method"],
        "plotly_spec": map_data.get("plotly_spec"),
        "directions": map_data.get("ors_steps", []),
        "error": None,
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": os.getenv("DEFAULT_MODEL", "priv-claude-sonnet-4-5-20250929"),
        "archia_configured": bool(os.getenv("ARCHIA_TOKEN")),
        "nass_configured": bool(os.getenv("NASS_API_KEY")),
        "ors_configured": bool(os.getenv("OPENROUTESERVICE_API_KEY")),
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
                    "create_scatter_map", "create_risk_heatmap",
                    "create_choropleth_map"],
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
