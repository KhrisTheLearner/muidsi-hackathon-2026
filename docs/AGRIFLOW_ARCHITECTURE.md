# AgriFlow System Architecture

AgriFlow is a multi-agent AI system for food supply chain intelligence. This document describes the complete system design: the LangGraph orchestration graph, LLM routing strategy, tool inventory, Archia cloud agent configuration, MCP server layer, FastAPI backend, and React frontend.

---

## System Overview

```
User (Browser, port 5173)
         |
         | HTTP / SSE stream
         v
FastAPI Backend (port 8000)
  /api/query/stream  -- Server-Sent Events (progressive rendering)
  /api/query         -- Synchronous JSON response
  /api/route         -- Direct route optimization
  /api/health        -- System status
         |
         v
LangGraph StateGraph  [src/agent/graph.py]
  9 nodes, pure Python edges, no LLM cost for routing
         |
         v
Tool Layer  [src/agent/tools/]
  30+ tools across 7 categories
         |
    +----+----+----+----+----+----+
    |    |    |    |    |    |    |
  USDA NASS FEMA ACS WX DDG  ML  Chart Route
         |
Archia Cloud Agents (8 agents)
  MCP servers expose tools to cloud agents
```

---

## LangGraph StateGraph (9 Nodes)

### Graph Topology

```
START
  |
  v
[router]  --- pure Python regex, no LLM ---
  |                                       |
  | multi-category query                  | single-category query
  v                                       |
[planner]  (Haiku)                        |
  |                                       |
  v                                       v
[tool_caller] <--------------------------+
  |
  +-- has tool_calls? --> [tools]  (ToolNode)
  |                           |
  |                      back to tool_caller
  |
  +-- more plan steps? --> [advance_step] --> [tool_caller]
  |
  +-- direct viz step? --> [direct_viz] --> [advance_step]
  |
  +-- substantive answer? --> [finalizer] --> END
  |
  v
[chart_validator]  (pure Python, no LLM)
  |
  v
[synthesizer]  (Sonnet)
  |
  v
END
```

### State Definition (`src/agent/state.py`)

```python
class AgriFlowState(TypedDict):
    messages: list           # Conversation history (HumanMessage, AIMessage, ToolMessage)
    plan: list[str]          # Decomposed plan steps (e.g. ["[data] ...", "[viz] ..."])
    current_step: int        # Index into plan
    tool_results: dict       # Accumulated tool outputs by tool name
    reasoning_trace: list    # Human-readable reasoning log
    charts: list             # Extracted Plotly chart specs
    final_answer: str | None # Synthesized answer string
```

### Node Descriptions

| Node | Type | Model | Purpose |
|------|------|-------|---------|
| `router` | Pure Python | None | Regex-classifies query into 8 categories. Single-category: fast-track. Multi-category: send to planner. |
| `planner` | LLM | Haiku | Decomposes query into categorized plan steps: `[data]`, `[viz]`, `[analytics]`, `[ml]`, `[route]`, `[sql]`, `[ingest]`. 45s timeout with fallback. |
| `tool_caller` | LLM + Synthetic | Sonnet/Haiku | Executes the current plan step. Routes to specialized agent (Haiku for data/viz/route, Sonnet for analytics/sql). Uses **synthetic tool call builder** to bypass LLM timeout. |
| `tools` | ToolNode | None | LangGraph prebuilt ToolNode. Executes tool calls from `tool_caller`, appends ToolMessages to state. |
| `advance_step` | Pure Python | None | Increments `current_step` counter. Routes to `direct_viz` for `[viz]` steps or back to `tool_caller`. |
| `direct_viz` | Pure Python | None | Programmatically calls chart tools without LLM. Extracts chart parameters from tool results in state. More reliable than LLM-based chart generation. |
| `chart_validator` | Pure Python | None | Validates and fixes Plotly specs: swapped axes, raw column names, bad FIPS codes, empty traces. Zero LLM cost. |
| `synthesizer` | LLM | Sonnet | Synthesizes all tool results and reasoning into a final markdown answer with citations. |
| `finalizer` | Pure Python | None | Skips synthesizer when `tool_caller` already produced a complete answer (e.g. simple data lookup). |

---

## Smart Router

The router (`_router_node` in `graph.py`) uses compiled regex patterns — no LLM, no latency:

```python
_DIRECT_PATTERNS = {
    "data":      r"(food insecur|poverty|snap|food desert|census|fema|disaster|"
                 r"disease|pest|outbreak|news|alert|web search|nass|crop report)",
    "sql":       r"(list tables|show tables|schema|run sql|query database)",
    "viz":       r"(chart|graph|plot|heatmap|scatter.?map|map|visualiz|dashboard)",
    "route":     r"(route|deliver|schedule|logistics|transport|dispatch)",
    "weather":   r"(weather|forecast|temperature|rain|drought|precipitation)",
    "ingest":    r"(load|import|ingest|profile|eda|add.*dataset|download.*data)",
    "analytics": r"(train.*model|xgboost|random forest|shap|feature importan|"
                 r"anomal|analytics|risk model|predict.*risk)",
    "ml":        r"(predict|scenario|evaluation|compare.*scenario)",
}
```

**If exactly one category matches**: skip planner, build a 1-step plan `[{cat}] {query}` — saves 1-3 s.

**If viz matches alone**: build a 2-step plan `[data] ... / [viz] ...` — ensures data is fetched first.

**If multiple match**: send to planner for full decomposition.

---

## Multi-Agent Routing (tool_executor.py)

Each plan step category uses a different model + minimal tool set:

| Category | Model | Tool Count | Purpose |
|----------|-------|------------|---------|
| `data` | Haiku (DATA_MODEL) | 9 | USDA, Census, FEMA, weather, web search |
| `viz` | Haiku (VIZ_MODEL) | 15 | Charts + data tools for self-sufficiency |
| `route` | Haiku (LOGISTICS_MODEL) | 4 | Delivery route optimization |
| `ml` | Sonnet (ML_MODEL) | 4 | Model evaluation only |
| `analytics` | Sonnet (ML_MODEL) | 12 | Full ML pipeline |
| `sql` | Sonnet (SQL_MODEL) | 2 | Database queries |
| `ingest` | Haiku (DATA_MODEL) | 5 | Dataset loading/profiling |
| `general` | Sonnet (SQL_MODEL) | all | Fallback for unclassified steps |

### Synthetic Tool Call Builder

The Archia API can time out when tools are bound to LLM requests. For tagged steps (`[data]`, `[analytics]`, `[route]`, `[sql]`), AgriFlow uses a **synthetic tool call builder** that bypasses the LLM entirely:

1. Parse keywords from the plan step text
2. Construct tool call dicts (name, args, id) directly from keyword rules
3. Return an `AIMessage` with `tool_calls` — same interface as LLM-generated calls

Example: a step containing "disease" or "pest" automatically generates a `search_agricultural_news` call. A step containing "corn" or "yield" generates a `query_nass` call.

This makes data and analytics steps deterministic and ~3x faster than LLM-based tool selection.

---

## Tool Inventory

### [data] — Data Retrieval (9 tools)

| Tool | Source | Free? |
|------|--------|-------|
| `query_food_atlas` | USDA Food Environment Atlas (SQLite) | Yes |
| `query_food_access` | USDA Food Access Research Atlas (SQLite) | Yes |
| `query_nass` | USDA NASS Quick Stats API | API key |
| `query_weather` | Open-Meteo API | Yes |
| `query_fema_disasters` | FEMA Disaster Declarations API | Yes |
| `query_census_acs` | US Census ACS API | Yes |
| `run_prediction` | Local heuristic/ML model | Yes |
| `search_web` | DuckDuckGo DDGS | Yes |
| `search_agricultural_news` | DuckDuckGo DDGS (ag-context) | Yes |

### [analytics] — ML Pipeline (12 tools)

| Tool | Purpose |
|------|---------|
| `build_feature_matrix` | Merge food atlas + census + derived features (115 MO counties x 38 features) |
| `build_tract_feature_matrix` | Build per-capita normalized tract features (72k+ tracts, EHI/SII/AI) |
| `train_risk_model` | Train XGBoost/GBM/RF on county data. Returns R2, RMSE, MAE, CCC, AUC + 3 charts |
| `predict_risk` | Load cached model, run inference under scenarios (drought, price_shock, baseline) |
| `get_feature_importance` | SHAP TreeExplainer global + per-sample feature importance |
| `detect_anomalies` | Isolation Forest outlier detection on county indicators |
| `compute_food_insecurity_risk` | End-to-end tract risk scorer (EHI + GBM SNAP prediction) |
| `run_analytics_pipeline` | Full pipeline: EDA + training in parallel. Supports full_analysis, quick_predict, risk_scan |
| `run_eda_pipeline` | Automated EDA: descriptive stats, distributions, correlations, outlier detection + charts |
| `web_search_risks` | DuckDuckGo DDGS search for agricultural threats (upgraded from instant-answer API) |
| `train_crop_model` | Crop dependency risk model (delegates to train_risk_model with crop features) |
| `predict_crop_yield` | Crop yield impact under scenarios |

### [viz] — Visualization (6 tools)

| Tool | Chart Types |
|------|-------------|
| `create_bar_chart` | Horizontal/vertical bar, rankings |
| `create_line_chart` | Time series, trends |
| `create_scatter_map` | Geographic scatter (lat/lon) |
| `create_risk_heatmap` | County x factor risk matrices |
| `create_choropleth_map` | County-level geographic heatmap (FIPS-based) |
| `create_chart` | Universal: scatter, pie, histogram, box, violin, area, funnel, treemap, sunburst, waterfall, indicator, roc_curve, actual_vs_predicted, feature_importance, correlation_matrix |

### [route] — Logistics (4 tools)

`optimize_delivery_route` (TSP), `calculate_distance` (haversine + ORS), `create_route_map` (Plotly map), `schedule_deliveries`

### [sql] — Database (2 tools)

`list_tables`, `run_sql_query` (read-only, SELECT/WITH/PRAGMA only)

### [ml] — Evaluation (4 tools)

`compute_evaluation_metrics` (RMSE, MAE, R2, CCC, F1), `compare_scenarios`, `compute_ccc`, `explain_with_shap`

### [ingest] — Data Ingestion (5 tools)

`list_db_tables`, `fetch_and_profile_csv`, `load_dataset`, `run_eda_query`, `drop_table`

---

## LLM Configuration (`src/agent/llm.py`)

### ArchiaChatModel

AgriFlow uses a custom `ArchiaChatModel` — a LangChain `BaseChatModel` subclass that wraps the **Archia Responses API** (not the Chat Completions API):

```
POST https://registry.archia.app/v1/responses
Authorization: Bearer {ARCHIA_TOKEN}
{
  "model": "priv-claude-sonnet-4-5-20250929",
  "input": [...messages...],
  "tools": [...tool schemas...]
}
```

This enables Archia's cloud routing, logging, and agent management while still using Claude models.

### Model Assignment

| Role | Model ID | Used For |
|------|----------|----------|
| `SONNET` | `priv-claude-sonnet-4-5-20250929` | Complex reasoning, analytics, synthesis |
| `HAIKU` | `priv-claude-haiku-4-5-20251001` | Fast tasks: data fetch, chart gen, routing |
| `DATA_MODEL` | Haiku | Data retrieval steps |
| `VIZ_MODEL` | Haiku | Chart generation steps |
| `LOGISTICS_MODEL` | Haiku | Route optimization steps |
| `ML_MODEL` | Sonnet | ML training and analytics |
| `SQL_MODEL` | Sonnet | SQL queries and general fallback |

### Timeout Protection

All LLM calls are wrapped with a 30-60 second timeout using `threading.Thread` + `queue.Queue`. On timeout, the node returns an empty `AIMessage("")` and the synthetic tool call builder generates the required tool calls.

---

## Archia Cloud Agents

8 agents are configured in `archia/agents/*.toml` and deployed via `push_agents.sh`:

| Agent | TOML | Model | MCP Tools |
|-------|------|-------|-----------|
| `agriflow-system` | agriflow-system.toml | Sonnet | sqlite, ml, charts, routes, data |
| `agriflow-analytics` | agriflow-analytics.toml | Sonnet | ml, sqlite |
| `agriflow-data` | agriflow-data.toml | Haiku | sqlite, data |
| `agriflow-viz` | agriflow-viz.toml | Haiku | charts, sqlite |
| `agriflow-logistics` | agriflow-logistics.toml | Haiku | routes |
| `agriflow-planner` | agriflow-planner.toml | Haiku | (none — text only) |
| `agriflow-ingest` | agriflow-ingest.toml | Haiku | sqlite |
| `agriflow-evaluator` | agriflow-evaluator.toml | Haiku | (none — text only) |

Each agent has a corresponding system prompt in `archia/prompts/{name}.md` that describes its available tools, workflow, and rules.

---

## MCP Servers (`src/mcp_servers/`)

Five FastMCP servers expose AgriFlow tools to Archia cloud agents and any MCP client (Claude Desktop, Cursor):

| Server | Archia Name | Tools | Run Command |
|--------|-------------|-------|-------------|
| `ml_server.py` | `agriflow-ml` | 9 ML tools | `python -m src.mcp_servers.ml_server` |
| `chart_server.py` | `agriflow-charts` | 6 chart tools | `python -m src.mcp_servers.chart_server` |
| `sqlite_server.py` | `agriflow-sqlite` | 2 SQL tools | `python -m src.mcp_servers.sqlite_server` |
| `route_server.py` | `agriflow-routes` | 4 route tools | `python -m src.mcp_servers.route_server` |
| `data_server.py` | `agriflow-data` | 8 data + search tools | `python -m src.mcp_servers.data_server` |

MCP servers are thin wrappers — all business logic lives in `src/agent/tools/` (single source of truth).

---

## FastAPI Backend (`src/api/main.py`)

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query/stream` | SSE streaming — emits `status`, `reasoning`, `tool_end`, `answer` events |
| POST | `/api/query` | Synchronous JSON response with 10-min response cache |
| POST | `/api/route` | Direct route optimization (bypasses LangGraph, ~3s) |
| GET | `/api/health` | Model, API key status, tool inventory |
| GET | `/api/charts` | List session Plotly charts |
| GET | `/api/charts/{id}` | Retrieve chart by ID |
| GET | `/api/eval/summary` | Heuristic quality scores across recent queries |
| GET | `/api/examples` | Example queries for the frontend |

### SSE Streaming Design

The streaming endpoint runs the LangGraph agent in a background thread (not async) to get the full final state (including chart ToolMessages). A keepalive ping is sent every 5 seconds to prevent SSE connection timeout:

```
data: {"type": "status", "message": "Analyzing query..."}
: keepalive
: keepalive
data: {"type": "reasoning", "step": "Router: fast-track [data] (skipped planner)"}
data: {"type": "tool_end", "tool": "query_food_atlas"}
data: {"type": "answer", "data": {...full response...}}
```

### Response Cache

A 10-minute in-memory cache (keyed by MD5 of normalized query) returns instant responses for repeated identical queries. Cache misses trigger full LangGraph execution.

### Chart Extraction

After agent execution, `_extract_charts()` scans all ToolMessages for chart-tool results and extracts Plotly specs. Charts are stored in a session-scoped `_chart_store` dict and embedded in the response `charts` array.

---

## React Frontend (`frontend/src/`)

### Architecture

Single-page app with 4 tabs (Query, Dashboard, Data, Map):

- **Query tab**: SSE streaming display with collapsible reasoning trace, tool badges, markdown answer, inline charts
- **Dashboard tab**: Grid of all session charts
- **Data tab**: Raw tool results as formatted JSON
- **Map tab**: Route visualization (reserved for logistics queries)

### Key Components

**PlotlyChart**: Uses `ResizeObserver` to call `Plotly.Plots.resize()` on container size change. This fixes 0-width charts in hidden tabs (Plotly renders at 0 width when `display:none`; tab activation changes width, ResizeObserver catches it).

**SSE Client** (`api.js`): EventSource connection to `/api/query/stream`. Accumulates `reasoning` events for progressive display and renders `answer` event as the final response.

### Vite Proxy

`vite.config.js` proxies `/api` requests to `http://localhost:8000` during development, so the frontend can call the backend without CORS issues:

```javascript
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', changeOrigin: true }
  }
}
```

---

## Chart Validator (`src/agent/nodes/chart_validator.py`)

Pure Python node — no LLM cost. Fixes common Plotly spec issues from LLM-generated charts:

| Problem | Fix |
|---------|-----|
| Swapped x/y axes | Detect if x contains county names when it should be numeric; swap |
| Raw column names | Map FIPS, County, POVRATE21 etc. to display-friendly labels |
| Bad FIPS codes | Zero-pad to 5 digits (e.g. "1001" -> "01001") |
| Empty traces | Remove traces with no data; return error if all empty |
| Missing layout | Add dark theme defaults: paper_bgcolor, plot_bgcolor, font color |

---

## Data Flow: Example Query

Query: *"Which Missouri counties have the highest poverty and what's their food insecurity?"*

```
1. Router
   - Matches "poverty" and "food insecur" -> category: [data]
   - Single category -> fast-track, skip planner
   - Plan: ["[data] Which Missouri counties have the highest poverty and what's their food insecurity?"]

2. tool_caller (step 1: [data])
   - Synthetic tool call builder detects "poverty" + "food insecur"
   - Builds: query_food_atlas(state="MO", limit=200)
   - Returns AIMessage(tool_calls=[{name:"query_food_atlas", args:{...}}])

3. tools (ToolNode)
   - Executes query_food_atlas
   - Queries SQLite: SELECT FIPS, County, POVRATE21, FOODINSEC_21_23 FROM food_environment WHERE State='MO'
   - Returns ToolMessage with 115 rows as JSON

4. tool_caller (coming_from_tools=True)
   - Plan complete (step 0 of 1 done)
   - Returns AIMessage("") to trigger finalizer

5. finalizer
   - tool_msg_count > 0 AND final_answer is None -> goes to chart_validator

6. chart_validator
   - No chart specs in this response -> pass through

7. synthesizer
   - Reads ToolMessage with 115 county rows
   - Identifies top counties by POVRATE21
   - Generates markdown answer with data table and citations
   - Sets final_answer

8. FastAPI
   - Emits SSE "answer" event with full response
   - Frontend renders markdown + county table
```

---

## Deployment Modes

### 1. Local Python (Development)

```bash
python run_agent.py
```

Runs LangGraph directly, prints reasoning trace and answer to terminal.

### 2. FastAPI + React (Full Stack)

```bash
# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
# Open http://localhost:5173
```

### 3. Archia Cloud

```bash
# Deploy agents and tools
bash push_agents.sh

# Query via Archia console
# https://console.archia.app
```

### 4. Hybrid (Recommended for Hackathon)

Archia console for the web UI + local FastAPI + LangGraph for tool execution. Best of both: collaborative web interface + full ML/data access.

---

## Security Notes

- `.env` is git-ignored — never commit API keys
- `ARCHIA_TOKEN` and `NASS_API_KEY` must be set before running
- SQLite is read-only for query tools (only `run_sql_query` with SELECT/WITH/PRAGMA allowed)
- CORS is open (`allow_origins=["*"]`) — restrict in production
- Ingest tools (`drop_table`, `load_dataset`) require explicit invocation, not triggered by normal queries

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent/graph.py` | ~200 | LangGraph 9-node StateGraph, router patterns, graph topology |
| `src/agent/llm.py` | ~200 | ArchiaChatModel, model constants, timeout wrappers |
| `src/agent/state.py` | ~30 | AgriFlowState TypedDict |
| `src/agent/nodes/planner.py` | ~100 | Query decomposition with timeout + fallback |
| `src/agent/nodes/tool_executor.py` | ~600 | Multi-agent routing, synthetic tool calls, LLM routing |
| `src/agent/nodes/synthesizer.py` | ~100 | Final answer synthesis |
| `src/agent/nodes/chart_validator.py` | ~150 | Pure Python Plotly spec fixing |
| `src/agent/nodes/analytics_supervisor.py` | ~200 | ML pipeline orchestration |
| `src/agent/prompts/system.py` | ~200 | 4 system prompts with visualization guide |
| `src/agent/tools/ml_engine.py` | ~600 | All ML tools (XGBoost, SHAP, anomaly, tract risk) |
| `src/agent/tools/chart_generator.py` | ~400 | All Plotly chart tools (12+ types) |
| `src/api/main.py` | ~440 | FastAPI with SSE, cache, chart extraction |
| `frontend/src/App.jsx` | ~800 | React UI with ResizeObserver, SSE streaming |
