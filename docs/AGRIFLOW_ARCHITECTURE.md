# AgriFlow Architecture Guide

AgriFlow is an agentic AI system for food distribution planning in Missouri. It combines a LangGraph multi-step reasoning pipeline with Archia cloud agents, an MCP tool layer, a FastAPI backend, and a React frontend.

---

## System Overview

```
User Query
    |
    v
React Frontend (port 5173)
    |  SSE stream
    v
FastAPI Backend (port 8000)
    |  /api/query/stream
    v
LangGraph StateGraph (9 nodes)
    |
    +-- router (pure Python, no LLM)
    |       |
    |   single-category? --> tool_caller --> tools --> chart_validator --> finalizer
    |   multi-step?      --> planner --> tool_caller --> ... --> synthesizer
    |
    v
Tool Layer (16 analytics tools + viz + route tools)
    |
    +-- USDA Food Environment Atlas (SQLite)
    +-- USDA Food Access Atlas      (SQLite)
    +-- USDA NASS Quick Stats API
    +-- FEMA Disaster Declarations  (SQLite)
    +-- US Census ACS               (SQLite)
    +-- Open-Meteo Weather API
    +-- ML Engine (sklearn, XGBoost, SHAP)
    +-- Chart Generator (Plotly)
    +-- Route Optimizer
    +-- Web Search (DuckDuckGo)
```

---

## LangGraph StateGraph (9 Nodes)

### Graph Topology

```
START
  |
  v
[router]  ---- regex match (no LLM) ----
  |                                      |
  | multi-category query                 | single-category query
  v                                      |
[planner]                                |
  |                                      |
  v                                      |
[tool_caller] <--------------------------+
  |
  +-- has tool_calls? ----------> [tools] --> back to tool_caller
  |
  +-- more plan steps? ---------> [advance_step]
  |                                      |
  |                          [viz] step? | other step?
  |                               |      |
  |                        [direct_viz]  |
  |                               |      |
  |                               +------+
  |                                      |
  +-- substantive answer? -----> [finalizer] --> END
  |
  v
[chart_validator]  (pure Python, no LLM)
  |
  v
[synthesizer] --> END
```

### Node Descriptions

| Node | Role | LLM? |
| ---- | ---- | ---- |
| `router` | Regex-based query classification — skips planner for simple queries | No |
| `planner` | Decomposes complex queries into categorized plan steps `[data]`, `[viz]`, `[analytics]`, etc. | Yes (Haiku) |
| `tool_caller` | Executes the current plan step by calling tools | Yes (Sonnet) |
| `tools` | LangGraph ToolNode — runs all 16+ registered tools | No (direct call) |
| `advance_step` | Increments `current_step` counter for multi-step plans | No |
| `direct_viz` | Programmatically calls chart tools for `[viz]` steps without LLM | No |
| `chart_validator` | Fixes Plotly specs: swapped axes, raw column names, bad FIPS, empty traces | No |
| `synthesizer` | Synthesizes tool results into a final markdown answer | Yes (Sonnet) |
| `finalizer` | Skips synthesizer when tool_caller already produced a complete answer | No |

### AgriFlowState Schema

```python
{
    "messages":       list[BaseMessage],   # Full conversation history
    "plan":           list[str],           # ["[data] ...", "[viz] ...", ...]
    "current_step":   int,                 # Which plan step we're on
    "tool_results":   dict[str, Any],      # Accumulated tool outputs
    "reasoning_trace": list[str],          # Debugging log of node decisions
    "charts":         list[dict],          # Extracted chart specs
    "final_answer":   str | None,          # Final synthesized response
}
```

### Smart Router Categories

The router uses `re.compile` patterns to classify queries without any LLM call:

| Category | Patterns |
| -------- | -------- |
| `data` | food insecurity, poverty, SNAP, food desert, census, FEMA |
| `sql` | list tables, schema, run sql, query database |
| `viz` | chart, graph, heatmap, map, visualize, dashboard |
| `route` | route, deliver, schedule, logistics, dispatch |
| `weather` | weather, forecast, rain, drought, precipitation |
| `ingest` | load, import, ingest, EDA, add dataset |
| `analytics` | train model, XGBoost, Random Forest, SHAP, risk model |
| `ml` | predict, scenario, evaluation, compare scenario |

Single-category queries skip the planner entirely. Viz queries get a 2-step plan: `[data] + [viz]`.

---

## LLM Configuration

| Role | Model | Model ID |
| ---- | ----- | -------- |
| Complex analysis (tool_caller, synthesizer) | Claude Sonnet 4.5 | `priv-claude-sonnet-4-5-20250929` |
| Light tasks (planner) | Claude Haiku 4.5 | `priv-claude-haiku-4-5-20251001` |

**ArchiaChatModel** (`src/agent/llm.py`): A custom `BaseChatModel` that wraps Archia's Responses API (not Chat Completions). System prompts are injected as a prefix on the first user message (the "developer" role causes hanging). Tool calls use `input_schema` format.

---

## Data Sources

| Source | Data | Access Method |
| ------ | ---- | ------------- |
| USDA Food Environment Atlas | County-level food insecurity, SNAP, poverty, obesity, store counts | SQLite (`food_environment` table) |
| USDA Food Access Atlas | Census-tract LILA classifications, vehicle access, demographics | SQLite (`food_access` table) |
| USDA NASS Quick Stats | County crop production, yields, acreage | REST API |
| FEMA Disaster Declarations | Historical floods, droughts, storms by county | SQLite (`fema_disasters` table) |
| US Census ACS | Demographics, income, unemployment | SQLite (`acs_data` table) |
| Open-Meteo | Current weather and forecasts | REST API |
| DuckDuckGo | Agricultural threats, pest/disease alerts | Web search API |

**Database**: `data/agriflow.db` (SQLite). Path configurable via `DB_PATH` env var.

---

## Tool Inventory (16 Analytics + Viz + Route)

### Analytics Tools (registered in `tool_executor.py`)

| Tool | Description |
| ---- | ----------- |
| `query_food_atlas` | County-level food insecurity, SNAP, poverty from USDA Atlas |
| `query_food_access` | Census-tract LILA food desert classifications |
| `query_nass_crops` | Crop yields, production, acreage from USDA NASS API |
| `query_fema_disasters` | Historical disaster declarations by county |
| `query_census_acs` | Demographics, income, unemployment from Census ACS |
| `get_weather_forecast` | Current conditions and forecast from Open-Meteo |
| `list_tables` | List SQLite database tables |
| `describe_table` | Get column schema for a table |
| `execute_sql` | Run arbitrary SQL on the local database |
| `build_feature_matrix` | County-level ML feature matrix (38 cols, Suyog + NEW1 methodology) |
| `build_tract_feature_matrix` | Census-tract feature matrix with EHI/SII/AI indices (NEW2 methodology) |
| `train_risk_model` | Train county risk model (GBM/RF/SVR/LinearReg/XGBoost) |
| `predict_risk` | Generate county-level risk predictions from trained model |
| `get_feature_importance` | SHAP feature importance for trained model |
| `detect_anomalies` | Isolation Forest anomaly detection on county data |
| `compute_food_insecurity_risk` | Census-tract composite risk score (NEW2 formula) |
| `web_search_risks` | DuckDuckGo search for agricultural threats |

### Visualization Tools

| Tool | Chart Types |
| ---- | ----------- |
| `create_bar_chart` | Horizontal/vertical bar charts for rankings |
| `create_line_chart` | Time series and trend charts |
| `create_scatter_map` | Geographic scatter maps (lat/lon) |
| `create_risk_heatmap` | County-vs-factor risk matrices |
| `create_choropleth_map` | County-level geographic heatmaps (FIPS codes) |
| `create_chart` | Universal Plotly chart (scatter, pie, histogram, box, violin, area, funnel, treemap, sunburst, waterfall, indicator) |

### Route Tools

| Tool | Description |
| ---- | ----------- |
| `optimize_delivery_route` | TSP route optimization between Missouri distribution points |
| `create_route_map` | Visualize delivery route on a map |
| `schedule_deliveries` | Time-based delivery schedule from a route |

---

## ML Analytics Pipeline

### County-Level (Suyog / NEW1 2 Methodology)

Validated against Suyog.ipynb and NEW1 2.ipynb notebooks.

**1. Feature Matrix** (`build_feature_matrix`)
- 115 Missouri counties
- 38 features: food insecurity rates, poverty, SNAP participation, WIC, food access, store counts, interaction features

**2. Interaction Features** (auto-computed in `build_feature_matrix`)
- `POVxSNAP = POVRATE21 * PCT_SNAP22 / 100`
- `POVxLACCESS = POVRATE21 * PCT_LACCESS_HHNV19 / 100`
- `FOODxSNAP = FOODINSEC_21_23 * PCT_SNAP22 / 100`
- `SNAPxLACCESS = PCT_SNAP22 * PCT_LACCESS_HHNV19 / 100`

**3. K-Means Clustering** (k=3, StandardScaler)
- Cluster 0: Medium-risk (POVRATE21 ~16.7%)
- Cluster 1: Low-risk (POVRATE21 ~12.6%)
- Cluster 2: High-risk (POVRATE21 ~19.1%)

**4. Model Types** (selectable via `model_type` parameter)

| Model | Hyperparameters | Notebook R² |
| ----- | --------------- | ----------- |
| `gradient_boosting` | n=300, lr=0.05, depth=3 | R²=0.998, RMSE=0.099 |
| `linear_regression` | baseline | R²=0.983, RMSE=0.328, MAE=0.049 |
| `random_forest` | n=200, depth=15, n_jobs=-1 | R²=0.987 |
| `svr` | RBF kernel, C=10, eps=0.1 | R²=0.912 |
| `xgboost` | n=100, depth=5, lr=0.1 | R²>0.98 |

Note: SVR uses a `Pipeline([StandardScaler, SVR])` internally — do not pre-scale features.

**5. Composite Risk Score** (Suyog 6-Feature Method)

```
Features: FOODINSEC_18_20, FOODINSEC_21_23, POVRATE21,
          PCT_SNAP22, PCT_WICWOMEN16, PCT_LACCESS_HHNV19

MinMaxNorm each feature to [0, 1]
Composite_Risk_Score = mean(normalized features)
Risk_Category = percentile cut at 33rd/66th (Low / Medium / High)
```

**Top Predictive Features** (from SHAP analysis):
1. `PCT_WICWOMEN16` (~65%): WIC women participation — strongest predictor
2. `VLFOODSEC_21_23` (~22%): Historical extreme food insecurity
3. `PCT_SNAP22` (~top 5): Direct economic hardship
4. Interaction terms: `POVxSNAP`, `POVxLACCESS`, `FOODxSNAP`, `SNAPxLACCESS`

### Census-Tract Level (NEW2 Methodology)

Validated against NEW2.ipynb notebook. 72,242 tracts nationally, 1,387 for Missouri.

**Feature Engineering** (`build_tract_feature_matrix`)
- Per-capita normalization: `SNAP_rate = TractSNAP / Pop2010`
- Economic Hardship Index (EHI): `SNAP_rate + HUNV_rate`
- Structural Inequality Index (SII): `Black_pct + Hispanic_pct - White_pct`
- Aging Index (AI): `Senior_pct`
- LILA binary features: Urban, LILATracts_1And10, LILATracts_halfAnd10, LILATracts_1And20, LILATracts_Vehicle
- Filters: Pop2010 >= 50

**Composite Food Insecurity Risk** (`compute_food_insecurity_risk`)
```
GBM(n=500, lr=0.05, depth=4) → Predicted_SNAP_rate  (R²=0.9949)
EHI_norm       = MinMaxScale(Economic_Hardship_Index)
Pred_SNAP_norm = MinMaxScale(Predicted_SNAP_rate)
Food_Insecurity_Risk = (EHI_norm + Pred_SNAP_norm) / 2  ∈ [0, 1]
```

**Top High-Risk Missouri Tracts** (from validation run):
1. Tract 29510111300 — St. Louis city — risk=0.959
2. Tract 29510125700 — St. Louis city — risk=0.945
3. Tract 29095005801 — Jackson County  — risk=0.939

---

## Archia Cloud Agents (9 Agents)

Archia is the cloud agent platform. Agents receive queries via the Responses API and can call MCP tools registered to them.

**Base URL**: `https://registry.archia.app`
**Auth**: `Authorization: Bearer $ARCHIA_TOKEN`

| Agent | Model | Description |
| ----- | ----- | ----------- |
| `agriflow-system` | Sonnet 4.5 | Primary analyst with ODBC SQL access |
| `agriflow-analytics` | Sonnet 4.5 | ML prediction, SHAP, anomaly detection, analytics pipelines |
| `agriflow-viz` | Haiku 4.5 | Plotly chart and map generation |
| `agriflow-data` | Haiku 4.5 | Fast data fetcher (USDA, Census, FEMA, weather) |
| `agriflow-logistics` | Haiku 4.5 | Route optimization and delivery scheduling |
| `agriflow-planner` | Haiku 4.5 | Query decomposition into sub-tasks |
| `agriflow-ingest` | Haiku 4.5 | Data ingestion, profiling, and EDA |
| `agriflow-evaluator` | Haiku 4.5 | Response quality evaluation and improvement |
| `agriflow-ml` | *(disabled)* | Merged into agriflow-analytics |

**Agent Config Files**: `archia/agents/*.toml`
**Agent Prompts**: `archia/prompts/*.md`

**Sync prompts to Archia**:
```bash
export ARCHIA_TOKEN=your_token
bash push_agents.sh
bash push_agents.sh --dry-run  # preview without making API calls
```

---

## MCP Servers (4 Servers)

MCP (Model Context Protocol) servers expose local tools to Archia cloud agents via stdio.

| Server | File | Tools Exposed |
| ------ | ---- | ------------- |
| SQLite | `src/mcp_servers/sqlite_server.py` | list_tables, describe_table, get_schema, execute_sql |
| Chart | `src/mcp_servers/chart_server.py` | create_bar_chart, create_line_chart, create_scatter_map, create_risk_heatmap, create_choropleth_map, create_chart |
| Route | `src/mcp_servers/route_server.py` | optimize_delivery_route, create_route_map, schedule_deliveries |
| ML | `src/mcp_servers/ml_server.py` | train_and_predict, train_risk_model, predict_risk, explain_predictions, detect_anomalies, search_agricultural_risks, build_features, build_tract_features, compute_tract_risk |

**Run MCP server standalone** (for Cursor/Claude Desktop):
```bash
python -m src.mcp_servers.ml_server
```

---

## FastAPI Backend

**File**: `src/api/main.py`
**Port**: 8000
**Start**: `uvicorn src.api.main:app --reload --port 8000`

### Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/query/stream` | POST | SSE streaming query execution |
| `/api/query` | POST | Synchronous query (non-streaming) |
| `/api/health` | GET | Health check |
| `/api/evaluations` | GET | Retrieve evaluation log |

### SSE Streaming Protocol

The `/api/query/stream` endpoint emits these event types:

```
event: status      data: {"status": "thinking", "message": "..."}
event: reasoning   data: {"reasoning": [...]}
event: chart       data: {"chart_id": "...", "plotly_spec": {...}}
event: complete    data: {"answer": "...", "charts": [...]}
event: error       data: {"error": "..."}
```

### Response Cache

Identical queries (normalized) are cached for 10 minutes to avoid redundant LLM and tool calls.

---

## React Frontend

**Directory**: `frontend/`
**Port**: 5173 (Vite dev), built to `frontend/dist/`
**Start**: `npm run dev` from `frontend/`
**Build**: `npx vite build` from `frontend/`

### Key Features

- SSE streaming display (reasoning trace + final answer appear progressively)
- Collapsible markdown sections for long responses
- Plotly chart rendering (charts received via SSE)
- Dark theme with CSS variables

**Vite proxy** (`frontend/vite.config.js`): Proxies `/api` to `http://localhost:8000` in development.

---

## Environment Setup

### Required

```bash
# Archia API token (for cloud agents + push_agents.sh)
ARCHIA_TOKEN=your_token_here
```

### Optional

```bash
# Database path (default: data/agriflow.db)
DB_PATH=data/agriflow.db

# Model cache directory (default: models/)
MODEL_DIR=models/

# USDA NASS API key (optional — some endpoints work without it)
NASS_API_KEY=your_key
```

### `.env` File

Create a `.env` file in the project root. It is loaded automatically by FastAPI at startup:

```
ARCHIA_TOKEN=your_token
DB_PATH=data/agriflow.db
```

---

## File Structure

```
muidsi-hackathon-2026/
├── src/
│   ├── agent/
│   │   ├── graph.py              # LangGraph StateGraph (9 nodes)
│   │   ├── llm.py                # ArchiaChatModel (Responses API wrapper)
│   │   ├── state.py              # AgriFlowState TypedDict
│   │   ├── nodes/
│   │   │   ├── planner.py        # Query decomposition
│   │   │   ├── tool_executor.py  # tool_caller_node, ALL_TOOLS registry
│   │   │   ├── synthesizer.py    # Final answer synthesis
│   │   │   ├── chart_validator.py # Pure Python chart spec fixer
│   │   │   └── analytics_supervisor.py  # run_analytics_pipeline
│   │   ├── tools/
│   │   │   ├── ml_engine.py      # All ML tools (county + tract)
│   │   │   ├── chart_generator.py # Plotly chart tools
│   │   │   ├── food_atlas.py     # USDA Food Atlas queries
│   │   │   ├── food_access.py    # USDA Food Access Atlas
│   │   │   ├── nass.py           # USDA NASS API
│   │   │   ├── fema.py           # FEMA disaster data
│   │   │   ├── weather.py        # Open-Meteo weather
│   │   │   ├── route_optimizer.py # Route optimization
│   │   │   └── evaluator.py      # Response quality scoring
│   │   └── prompts/
│   │       └── system.py         # 4 prompts: AGRIFLOW_SYSTEM, PLANNER, SYNTHESIZER, RESPONDER
│   ├── api/
│   │   └── main.py               # FastAPI + SSE streaming
│   └── mcp_servers/
│       ├── sqlite_server.py      # SQLite MCP server
│       ├── chart_server.py       # Chart MCP server
│       ├── route_server.py       # Route MCP server
│       └── ml_server.py          # ML MCP server (8 tools)
├── archia/
│   ├── agents/                   # Agent TOML configs (name, model, prompt_file)
│   └── prompts/                  # Agent system prompt Markdown files
├── frontend/                     # React + Vite frontend
├── data/
│   └── agriflow.db               # SQLite database
├── models/                       # Trained model pkl files (auto-created)
├── notebooks/                    # Research notebooks (Suyog, NEW1, NEW2)
├── docs/                         # Documentation
├── scripts/
│   └── test_ml_pipeline.py       # ML pipeline validation script
└── push_agents.sh                # Archia agent prompt sync script
```

---

## Testing

### ML Pipeline Validation

Run the Suyog/NEW2 methodology validation against local data:

```bash
cd muidsi-hackathon-2026
python scripts/test_ml_pipeline.py

# Options:
python scripts/test_ml_pipeline.py --model gradient_boosting --state MO
python scripts/test_ml_pipeline.py --model random_forest --state IL
```

**What it validates**:
1. County feature matrix loads (115 Missouri counties, 38 features)
2. 4 interaction features created (POVxSNAP, POVxLACCESS, FOODxSNAP, SNAPxLACCESS)
3. K-Means forms exactly 3 clusters
4. County model R² exceeds threshold (GBM >= 0.95, LinReg >= 0.90)
5. Risk prediction scores are in valid range
6. 6-feature composite risk score (Suyog) is in [0, 1]
7. Risk categories include Low / Medium / High
8. Census-tract pipeline loads 1,387+ MO tracts
9. Tract risk scores are in [0, 1]

**Note on county food insecurity columns**: `FOODINSEC_21_23` and `PCT_SNAP22` are state-level aggregates in the USDA Atlas (same value for all counties in a state). The test script auto-selects `POVRATE21` (county-level) as the model target when the preferred column has zero variance.

### MCP Tool Verification

```bash
python -c "from src.mcp_servers.ml_server import mcp; print([t.name for t in mcp.list_tools()])"
# Expected: ['train_and_predict', 'train_risk_model', 'predict_risk',
#            'explain_predictions', 'detect_anomalies', 'search_agricultural_risks',
#            'build_features', 'build_tract_features', 'compute_tract_risk']
```

### Archia Agent Sync

```bash
export ARCHIA_TOKEN=your_token
bash push_agents.sh --dry-run  # preview
bash push_agents.sh             # sync all enabled agents
```

### Full System Test

```bash
# 1. Start backend
uvicorn src.api.main:app --reload --port 8000

# 2. Start frontend (separate terminal)
cd frontend && npm run dev

# 3. Test via curl
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which Missouri counties have the highest food insecurity rates?"}'
```

---

## Known Limitations

1. **State-level food insecurity data**: `FOODINSEC_21_23` is a state-level aggregate in the USDA Food Atlas — all counties in Missouri share the same value (12.7%). County-level variation comes from `POVRATE21`, `PCT_LACCESS_HHNV19`, and `PCT_WICWOMEN16`.

2. **ARCHIA_TOKEN required**: Cloud agents and `push_agents.sh` require the token. The local LangGraph agent works without it.

3. **Feature importance caching**: `get_feature_importance` requires a model to have been trained in the same session (models/ directory). Call `train_risk_model` first.

4. **Windows console encoding**: Some Unicode characters in terminal output may be replaced on Windows. Functionality is not affected.

5. **Web search dependency**: `web_search_risks` uses DuckDuckGo Instant Answers API. Rate-limited for high-frequency use.
