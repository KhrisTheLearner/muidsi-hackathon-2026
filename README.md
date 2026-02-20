# AgriFlow — Food Supply Chain Intelligence
---
**MUIDSI Hackathon 2026** | Agriculture/Plant Track Team: Pierce (AI/ML Lead), Alfiya (Data Pipeline + EDA), Suyog (Feature Engineering + ML), Christophe (Problem Framing + Pitch)

**AgriFlow** is a multi-agent AI system built for the **MUIDSI Hackathon 2026** that helps food distribution planners address agricultural risk and food insecurity across Missouri. It combines LangGraph agentic orchestration, machine learning, real-time data from federal APIs, and an interactive React dashboard into a single conversational intelligence platform.

> Ask AgriFlow anything about Missouri food supply chains — it reasons across crop data, census demographics, disaster history, weather, and trained ML models to give actionable answers with charts, maps, and delivery routes.

### Working Demos: 

[AgriFlow Live Demo](https://agriflow-team02.netlify.app)
[AgriFlow U.S. Food Environment Atlas Dashboard](https://rshiny.dsa.missouri.edu/students/sspnp6/ShinyApps/DATA-SCI-8310/Hackathon/)
 
---

## What It Does

AgriFlow answers natural-language questions like:

- *"Which Missouri counties have the highest food insecurity and what drives it?"*
- *"Train an XGBoost risk model and show feature importance for a 30% corn yield drop"*
- *"Optimize a delivery route from Cape Girardeau Hub to Wayne, Pemiscot, and Dunklin counties"*
- *"Are there any current crop disease outbreaks in Missouri I should know about?"*
- *"Show a choropleth map of food desert risk across southeastern Missouri"*
- *"Run a full analytics pipeline under a drought scenario with anomaly detection and web research"*

---

## Architecture Overview

```
User (React, port 5173)
       |  SSE streaming
       v
FastAPI Backend (port 8000)
       |
       v
LangGraph StateGraph (9 nodes)
       |
  [router] --regex--> single-category --> [tool_caller] --> [finalizer]
       |
  multi-step --> [planner] --> [tool_caller] --> [tools] --> [chart_validator] --> [synthesizer]
       |
  Tool Layer (30+ tools across 7 categories)
       |
  +-- USDA Food Environment Atlas (SQLite)
  +-- USDA NASS Quick Stats API
  +-- FEMA Disaster Declarations API
  +-- US Census ACS API
  +-- Open-Meteo Weather API
  +-- DuckDuckGo Web Search (DDGS)
  +-- ML Engine (XGBoost, RandomForest, SHAP, Isolation Forest)
  +-- Plotly Chart Generator (12+ chart types)
  +-- Route Optimizer (TSP-based)
       |
  Archia Cloud Agents (8 specialized agents, parallel execution)
```

**Full architecture details:** [docs/AGRIFLOW_ARCHITECTURE.md](docs/AGRIFLOW_ARCHITECTURE.md)

---

## Machine Learning Methodology

Our ML pipeline was developed and validated across four Jupyter notebooks before being implemented as production Python tools.

### County-Level Risk Modeling (Suyog / NEW1 Notebooks)

Validated on **115 Missouri counties** with **38 features** from USDA Food Environment Atlas:

| Model | R2 | RMSE | Notes |
|---|---|---|---|
| Gradient Boosting | 0.998 | 0.099 | Best overall — production default |
| Random Forest | 0.987 | — | Strong non-linear baseline |
| Linear Regression | 0.983 | 0.328 | Interpretable baseline |
| SVR (RBF, C=10) | 0.912 | 0.746 | Alternative interpretable |

**4 validated interaction features** (all rank top-15 predictors):
- `POVxSNAP` = POVRATE21 x PCT_SNAP22
- `POVxLACCESS` = POVRATE21 x LACCESS_HHNV19
- `FOODxIncome` = FOODINSEC_21_23 / MEDHHINC21
- `SNAPxLACCESS` = PCT_SNAP22 x LACCESS_HHNV19

**Top predictors**: PCT_WICWOMEN16 (~65%), VLFOODSEC_21_23 (~22%), FOODINSEC_18_20 (~5%)

**3 K-Means clusters** (k=3): High-Risk, Low-Risk, Access-Constrained — each with distinct intervention strategy.

### Census-Tract Risk Modeling (NEW2 Notebook)

Validated on **72,242 US census tracts** (1,387 for Missouri):

- **Economic Hardship Index (EHI)** = SNAP_rate + HUNV_rate
- **Structural Inequality Index (SII)** = Black_pct + Hispanic_pct - White_pct
- **Aging Index (AI)** = Senior_pct
- **GBM model**: R2 = 0.9949, RMSE = 0.0033 (target: SNAP_rate)
- **Composite risk score**: `Food_Insecurity_Risk = (EHI_norm + Pred_SNAP_norm) / 2` in [0,1]
- **Highest-risk tract**: 29510111300 (St. Louis city), risk = 0.959

**For detailed methodology:** [docs/NOTEBOOKS_TO_AGENT.md](docs/NOTEBOOKS_TO_AGENT.md)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- API keys: `ARCHIA_TOKEN`, `NASS_API_KEY` (Census/FEMA/Weather are free)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd muidsi-hackathon-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env and set ARCHIA_TOKEN and NASS_API_KEY
```

### Run the System

```bash
# Terminal 1: Start the FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start the React frontend
cd frontend && npm run dev

# Open http://localhost:5173
```

### Verify Setup

```bash
# Check backend health
curl http://localhost:8000/api/health

# Run ML pipeline validation
python scripts/test_ml_pipeline.py
```

---

## Project Structure

```
muidsi-hackathon-2026/
|- src/
|   |- agent/
|   |   |- graph.py                  # LangGraph 9-node StateGraph orchestrator
|   |   |- llm.py                    # ArchiaChatModel (Archia Responses API wrapper)
|   |   |- state.py                  # AgriFlowState TypedDict
|   |   |- nodes/
|   |   |   |- planner.py            # Query decomposition (Haiku)
|   |   |   |- tool_executor.py      # Multi-agent routing + synthetic tool calls
|   |   |   |- synthesizer.py        # Final answer synthesis (Sonnet)
|   |   |   |- chart_validator.py    # Pure Python Plotly spec validator
|   |   |   `- analytics_supervisor.py  # ML pipeline orchestration
|   |   |- tools/
|   |   |   |- food_atlas.py         # USDA Food Environment + Access Atlas
|   |   |   |- nass_api.py           # USDA NASS Quick Stats crop data
|   |   |   |- census_acs.py         # US Census ACS demographics
|   |   |   |- fema_disasters.py     # FEMA disaster declarations
|   |   |   |- weather.py            # Open-Meteo weather API
|   |   |   |- ml_engine.py          # XGBoost, RF, SHAP, anomaly detection
|   |   |   |- chart_generator.py    # Plotly: 12+ chart types
|   |   |   |- route_optimizer.py    # TSP delivery route optimization
|   |   |   |- web_search.py         # DuckDuckGo DDGS agricultural news
|   |   |   |- sql_query.py          # SQLite query execution
|   |   |   |- ingest.py             # Dataset loading and profiling
|   |   |   `- evaluation.py         # RMSE, MAE, CCC, scenario comparison
|   |   `- prompts/
|   |       `- system.py             # 4 system prompts
|   |- api/
|   |   `- main.py                   # FastAPI: SSE streaming, REST endpoints, cache
|   `- mcp_servers/
|       |- ml_server.py              # ML analytics MCP (9 tools)
|       |- chart_server.py           # Chart generation MCP (6 tools)
|       |- sqlite_server.py          # SQLite MCP (2 tools)
|       |- route_server.py           # Route optimization MCP (4 tools)
|       `- data_server.py            # Live data APIs MCP (8 tools)
|- archia/
|   |- agents/                       # Archia cloud agent TOML configs (8 agents)
|   |- prompts/                      # Agent system prompts (Markdown)
|   |- tools/                        # MCP tool TOML configs
|   `- push_agents.sh                # Deploy/sync agents to Archia cloud
|- frontend/
|   `- src/
|       |- App.jsx                   # 4-tab React UI (Query, Dashboard, Data, Map)
|       `- api.js                    # SSE streaming + REST API client
|- notebooks/
|   |- Suyog.ipynb                   # County-level ML (6-feature composite risk)
|   |- NEW1 2.ipynb                  # Multi-model county comparison
|   |- NEW2.ipynb                    # Census-tract ML (72k tracts, EHI/SII/AI)
|   `- NEW2 1.ipynb                  # Tract validation + top risk tracts MO
|- data/
|   `- agriflow.db                   # SQLite: food atlas, census, FEMA data
|- models/                           # Cached trained models (git-ignored)
|- scripts/
|   `- test_ml_pipeline.py           # 7-section ML validation script
|- docs/                             # Project documentation
|- .env.example                      # Environment variable template
|- requirements.txt                  # Python dependencies
`- push_agents.sh                    # Archia cloud deployment script
```

---

## Data Sources

| Source | Data | Access | Key Variables |
|--------|------|--------|---------------|
| USDA Food Environment Atlas | County food insecurity, SNAP, poverty | SQLite (local) | FOODINSEC_21_23, POVRATE21, PCT_SNAP22 |
| USDA Food Access Research Atlas | Census-tract food desert classifications | SQLite (local) | TractSNAP, TractHUNV, LILATracts |
| USDA NASS Quick Stats | Crop yields, production, acreage | REST API (key required) | CORN, SOYBEANS, WHEAT yields |
| US Census ACS | Demographics, income, vehicle access | REST API (free) | B19013 (income), B17001 (poverty) |
| FEMA Disaster Declarations | Historical floods, droughts, disasters | REST API (free) | declaration type, county, date |
| Open-Meteo | 7-day weather forecasts | REST API (free) | temperature, precipitation |
| DuckDuckGo (DDGS) | Current ag news, disease alerts, policy | Web search (free) | Outbreaks, USDA updates, prices |

---

## Tools Inventory (30+ tools)

### [data] — Data Retrieval (9 tools)
`query_food_atlas`, `query_food_access`, `query_nass`, `query_weather`, `query_fema_disasters`, `query_census_acs`, `run_prediction`, `search_web`, `search_agricultural_news`

### [sql] — Database Queries (2 tools)
`list_tables`, `run_sql_query`

### [analytics] — ML Pipeline (12 tools)
`build_feature_matrix`, `train_risk_model`, `predict_risk`, `get_feature_importance`, `detect_anomalies`, `build_tract_feature_matrix`, `compute_food_insecurity_risk`, `run_analytics_pipeline`, `run_eda_pipeline`, `web_search_risks`, `train_crop_model`, `predict_crop_yield`

### [ml] — Model Evaluation (4 tools)
`compute_evaluation_metrics`, `compare_scenarios`, `compute_ccc`, `explain_with_shap`

### [viz] — Visualization (6 tools)
`create_bar_chart`, `create_line_chart`, `create_scatter_map`, `create_risk_heatmap`, `create_choropleth_map`, `create_chart` (supports: scatter, pie, histogram, box, violin, area, funnel, treemap, sunburst, waterfall, indicator, roc_curve, actual_vs_predicted, feature_importance, correlation_matrix)

### [route] — Logistics (4 tools)
`optimize_delivery_route`, `calculate_distance`, `create_route_map`, `schedule_deliveries`

### [ingest] — Data Ingestion (5 tools)
`list_db_tables`, `fetch_and_profile_csv`, `load_dataset`, `run_eda_query`, `drop_table`

---

## Archia Cloud Agents

AgriFlow deploys 8 specialized agents to the Archia cloud platform. Each has a minimal tool set tuned for its domain:

| Agent | Model | Role |
|-------|-------|------|
| `agriflow-system` | Sonnet 4.5 | Primary orchestrator — full SQL + tool access |
| `agriflow-analytics` | Sonnet 4.5 | ML training, SHAP, anomaly detection |
| `agriflow-data` | Haiku 4.5 | USDA/Census/FEMA/weather + web search |
| `agriflow-viz` | Haiku 4.5 | Plotly chart and map generation |
| `agriflow-logistics` | Haiku 4.5 | TSP route optimization and scheduling |
| `agriflow-planner` | Haiku 4.5 | Query decomposition into categorized steps |
| `agriflow-ingest` | Haiku 4.5 | Dataset loading, profiling, EDA |
| `agriflow-evaluator` | Haiku 4.5 | Response quality scoring |

```bash
# Preview changes (dry run)
bash push_agents.sh --dry-run

# Deploy all agents
bash push_agents.sh
```

---

## Environment Configuration

Copy `.env.example` to `.env`:

```bash
# Required
ARCHIA_TOKEN=your_archia_api_token     # Archia cloud platform token
NASS_API_KEY=your_nass_api_key         # USDA NASS Quick Stats (free signup)

# Optional — defaults work for local dev
DEFAULT_MODEL=priv-claude-sonnet-4-5-20250929
DB_PATH=data/agriflow.db
API_HOST=0.0.0.0
API_PORT=8000
```

Census, FEMA, Open-Meteo, and DuckDuckGo APIs are free with no keys required.

`.env` is git-ignored. Never commit API keys.

---

## FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query/stream` | SSE streaming query (progressive rendering) |
| `POST` | `/api/query` | Synchronous query (JSON) |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/charts` | List session charts |
| `POST` | `/api/route` | Direct route optimization |
| `GET` | `/api/eval/summary` | Quality evaluation summary |
| `GET` | `/api/examples` | Example queries for UI |

---

## Documentation

| Document | Description |
|----------|-------------|
| [AGRIFLOW_ARCHITECTURE.md](docs/AGRIFLOW_ARCHITECTURE.md) | Full system design, LangGraph nodes, LLM routing, tool inventory |
| [NOTEBOOKS_TO_AGENT.md](docs/NOTEBOOKS_TO_AGENT.md) | How ML notebooks were translated into production agent tools |
| [SETUP.md](docs/SETUP.md) | Installation, database loading, environment setup |
| [RUNNING_AGENTS.md](docs/RUNNING_AGENTS.md) | 4 deployment modes (local, FastAPI, Archia, hybrid) |
| [DATABASE_SCHEMA.md](docs/DATABASE_SCHEMA.md) | SQLite schema: food_environment, food_access, fema, acs |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Command cheat sheet |
| [HACKATHON_ARCHIA_IO_AGENT_GUIDE.md](docs/HACKATHON_ARCHIA_IO_AGENT_GUIDE.md) | Archia-specific configuration guide |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Agent orchestration | LangGraph (StateGraph, ToolNode), LangChain |
| LLM | Claude Sonnet 4.5 + Haiku 4.5 (via Archia/Anthropic) |
| ML | XGBoost, Scikit-learn (GBM, RF, SVR, Isolation Forest), SHAP |
| Data processing | Pandas, NumPy, SciPy |
| Visualization | Plotly |
| Backend API | FastAPI, Uvicorn (SSE streaming) |
| Frontend | React + Vite |
| Database | SQLite (USDA + Census + FEMA) |
| MCP protocol | FastMCP (5 servers) |
| Cloud agents | Archia (console.archia.app) |
| Web search | DuckDuckGo DDGS |
| External APIs | USDA NASS, Census ACS, FEMA, Open-Meteo |

---

## Performance

| Query Type | Typical Latency | LLM |
|------------|-----------------|-----|
| Simple data lookup | 1-3 s | Haiku |
| Chart generation | 2-5 s | Haiku |
| Route optimization | 2-4 s | Haiku |
| ML model training | 10-30 s | Sonnet |
| Full analytics pipeline | 30-90 s | Sonnet |

The smart router (pure Python regex, no LLM) fast-tracks single-category queries directly to tool execution, saving 1-3 s per query.

---

## Hackathon Context — MUIDSI 2026

AgriFlow was developed for the **MU Institute for Data Science and Informatics (MUIDSI) Hackathon 2026**, themed around AI-driven solutions to agricultural and food supply chain challenges in Missouri.

### Problem Statement

Missouri has 115 counties with varying degrees of food insecurity, crop dependency, and infrastructure gaps. Food distribution planners need tools that can:
1. Identify the most vulnerable counties and census tracts
2. Predict risk under drought or supply-disruption scenarios
3. Optimize delivery logistics for emergency food distribution
4. Stay current on crop diseases, weather alerts, and policy changes

### Our Approach

We built a conversational AI system that sits on top of all relevant federal data sources (USDA, Census, FEMA) and trained ML models. Planners can ask natural-language questions and receive structured answers with charts, maps, and delivery routes — no data science expertise required.

The ML models were developed iteratively through four research notebooks, validated against federal data, and then embedded directly into the agentic tool layer so that any query can trigger model inference on demand.

---

## Acknowledgments

- **USDA Economic Research Service** — Food Environment Atlas, Food Access Research Atlas
- **USDA NASS** — National Agricultural Statistics Service Quick Stats
- **US Census Bureau** — American Community Survey (ACS)
- **FEMA** — Disaster Declarations API
- **Open-Meteo** — Free weather forecast API
- **Anthropic** — Claude Sonnet 4.5 / Haiku 4.5
- **Archia** — Cloud agent platform (console.archia.app)
- **LangGraph / LangChain** — Agent orchestration framework
