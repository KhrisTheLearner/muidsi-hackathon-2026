# AgriFlow Setup & Usage

Quick reference for installation, running agents, and common commands.

## Prerequisites

- Python 3.10+
- Git
- Archia account (optional, for cloud deployment)

## Installation

```bash
git clone <repo-url>
cd muidsi-hackathon-2026
pip install -r requirements.txt
cp .env.example .env   # Edit with your API keys
```

## Environment Variables

```bash
# Required
DEFAULT_MODEL=priv-claude-sonnet-4-5-20250929

# Optional
NASS_API_KEY=your_key          # USDA NASS data
ARCHIA_TOKEN=your_token        # Archia cloud
ARCHIA_BASE_URL=https://registry.archia.app
CENSUS_API_KEY=your_key        # US Census ACS
DB_PATH=data/agriflow.db
MODEL_DIR=models
```

## Database Setup

Populate `data/agriflow.db` with USDA data:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect('data/agriflow.db')

# Food Environment Atlas (https://www.ers.usda.gov/data-products/food-environment-atlas/)
food_env = pd.read_csv('path/to/food_environment.csv')
food_env.to_sql('food_environment', conn, if_exists='replace', index=False)

# Food Access Research Atlas (https://www.ers.usda.gov/data-products/food-access-research-atlas/)
food_access = pd.read_csv('path/to/food_access.csv')
food_access.to_sql('food_access', conn, if_exists='replace', index=False)
conn.close()
```

## Running Agents

### Option 1: Local Python (fastest)

```python
from src.agent.graph import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent()
result = agent.invoke({
    "messages": [HumanMessage("Which MO counties have highest food insecurity?")],
    "plan": [], "current_step": 0, "tool_results": {},
    "reasoning_trace": [], "final_answer": None,
})
print(result["final_answer"])
```

Or: `python run_agent.py`

### Option 2: FastAPI + React Frontend (production)

```bash
# Backend
uvicorn src.api.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm run dev
```

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Send query, get response |
| `/api/query/stream` | POST | SSE streaming response |
| `/api/health` | GET | System status |
| `/api/examples` | GET | Example queries |
| `/api/charts` | GET | List generated charts |
| `/api/analytics` | GET | List analytics reports |

### Option 3: Archia Cloud (team collaboration)

```bash
export ARCHIA_TOKEN="your_token"
python archia/setup_skills.py          # Push skills
python archia/setup_skills.py --verify # Verify deployment
```

Use via web UI at https://console.archia.app or API:
```bash
curl -X POST https://registry.archia.app/v1/responses \
  -H "Authorization: Bearer $ARCHIA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "agent:agriflow-analytics", "input": "Train XGBoost on MO", "stream": false}'
```

## Verification Commands

```bash
python -c "from src.agent.graph import create_agent; create_agent()"       # Graph compiles
python -c "from src.agent.nodes.tool_executor import ALL_TOOLS; print(len(ALL_TOOLS))"  # Tool count
curl http://localhost:8000/api/health                                       # API health
```

## Tool Categories

| Category | Tools | Model | Examples |
|----------|-------|-------|---------|
| data | 7 | Haiku | query_food_atlas, query_nass, query_weather |
| sql | 2 | Sonnet | list_tables, run_sql_query |
| ml | 4 | Sonnet | compute_evaluation_metrics, compare_scenarios |
| analytics | 13 | Sonnet | train_risk_model, predict_risk, get_feature_importance |
| viz | 4 | Haiku | create_bar_chart, create_scatter_map |
| route | 4 | Haiku | optimize_delivery_route, schedule_deliveries |
| ingest | 5 | Haiku | load_dataset, fetch_and_profile_csv |

## File Locations

```
src/agent/graph.py                 LangGraph workflow (router → planner → tools → synthesizer)
src/agent/nodes/tool_executor.py   Multi-agent routing logic
src/agent/tools/                   30+ tool implementations
src/api/main.py                    FastAPI REST + SSE endpoints
frontend/src/App.jsx               React frontend
archia/setup_skills.py             Push skills to Archia cloud
data/agriflow.db                   SQLite database
models/                            Cached ML models (gitignored)
```

## Example Queries

```
Data:      "Food insecurity rates in Missouri counties"
ML:        "Train XGBoost, predict drought risk, show SHAP"
Viz:       "Bar chart of top 10 food insecure counties"
Route:     "Optimize delivery from Cape Girardeau to Wayne, Pemiscot, Dunklin"
Full:      "Full analytics on Missouri drought scenario with web research"
Ingest:    "Load a CSV from URL and profile it"
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Empty database results | Load USDA data into `data/agriflow.db` |
| ModuleNotFoundError | `pip install -r requirements.txt` |
| NASS API errors | Add `NASS_API_KEY` to .env ([get key](https://quickstats.nass.usda.gov/api)) |
| Archia 401/skills missing | Check token + workspace at console.archia.app |
| Graph compilation fails | Check imports in `tool_executor.py` |
| Model not found | Train first with `train_risk_model` |
| Streaming 404 | Restart uvicorn (stale process won't pick up new routes) |

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design, agent pipeline, ML standards
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) — Table schemas and column details
- [CHANGELOG.md](../CHANGELOG.md) — Version history
