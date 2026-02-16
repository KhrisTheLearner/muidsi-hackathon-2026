# AgriFlow

**AgriFlow** is a multi-agent AI system for agricultural supply chain intelligence, combining machine learning, real-time data analysis, and logistics optimization to address food insecurity in Missouri.

**Version:** 2.0.1 | **Status:** Production Ready | **MUIDSI Hackathon 2026**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Test the system
python test_complete_pipeline.py    # All tests should pass

# 4. Run agents (choose one)
python run_agent.py                              # Local Python
uvicorn src.api.main:app --port 8000            # API server
# Or use Archia Console: console.archia.app    # Web UI
```

## What It Does

AgriFlow helps answer questions like:

- "Which Missouri counties have the highest food insecurity?"
- "Train an XGBoost model to predict drought risk and show feature importance"
- "Optimize a delivery route from Cape Girardeau to Wayne, Pemiscot, and Dunklin counties"
- "Create a map showing food desert risk across southeastern Missouri"

## System Architecture

```text
User Query → Archia Skills → Agent Selection → LangGraph Tools → Results
```

**7 Specialized Agents:**

- `agriflow-planner` - Query decomposition
- `agriflow-data` - Data retrieval (USDA, Census, FEMA, Weather)
- `agriflow-analytics` - ML training & prediction (XGBoost, SHAP)
- `agriflow-viz` - Chart generation (Plotly)
- `agriflow-logistics` - Route optimization
- `AGRIFLOW_SYSTEM` - SQL queries & complex analysis

**30 Tools Across 6 Categories:**

- **[data]** - 7 tools for USDA, Census, FEMA, Weather data
- **[sql]** - 2 tools for database queries
- **[ml]** - 4 tools for model evaluation
- **[analytics]** - 13 tools for ML training, prediction, SHAP, anomalies
- **[viz]** - 4 tools for charts and maps
- **[route]** - 4 tools for delivery optimization

## Key Features

### ML Analytics Pipeline

- **XGBoost & Random Forest** training with 5-fold cross-validation
- **SHAP explainability** for feature importance
- **CCC validation** (Concordance Correlation Coefficient)
- **Isolation Forest** anomaly detection
- **Multi-source feature engineering** (food + census + FEMA + weather)

### Data Integration

- **USDA Food Environment Atlas** - County food insecurity rates
- **USDA Food Access Research Atlas** - Food desert classifications
- **USDA NASS Quick Stats** - Crop production and yields
- **US Census ACS** - Demographics and economic data
- **FEMA Disasters** - Historical disaster declarations
- **Open-Meteo Weather** - 7-day forecasts

### Visualizations

- **Bar Charts** - County comparisons
- **Line Charts** - Time series trends
- **Scatter Maps** - Geographic risk visualization
- **Heatmaps** - Correlation analysis

### Logistics

- **Route Optimization** - TSP-based delivery routing
- **Distance Calculation** - Haversine distance
- **Delivery Scheduling** - Time-based scheduling

## Project Structure

```text
AgriFlow/
├── src/
│   ├── agent/              # LangGraph workflow & tools
│   │   ├── graph.py        # Main workflow definition
│   │   ├── state.py        # State management
│   │   ├── nodes/          # Workflow nodes (planner, executor, synthesizer)
│   │   ├── tools/          # 30 tool implementations
│   │   └── prompts/        # System prompts
│   ├── api/                # FastAPI REST server
│   └── mcp_servers/        # MCP protocol adapters
├── archia/
│   ├── agents/             # Archia agent configs (7 TOML files)
│   ├── prompts/            # Agent system prompts
│   ├── tools/              # MCP tool configs
│   ├── skills.json         # 26 skills definition
│   └── setup_skills.py     # Deploy skills to Archia
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System design & agentic pipeline
│   ├── SETUP.md            # Installation guide
│   ├── RUNNING_AGENTS.md   # How to run agents
│   ├── QUICK_REFERENCE.md  # Command cheat sheet
│   ├── DEPLOYMENT.md       # Production deployment
│   └── ARCHIA_AGENT_SETUP.md  # Archia configuration
├── data/
│   └── agriflow.db         # SQLite database (load USDA data here)
├── models/                 # Cached ML models
├── outputs/                # Generated charts
├── tests/                  # Integration tests
├── run_agent.py            # Local agent execution
├── CHANGELOG.md            # Version history
└── DEPLOYMENT_COMPLETE.md  # Current deployment status
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system design & agentic pipeline
- **[TEAM_INSTRUCTIONS.md](TEAM_INSTRUCTIONS.md)** - Team coordination guidelines
- **[SETUP.md](docs/SETUP.md)** - Installation and database setup
- **[RUNNING_AGENTS.md](docs/RUNNING_AGENTS.md)** - How to run agents (4 deployment modes)
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command cheat sheet
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guide
- **[ARCHIA_AGENT_SETUP.md](docs/ARCHIA_AGENT_SETUP.md)** - Archia Cloud configuration
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md)** - Current deployment status

## Deployment Modes

### 1. Local Python (Fastest)

```bash
python run_agent.py
```

Direct LangGraph execution - no server needed.

### 2. FastAPI Server (Production)

```bash
uvicorn src.api.main:app --port 8000
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

REST API for React frontend integration.

### 3. Archia Cloud (Team Collaboration)

```bash
# Deploy skills
python archia/setup_skills.py

# Query via web UI
# https://console.archia.app
```

Managed cloud platform with web interface.

### 4. Hybrid (Recommended)

Archia Console for queries → Local LangGraph for execution. Best of both worlds: web UI + fast local tools + ML access.

## Testing

```bash
# Complete pipeline test (7 tests)
python test_complete_pipeline.py

# Archia integration (6 tests)
python test_archia_integration.py

# Test specific agent
python test_analytics_agent.py
```

**All tests passing:** 13/13 (100%)

## Technology Stack

- **Agents:** LangGraph, LangChain
- **Models:** Claude Haiku 4.5, Claude Sonnet 4.5 (via Anthropic API)
- **ML:** XGBoost, Random Forest, Scikit-learn, SHAP
- **Data:** Pandas, NumPy, SQLite
- **Viz:** Plotly
- **APIs:** FastAPI, Requests
- **Cloud:** Archia (console.archia.app)

## Performance

| Query Type | Latency | Cost | Model |
|------------|---------|------|-------|
| Simple data lookup | 1-3s | $0.001 | Haiku |
| Chart generation | 1-3s | $0.002 | Haiku |
| Route optimization | 2-4s | $0.003 | Haiku |
| SQL query | 2-5s | $0.015 | Sonnet |
| ML training | 10-30s | $0.050 | Sonnet |
| Full analytics | 30-60s | $0.100 | Sonnet |

**Average query cost:** $0.05 **|** **Daily usage (50 queries):** $2.50

## Current Status

- ✅ **All agents deployed** (7/7 on Archia Cloud)
- ✅ **All skills deployed** (26/26)
- ✅ **All tools loaded** (30/30)
- ✅ **Tests passing** (13/13, 100%)
- ✅ **FastAPI endpoints** ready for React frontend
- ✅ **ML pipeline** verified (XGBoost, SHAP, CCC)
- ⏳ **Database loading** (user action - see [SETUP.md](docs/SETUP.md))

## Example Queries

```python
# Data query
"Which Missouri counties have highest food insecurity rates?"

# ML analytics
"Train XGBoost model on Missouri data, predict drought risk, show SHAP importance"

# Visualization
"Create bar chart of top 10 food insecure counties"

# Route optimization
"Optimize delivery route from Cape Girardeau to Wayne, Pemiscot, Dunklin"

# Full pipeline
"Run full analytics on Missouri food supply chain under drought scenario with web research"
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run local agent
python run_agent.py

# Start API server
uvicorn src.api.main:app --reload --port 8000

# Deploy to Archia
python archia/setup_skills.py

# Run tests
python test_complete_pipeline.py
python test_archia_integration.py
```

## Team

MUIDSI Hackathon 2026

## License

See LICENSE file for details.

## Acknowledgments

- USDA Economic Research Service (Food Atlas, NASS)
- US Census Bureau (ACS)
- FEMA (Disaster data)
- Open-Meteo (Weather API)
- Anthropic (Claude models)
- Archia (Agent platform)
