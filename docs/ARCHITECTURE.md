# AgriFlow Architecture

**AgriFlow** is a hybrid multi-agent AI system for agricultural supply chain intelligence, combining Archia Cloud orchestration with local LangGraph execution.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
│  • Archia Console (console.archia.app)                      │
│  • React Frontend (localhost:3000)                          │
│  • FastAPI REST (localhost:8000)                            │
│  • Python Direct (run_agent.py)                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              Archia Cloud (registry.archia.app)             │
│  • 26 Skills (user query → agent routing)                   │
│  • 7 Agents (specialized AI personas)                       │
│  • Model Routing (Haiku cheap, Sonnet powerful)             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│           Local LangGraph Engine (Python)                    │
│  • StateGraph workflow (planner → tools → synthesizer)      │
│  • 30 Tools across 6 categories                             │
│  • Multi-source data integration                            │
│  • ML model training & inference                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data & Model Layer                          │
│  • SQLite (food_environment, food_access)                   │
│  • External APIs (USDA NASS, Census, FEMA, Weather)         │
│  • ML Models (XGBoost, RandomForest) cached in models/      │
│  • MCP Servers (3 local servers for charts, routes, SQL)    │
└─────────────────────────────────────────────────────────────┘
```

## Agentic Pipeline

### 1. Query Flow

```
User Query
  ↓
Skill Matching (Archia)
  ↓
Agent Selection (7 agents available)
  ↓
LangGraph Workflow (local execution)
  ├─ Planner: Decompose query into sub-tasks
  ├─ Tool Executor: Execute tools based on categories
  │   ├─ [data] → 7 data retrieval tools
  │   ├─ [sql] → 2 database tools
  │   ├─ [ml] → 4 ML evaluation tools
  │   ├─ [analytics] → 13 ML training/prediction tools
  │   ├─ [viz] → 4 visualization tools
  │   └─ [route] → 4 logistics tools
  └─ Synthesizer: Combine results into coherent answer
  ↓
Response (JSON with answer, charts, analytics)
```

### 2. Agent Architecture

**7 Specialized Agents:**

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| **agriflow-planner** | Haiku | 0 | Query decomposition, task routing |
| **AGRIFLOW_SYSTEM** | Sonnet 4.5 | 2 | SQL queries, complex analysis |
| **agriflow-data** | Haiku | 7 | Fast data retrieval (USDA, Census, FEMA, Weather) |
| **agriflow-viz** | Haiku | 4 | Chart generation (Plotly) |
| **agriflow-logistics** | Haiku | 4 | Route optimization, delivery scheduling |
| **agriflow-analytics** | Sonnet 4.5 | 13 | ML training, prediction, SHAP, anomalies |
| **agriflow-ml** | Sonnet 4.5 | 3 | *Deprecated* - merged into analytics |

**Cost Optimization:**
- 70% of queries use **Haiku** ($0.001/call) for simple tasks
- 30% use **Sonnet 4.5** ($0.015-0.050/call) for complex analysis/ML

### 3. Tool Categories

**[data] - 7 Tools (Haiku, fast)**
- `query_food_atlas` - USDA Food Environment Atlas (county-level food insecurity)
- `query_food_access` - USDA Food Access Research Atlas (food deserts)
- `query_nass` - USDA NASS Quick Stats (crop production, yields)
- `query_weather` - Open-Meteo weather forecasts
- `query_fema_disasters` - FEMA disaster declarations
- `query_census_acs` - US Census demographics
- `run_prediction` - Quick risk scoring (heuristic or ML)

**[sql] - 2 Tools (Sonnet, powerful)**
- `list_tables` - Database schema discovery
- `run_sql_query` - Custom SQL queries (read-only, auto-limited)

**[ml] - 4 Tools (Sonnet, complex)**
- `compute_evaluation_metrics` - RMSE, MAE, R², CCC, accuracy, F1
- `compare_scenarios` - Side-by-side scenario comparison
- `compute_ccc` - Concordance Correlation Coefficient
- `explain_with_shap` - SHAP explainability for any model

**[analytics] - 13 Tools (Sonnet, ML-heavy)**
- `build_feature_matrix` - Multi-source data fusion (food + census + FEMA + weather)
- `train_risk_model` - XGBoost/RF training with 5-fold CV
- `predict_risk` - Inference with confidence intervals
- `train_crop_model` - Crop dependency modeling
- `predict_crop_yield` - Yield impact predictions
- `get_feature_importance` - SHAP-based feature analysis
- `detect_anomalies` - Isolation Forest outlier detection
- `web_search_risks` - DuckDuckGo web search for emerging threats
- `run_analytics_pipeline` - Full automated analytics workflow
- Plus 4 additional evaluation/prediction tools

**[viz] - 4 Tools (Haiku, fast)**
- `create_bar_chart` - Horizontal/vertical bar charts
- `create_line_chart` - Time series line plots
- `create_scatter_map` - Geographic scatter maps with color encoding
- `create_risk_heatmap` - Correlation heatmaps

**[route] - 4 Tools (Haiku, fast)**
- `optimize_delivery_route` - TSP-based route optimization
- `calculate_distance` - Haversine distance between locations
- `create_route_map` - Plotly route visualization
- `schedule_deliveries` - Time-based delivery scheduling

### 4. LangGraph Workflow

**State Machine:**

```python
class AgriFlowState(TypedDict):
    messages: list[BaseMessage]     # Conversation history
    plan: list[dict]                # Decomposed sub-tasks
    current_step: int               # Current execution step
    tool_results: dict              # Tool outputs by name
    reasoning_trace: list[str]      # Step-by-step reasoning
    final_answer: Optional[str]     # Synthesized response
```

**Node Flow:**

```
START
  ↓
planner (Haiku)
  • Decomposes query into categorized sub-tasks
  • Tags: [data], [sql], [ml], [analytics], [viz], [route]
  ↓
tool_caller (Model routing based on category)
  • [data], [viz], [route] → Haiku
  • [sql], [ml], [analytics] → Sonnet 4.5
  ↓
tools (ToolNode with ALL_TOOLS)
  • Executes tools in parallel where possible
  • Returns results to state.tool_results
  ↓
should_continue? (Conditional edge)
  • More tools needed? → Return to tool_caller
  • All done? → Go to synthesizer
  ↓
synthesizer (Sonnet 4.5)
  • Combines tool results into coherent narrative
  • Generates final_answer
  ↓
END
```

**Parallelization:**
- Independent tool calls execute concurrently
- Data queries from multiple sources run in parallel
- Chart generation happens simultaneously with analysis

### 5. ML Analytics Pipeline

**Feature Engineering:**
```
Data Sources:
  ├─ food_environment (USDA Food Atlas) - 21 columns
  ├─ food_access (USDA Food Access) - 15 columns
  ├─ Census ACS - 10 derived features
  ├─ FEMA disasters - Disaster count + recency
  └─ Weather - Anomaly flags
       ↓
  Feature Matrix: 115 MO counties × 35 features
```

**Model Training:**
```
build_feature_matrix(state="MO")
  ↓
train_risk_model(
  model_type="xgboost",      # or "random_forest"
  target="food_insecurity",  # or custom target
  cv_folds=5
)
  ↓
Model saved to models/risk_model_xgboost_MO.pkl
Metrics: R², RMSE, MAE, CCC
```

**Prediction:**
```
predict_risk(
  state="MO",
  scenario="drought",         # or "baseline", "price_shock"
  yield_reduction_pct=30
)
  ↓
Returns: County scores + confidence intervals + rankings
```

**Explainability:**
```
get_feature_importance(model="latest")
  ↓
SHAP TreeExplainer
  ↓
Feature importance rankings + SHAP plots
```

### 6. Hybrid Architecture Benefits

**Why Archia + LangGraph?**

| Aspect | Archia Cloud | Local LangGraph |
|--------|--------------|-----------------|
| **Routing** | Skills → Agents | Agents → Tools |
| **Collaboration** | Web UI for team | Direct Python API |
| **Data Access** | Remote (slow) | Local (fast) |
| **Model Training** | Not supported | Full ML pipeline |
| **Cost** | Per-call API fees | Local compute |
| **Latency** | Network overhead | Instant |
| **Security** | Cloud-hosted | Data stays local |

**Best of Both Worlds:**
- Archia provides: Team collaboration, skill-based routing, web UI
- LangGraph provides: Fast tool execution, local data access, ML training

## Deployment Modes

### 1. Pure Archia Cloud
```bash
# Query via console.archia.app web UI
# All tools executed remotely (slower, simpler setup)
```

**Pros:** No local setup, team access, managed infrastructure
**Cons:** Slower, no ML training, data must be in cloud

### 2. Pure Local Python
```bash
python run_agent.py
# Direct LangGraph execution, no Archia
```

**Pros:** Fastest, full ML access, data stays local
**Cons:** No web UI, single-user, manual setup

### 3. Local FastAPI Server
```bash
uvicorn src.api.main:app --port 8000
# REST API for React frontend
```

**Pros:** Web frontend, team access, local execution
**Cons:** Requires server management

### 4. Hybrid (Recommended)
```bash
# Archia Console for queries → Local LangGraph for execution
# Skills route to agents → Agents route to local tools
```

**Pros:** Web UI + fast execution + ML access + team collaboration
**Cons:** Requires both Archia account and local setup

## Data Flow Examples

### Example 1: Simple Data Query

```
User: "What is the food insecurity rate in Wayne County, MO?"

Archia Skill: "Food Insecurity Analysis"
  ↓
Agent: agriflow-data (Haiku)
  ↓
LangGraph:
  planner: [data] query_food_atlas(state="MO", county="Wayne")
  ↓
  tool_executor: Execute query_food_atlas
  ↓
  synthesizer: "Wayne County, MO has a food insecurity rate of 18.2% (2015-2017)"
```

**Time:** ~2 seconds | **Cost:** $0.001

### Example 2: Complex ML Analytics

```
User: "Train an XGBoost model to predict drought risk in Missouri counties.
       Show feature importance using SHAP."

Archia Skill: "Train Risk Model" + "Feature Analysis"
  ↓
Agent: agriflow-analytics (Sonnet 4.5)
  ↓
LangGraph:
  planner:
    [analytics] build_feature_matrix(state="MO")
    [analytics] train_risk_model(model_type="xgboost", target="risk_score")
    [analytics] get_feature_importance(model="latest")
  ↓
  tool_executor (parallel):
    • build_feature_matrix → 115 counties × 35 features
    • train_risk_model → R²=0.94, RMSE=0.12, CCC=0.91
    • get_feature_importance → SHAP explanations
  ↓
  synthesizer:
    "XGBoost model trained on 115 Missouri counties with R²=0.94.
     Top features: food_insecurity (32%), poverty_rate (25%), SNAP (18%).
     Model saved to models/risk_model_xgboost_MO.pkl"
```

**Time:** ~30 seconds | **Cost:** $0.050

### Example 3: Visualization Pipeline

```
User: "Create a bar chart of the top 10 Missouri counties with highest
       food insecurity and a map showing their locations."

Archia Skills: "Bar Chart" + "Geographic Map"
  ↓
Agents: agriflow-data (Haiku) + agriflow-viz (Haiku)
  ↓
LangGraph:
  planner:
    [data] query_food_atlas(state="MO")
    [viz] create_bar_chart(x="County", y="FOODINSEC_15_17", top_n=10)
    [viz] create_scatter_map(color_col="FOODINSEC_15_17")
  ↓
  tool_executor (parallel):
    • query_food_atlas → 115 rows
    • create_bar_chart → Plotly bar chart saved to outputs/charts/
    • create_scatter_map → Plotly map with color-coded points
  ↓
  synthesizer: Answer + embedded charts
```

**Time:** ~5 seconds | **Cost:** $0.003

## Technology Stack

**Frontend:**
- React (when connected)
- Archia Console Web UI

**API Layer:**
- FastAPI (REST endpoints)
- Archia Cloud (agent routing, skill management)

**Agent Framework:**
- LangGraph (workflow orchestration)
- LangChain (tool integration)
- Anthropic Claude (Haiku 4.5, Sonnet 4.5)

**Tools:**
- Pandas, NumPy (data processing)
- Scikit-learn (baseline ML)
- XGBoost, RandomForest (production ML)
- SHAP (explainability)
- Plotly (visualizations)
- Requests (API calls)

**Data:**
- SQLite (local database)
- USDA APIs (NASS Quick Stats)
- US Census API (ACS)
- FEMA API (disasters)
- Open-Meteo API (weather)

**Infrastructure:**
- MCP Protocol (tool adapters)
- Python 3.10+
- Windows/Linux/Mac compatible

## Performance Characteristics

**Query Latency:**
| Query Type | Time | Cost | Model |
|------------|------|------|-------|
| Simple data lookup | 1-3s | $0.001 | Haiku |
| SQL query | 2-5s | $0.015 | Sonnet |
| Chart generation | 1-3s | $0.002 | Haiku |
| Route optimization | 2-4s | $0.003 | Haiku |
| ML training | 10-30s | $0.050 | Sonnet |
| Full analytics pipeline | 30-60s | $0.100 | Sonnet |

**Throughput:**
- Concurrent queries: 10-20 (limited by API rate limits)
- Tool parallelization: 3-5 tools simultaneously
- Database queries: < 100ms (local SQLite)

**Cost Analysis:**
- Average query: $0.05
- Heavy ML session: $0.50
- Daily usage (50 queries): $2.50
- Monthly estimate: $75

## Security & Privacy

**Data:**
- All sensitive data stays local (SQLite database)
- No PII sent to Archia Cloud
- API keys stored in .env (gitignored)

**Authentication:**
- Archia: Bearer token authentication
- FastAPI: Can add auth middleware
- Local: No auth required

**Network:**
- Archia: HTTPS (TLS 1.3)
- External APIs: HTTPS
- Local: Localhost only (no external exposure)

## Scalability

**Current Capacity:**
- Database: 115 MO counties, 70K census tracts
- Models: XGBoost on 115 × 35 features (~10 seconds training)
- Predictions: 100+ counties per query
- Charts: Unlimited (file-based storage)

**Scale-Up Path:**
1. Expand to all 50 states (~3,100 counties)
2. Add more data sources (USGS, NOAA, etc.)
3. Implement caching layer (Redis)
4. Deploy FastAPI with load balancer
5. Use cloud database (PostgreSQL)
6. Add model versioning (MLflow)

## Monitoring & Debugging

**Logs:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**LangGraph Visualization:**
```python
from src.agent.graph import create_agent
agent = create_agent()
agent.get_graph().print_ascii()  # Print workflow
```

**Tool Inspection:**
```python
from src.agent.nodes.tool_executor import ALL_TOOLS
print(f"{len(ALL_TOOLS)} tools:")
for tool in ALL_TOOLS:
    print(f"  - {tool.name}")
```

**Archia Monitoring:**
- console.archia.app → Execution History
- View all agent calls, tool usage, errors

## Testing

**Integration Tests:**
```bash
python test_archia_integration.py    # 6 tests
python test_complete_pipeline.py     # 7 tests
```

**Unit Tests:**
```bash
pytest tests/                        # (add unit tests as needed)
```

**Manual Testing:**
```bash
python run_agent.py                  # Interactive mode
```

## ML Evaluation Standards

All regression models must report these metrics:
- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE** — Mean Absolute Error (interpretable units)
- **R²** — Coefficient of determination (0–1, higher is better)
- **CCC** — Concordance Correlation Coefficient (agricultural standard, 0–1)

Minimum acceptable: R² > 0.85, CCC > 0.80. All models generate SHAP values for top 10 features.

## API Rate Limits & Data Freshness

| Source | Rate Limit | Data Vintage |
| ------ | ---------- | ------------ |
| USDA NASS | 1000 req/day | Weekly (growing season) |
| Census ACS | 500 req/day per key | 2022 5-year estimates |
| Open-Meteo | Unlimited (no key) | 7-day forecast (real-time) |
| FEMA | Unlimited (public) | Continuous |
| Food Environment Atlas | Local DB | 2015–2017 |

Always check local database before calling external APIs.

## Test Suites

```bash
python test_archia_integration.py    # 6 tests — Archia connectivity
python test_complete_pipeline.py     # 7 tests — End-to-end workflow
pytest tests/                        # Unit tests
```

## Future Enhancements

**Short Term (v2.1):**
- [ ] Add unit tests for all tools
- [ ] Implement response caching
- [ ] Add authentication to FastAPI
- [ ] Create React dashboard mockups

**Medium Term (v2.2):**
- [ ] Expand to all 50 states
- [ ] Add real-time data feeds
- [ ] Implement model A/B testing
- [ ] Add user analytics

**Long Term (v3.0):**
- [ ] Multi-tenant support
- [ ] Cloud database migration
- [ ] Mobile app
- [ ] Real-time alerts

## References

- [Archia Cloud Documentation](references/Archia_Docs.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Claude API Documentation](https://docs.anthropic.com/)
- [USDA Data Sources](https://www.ers.usda.gov/)
