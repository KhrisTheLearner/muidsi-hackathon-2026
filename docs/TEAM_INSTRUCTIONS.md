# AgriFlow Team Instructions

**Version:** 2.0.2 | **Date:** 2026-02-16 | **MUIDSI Hackathon 2026**

## Purpose

This document provides team coordination guidelines based on the foundational design principles established during AgriFlow's architecture planning. These instructions ensure consistent development, deployment, and maintenance across all team members working on the multi-agent agricultural supply chain intelligence system.

## 1. Multi-Agent Architecture Coordination

### Agent Specialization Principle

AgriFlow uses 7 specialized agents, each with a distinct purpose. **Never** combine agent responsibilities or create overlapping functionality.

**Agent Roster:**

- **agriflow-planner** - Query decomposition and task planning only. No tool execution.
- **agriflow-data** - Data retrieval from external APIs (USDA, Census, FEMA, Weather). Read-only operations.
- **agriflow-analytics** - ML training, prediction, SHAP explainability, CCC validation, anomaly detection. Sonnet 4.5 for complex reasoning.
- **agriflow-viz** - Chart and map generation using Plotly. Visual output only.
- **agriflow-logistics** - Route optimization, distance calculation, delivery scheduling. Transportation focus.
- **AGRIFLOW_SYSTEM** - SQL queries and complex multi-step analysis. Database operations and orchestration.
- **agriflow-ml** - Legacy agent. Deprecated in v2.0.0. Do not use for new features.

**Coordination Rule:** When adding new functionality, first identify which agent owns that capability. If no agent fits, propose a new agent rather than overloading an existing one.

### Model Assignment Strategy

Cost optimization is critical. Use this decision tree:

- **Haiku (70% of queries, $0.001/call):**
  - Simple data lookups
  - Chart generation
  - Route calculations
  - Any task with deterministic logic

- **Sonnet 4.5 (30% of queries, $0.015-0.050/call):**
  - SQL query generation
  - ML model training and evaluation
  - Multi-step reasoning
  - Complex analysis requiring context

**Rule:** Never use Sonnet for tasks Haiku can handle. Budget for 50 daily queries is $2.50. Monitor costs weekly.

## 2. Tool Categorization and Routing

### Tool Organization

All tools are organized into 6 categories with strict boundaries:

**[data] - 7 tools (Haiku):**
- query_food_atlas
- query_food_access_atlas
- query_nass_api
- get_weather_forecast
- query_fema_disasters
- query_census_acs
- get_drought_monitor

**Purpose:** External API calls for data retrieval. No computation or analysis.

**[sql] - 2 tools (Sonnet):**
- list_tables
- run_sql_query

**Purpose:** Database schema inspection and SQL execution on agriflow.db.

**[ml] - 4 tools (Sonnet):**
- run_prediction (legacy heuristic)
- compute_evaluation_metrics
- compare_scenarios
- compute_ccc

**Purpose:** Model evaluation and comparison. Legacy tools maintained for backwards compatibility.

**[analytics] - 13 tools (Sonnet):**
- build_feature_matrix
- train_risk_model
- predict_risk
- train_crop_model
- predict_crop_yield
- get_feature_importance
- detect_anomalies
- web_search_risks
- (Plus 5 evaluation tools from [ml] category)

**Purpose:** Full ML analytics pipeline. XGBoost, Random Forest, SHAP, anomaly detection, web research.

**[viz] - 4 tools (Haiku):**
- create_bar_chart
- create_line_chart
- create_scatter_map
- create_heatmap

**Purpose:** Plotly visualization generation. Outputs JSON specs for React rendering.

**[route] - 4 tools (Haiku):**
- optimize_delivery_route
- calculate_distance
- create_route_map
- schedule_deliveries

**Purpose:** Logistics and transportation optimization. TSP-based routing.

### Routing Logic

The LangGraph planner tags each step with a category. The tool_executor routes based on tags:

```
[data] -> agriflow-data (Haiku)
[sql] -> AGRIFLOW_SYSTEM (Sonnet)
[ml] -> agriflow-analytics (Sonnet)
[analytics] -> agriflow-analytics (Sonnet)
[viz] -> agriflow-viz (Haiku)
[route] -> agriflow-logistics (Haiku)
```

**Rule:** When creating new tools, always assign a category. Update `src/agent/nodes/tool_executor.py` and `src/agent/prompts/system.py` with the new tool.

## 3. LangGraph Workflow Coordination

### State Machine Flow

AgriFlow uses a 5-node StateGraph:

```
START -> planner -> tool_caller <-> tools -> synthesizer -> END
```

**Node Responsibilities:**

1. **planner** - Receives user query, decomposes into steps, assigns category tags
2. **tool_caller** - Routes to appropriate agent based on category
3. **tools** - Executes tools, returns results to state
4. **synthesizer** - Combines results into natural language answer with sources
5. **END** - Returns final answer, charts, analytics to user

**State Fields (AgriFlowState):**
- messages (list[BaseMessage])
- plan (list[dict])
- current_step (int)
- tool_results (dict)
- reasoning_trace (list[str])
- final_answer (str)

**Rule:** Never modify state structure without updating all 5 nodes. State is shared across all agents.

### Parallel Execution

Tools within the same category execute in parallel when possible. The planner identifies independent steps:

```python
# Good - parallel execution
[data] query_food_atlas(state="MO")
[data] query_census_acs(state="MO")
[data] get_weather_forecast(state="MO")

# Bad - unnecessary sequential execution
[data] query_food_atlas(state="MO") -> wait -> [data] query_census_acs(state="MO")
```

**Rule:** When writing planner prompts, encourage parallel tool calls for independent data retrieval.

## 4. ML Analytics Pipeline Workflow

### Feature Engineering First

All ML workflows start with `build_feature_matrix`. This tool joins data from all sources:

```
food_environment (21 cols) + census_acs (10 cols) + fema_history + weather -> 115 counties x 35 features
```

**Never** train models on raw API data. Always use the feature matrix.

### Model Training and Caching

Models are trained on-the-fly and cached in `models/` directory:

- `models/risk_xgboost_MO.pkl` - State-specific risk model
- `models/crop_rf_MO_corn.pkl` - Crop-specific yield model

**Retraining Logic:**
- Retrain if model file missing
- Retrain if data updated (check DB timestamp)
- Retrain if user explicitly requests it

**Rule:** Never commit trained model files to git. They are generated artifacts.

### Evaluation Standards

All regression models must report 4 metrics:

1. **RMSE** - Root Mean Squared Error (lower is better)
2. **MAE** - Mean Absolute Error (interpretable units)
3. **R²** - Coefficient of determination (0 to 1, higher is better)
4. **CCC** - Concordance Correlation Coefficient (agricultural standard, 0 to 1)

**Minimum acceptable performance:**
- R² > 0.85
- CCC > 0.80

**Rule:** If model performance is below thresholds, add more features or use ensemble methods. Never deploy low-quality models.

### SHAP Explainability

All trained models generate SHAP values for top 10 features. Include in every analytics report:

```json
{
  "feature_importance": [
    {"feature": "FOODINSEC_15_17", "importance": 0.32},
    {"feature": "POVRATE15", "importance": 0.25},
    ...
  ]
}
```

**Rule:** Explainability is not optional. Users must understand why predictions are made.

## 5. Data Integration Standards

### Database Schema

The SQLite database at `data/agriflow.db` contains 4 core tables:

- **food_environment** - USDA Food Environment Atlas (115 MO counties, 21 features)
- **food_access** - USDA Food Access Research Atlas (food deserts, low access areas)
- **nass_data** - USDA NASS crop production and yields (time series)
- **census_acs** - US Census ACS demographics and economics

**Rule:** Never modify table schemas without updating documentation. All schema changes require team approval.

### API Rate Limits

External APIs have limits. Cache responses when possible:

- **USDA NASS:** 1000 requests/day (use sparingly)
- **Census ACS:** 500 requests/day per key
- **Open-Meteo Weather:** Unlimited (no key required)
- **FEMA Disasters:** Unlimited (public API)

**Rule:** Always check local database before calling external APIs. API calls are a last resort.

### Data Freshness

- **Food Environment Atlas:** Updated annually (use 2015-2017 data)
- **Census ACS:** Updated annually (use 2022 5-year estimates)
- **NASS:** Updated weekly during growing season
- **Weather:** 7-day forecast (real-time)

**Rule:** Document data vintage in all analytics reports. Users must know data recency.

## 6. Visualization and Frontend Integration

### Chart Specifications

All charts return Plotly JSON specs compatible with React:

```json
{
  "data": [...],
  "layout": {
    "title": "...",
    "xaxis": {...},
    "yaxis": {...}
  }
}
```

**React Integration:**
```javascript
import Plot from 'react-plotly.js';
<Plot data={chart.data} layout={chart.layout} />
```

**Rule:** Test all chart specs with `plotly.graph_objects` before returning. Invalid specs crash the frontend.

### Analytics Dashboard Format

The `/api/analytics` endpoint returns structured data for dashboard rendering:

```json
{
  "model_type": "xgboost",
  "predictions": [...],
  "evaluation": {"rmse": 0.12, "r2": 0.94, "ccc": 0.91, "mae": 0.08},
  "feature_importance": [...],
  "anomalies": [...],
  "web_research": [...],
  "confidence": "high",
  "sources": ["USDA Food Atlas", "Census ACS 2022", ...],
  "charts": [...]
}
```

**Rule:** Every analytics response must include sources and confidence level. Transparency is critical for user trust.

## 7. Deployment Modes and Environment Management

### Four Deployment Options

**Local Python (Development):**
```bash
python run_agent.py
```
- Direct LangGraph execution
- Fastest iteration
- No server required
- Use for debugging and testing

**FastAPI Server (Production):**
```bash
uvicorn src.api.main:app --port 8000
```
- REST API for React frontend
- Stateless requests
- Health monitoring at `/api/health`
- Use for production deployment

**Archia Cloud (Team Collaboration):**
- Web UI at console.archia.app
- Managed infrastructure
- Team chat history
- Use for demos and stakeholder reviews

**Hybrid (Recommended):**
- Archia Console for query routing
- Local LangGraph for tool execution
- Best performance + collaboration
- Use for team development

**Rule:** Development happens locally. Staging uses FastAPI. Production uses Hybrid. Never develop directly on Archia Cloud.

### Environment Variables

Required in `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...          # Claude API access
ARCHIA_TOKEN=pat_...                   # Archia Cloud authentication
CENSUS_API_KEY=...                     # US Census API access
NASS_API_KEY=...                       # USDA NASS API access
ARCHIA_BASE_URL=https://registry.archia.app
```

**Rule:** Never commit `.env` files. Use `.env.example` as template. Rotate keys quarterly.

## 8. Testing and Validation Workflow

### Three Test Suites

**Integration Tests (test_archia_integration.py):**
- Archia API connectivity (6 tests)
- Agent deployment verification
- Skills configuration check
- Run before any Archia changes

**Complete Pipeline (test_complete_pipeline.py):**
- End-to-end workflow (7 tests)
- Agent routing
- Tool execution
- API endpoints
- Run before commits to main branch

**Analytics Agent (test_analytics_agent.py):**
- ML pipeline validation
- Model training and prediction
- SHAP explainability
- Run after any analytics tool changes

**Acceptance Criteria:** All 13 tests must pass before deployment. No exceptions.

**Rule:** Write tests for new features before implementation. Test-driven development is required.

## 9. Code Quality and Documentation Standards

### Code Style

- **Line length:** 100 characters maximum
- **Docstrings:** Google style for all functions
- **Type hints:** Required for all function signatures
- **Comments:** Explain "why", not "what"

**Example:**
```python
def build_feature_matrix(state: str) -> pd.DataFrame:
    """
    Builds ML-ready feature matrix by joining all data sources.

    Combines food_environment, census_acs, fema_history, and weather
    into a single DataFrame with 35 features per county.

    Args:
        state: Two-letter state code (e.g., "MO")

    Returns:
        DataFrame with counties as rows, features as columns
    """
```

### Documentation Updates

When making changes, update these files in order:

1. **Code** - Implement feature
2. **Tests** - Add test coverage
3. **CHANGELOG.md** - Document what changed and why
4. **README.md** - Update if user-facing feature
5. **docs/ARCHITECTURE.md** - Update if architectural change

**Rule:** No pull requests without documentation updates. Undocumented code is rejected.

### Git Workflow

**Branch naming:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation only

**Commit messages:**
```
[category] Brief description

Detailed explanation of what changed and why.

Affects: file1.py, file2.py
Tests: test_suite.py (all passing)
```

**Rule:** Squash commits before merging to main. Keep history clean.

## Summary Checklist

Before deploying any changes, verify:

- [ ] All 13 tests pass (test_complete_pipeline.py)
- [ ] CHANGELOG.md updated with version and date
- [ ] Documentation reflects current system state
- [ ] No hardcoded API keys or secrets
- [ ] Model routing uses cost-optimized strategy
- [ ] All tools assigned to correct category
- [ ] Charts tested with Plotly
- [ ] Analytics reports include sources and confidence
- [ ] Code follows style guide
- [ ] Git commits are descriptive

---

**Document Version:** 2.0.2
**Last Updated:** 2026-02-16
**Maintained By:** MUIDSI Hackathon Team
**Questions:** See docs/ARCHITECTURE.md for technical details
