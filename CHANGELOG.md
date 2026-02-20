# Changelog

All notable changes to AgriFlow.

## [2.1.0] - 2026-02-20

### Added — Web Search, Choropleth Maps, Tract-Level ML, Data MCP

#### Web Search Integration

- **DuckDuckGo DDGS** — real web search replacing Wikipedia-only instant-answer API
- `search_web` — general web search tool in DATA_TOOLS
- `search_agricultural_news` — targeted agricultural news/threat search in DATA_TOOLS
- `src/mcp_servers/data_server.py` — new FastMCP server exposing all live APIs + web search to Archia cloud agents (thin wrapper; business logic stays in `src/agent/tools/`)
- Router expanded: `disease`, `pest`, `outbreak`, `news`, `alert`, `tar spot`, `aphid`, `rootworm`, `avian flu` patterns route to `[data]`
- Synthetic tool call builder now auto-generates `search_agricultural_news` calls for disease/pest/news steps without waiting for LLM tool selection

#### Visualization Enhancements

- **Choropleth maps** — `create_choropleth_map` with FIPS code support and US county/state outlines
- 4 new ML chart types in `create_chart`: `roc_curve`, `actual_vs_predicted`, `feature_importance`, `correlation_matrix`
- **Chart validator node** — pure Python post-processing: fixes swapped axes, raw column names, FIPS formatting, empty traces (no LLM cost)

#### Tract-Level ML Tools (NEW2 notebook methodology)

- `build_tract_feature_matrix` — per-capita normalization, EHI/SII/AI vulnerability taxonomy for 72k US census tracts
- `compute_food_insecurity_risk` — GBM SNAP prediction + composite risk score (validated: R2=0.9949, top tract 29510111300 St. Louis city, risk=0.959)
- `run_eda_pipeline` — automated exploratory data analysis pipeline

#### Ingest Tools (5 new)

- `list_db_tables`, `fetch_and_profile_csv`, `load_dataset`, `run_eda_query`, `drop_table`

### Updated

- `src/agent/nodes/tool_executor.py` — DATA_TOOLS 7→9, VIZ_TOOLS 4→15, ANALYTICS_TOOLS 13→16; new INGEST_TOOLS group
- `archia/agents/agriflow-data.toml` and `agriflow-system.toml` — added `agriflow-data` MCP grant for live API access
- `archia/prompts/agriflow-data.md` — documented web search tools + DuckDuckGo citation format
- `archia/prompts/agriflow-viz.md` — added 4 new ML chart types
- `archia/prompts/agriflow-system.md` — added web search + new chart type guidance
- `requirements.txt` — added `duckduckgo-search>=6.0.0`

### Documentation Overhaul

- **README.md** — complete rewrite for MUIDSI Hackathon 2026 GitHub publication; architecture diagram, ML methodology tables, 30+ tool inventory, all agent configs
- **docs/NOTEBOOKS_TO_AGENT.md** — new traceability doc: Suyog/NEW1 2/NEW2/NEW2 1 notebooks → `ml_engine.py` → agent tools; includes exact formulas and validated results
- **docs/AGRIFLOW_ARCHITECTURE.md** — complete rewrite: 9-node graph, routing logic, multi-agent table, all 42 tools, Archia agents, MCP servers, SSE streaming, React frontend
- **docs/deprecated/** — archived 4 old docs: MASTERDOC.md, ARCHITECTURE.md, SETUP_AND_USAGE.md, TEAM_INSTRUCTIONS.md
- **.gitignore** — added: server_log.txt, archia website docs, evaluations.jsonl, models artifacts, NUL, .claude/settings.local.json

### Codebase Organization

- `tests/` — test files moved here: `test_analytics_agent.py`, `test_archia_integration.py`, `test_complete_pipeline.py`
- `archia/` — deployment scripts moved here: `create_analytics_agent.py`, `deploy_archia_cloud.py`
- `scripts/` — diagnostic/utility scripts moved here: `check_environment.py`, `demo_autonomous_behavior.py`
- `push_agents.sh` — root copy removed (canonical copy lives in `archia/push_agents.sh`)

### Fixed

- Stale tool counts in `tests/test_complete_pipeline.py` (data: 7→9, analytics: 13→16, viz: 4→15, ALL_TOOLS: 30→42)
- `docs/QUICK_REFERENCE.md` and `docs/RUNNING_AGENTS.md` updated with correct counts and new file paths

### Tool Count Summary (current)

| Category | Count |
| -------- | ----- |
| data | 9 |
| sql | 2 |
| ml | 4 |
| analytics | 16 |
| viz | 15 |
| route | 4 |
| ingest | 5 |
| **ALL_TOOLS (unique)** | **42** |

## [2.0.2] - 2026-02-16

### Documentation Reorganization

- **Created ARCHITECTURE.md** - Comprehensive system design and agentic pipeline documentation
- **Created TEAM_INSTRUCTIONS.md** - Team coordination guidelines based on design principles
- **Reorganized docs/** - Moved deprecated/historical docs to docs/deprecated/
- **Created docs/references/** - Moved Archia API reference to separate folder
- **Updated README.md** - Streamlined with current architecture and test results
- **Removed redundancies** - Eliminated duplicate or outdated documentation

### Final Documentation Structure

- docs/ARCHITECTURE.md - Complete system design & agentic pipeline
- docs/SETUP.md - Installation and database setup
- docs/RUNNING_AGENTS.md - How to run agents (4 deployment modes)
- docs/QUICK_REFERENCE.md - Command cheat sheet
- docs/DEPLOYMENT.md - Production deployment guide
- docs/ARCHIA_AGENT_SETUP.md - Archia Cloud configuration
- docs/deprecated/ - Historical audit documents (4 files)
- docs/references/ - Archia API reference documentation

### Documentation Coverage

- System Architecture - Agentic pipeline, agent routing, tool categories
- Data Flow Examples - 3 detailed examples (simple query, ML analytics, visualization)
- Deployment Modes - 4 modes (Local, FastAPI, Archia Cloud, Hybrid)
- Performance Metrics - Latency, cost analysis, throughput
- Technology Stack - Complete tech stack with versions
- Testing - All test suites documented (13/13 passing)
- Security - Privacy and authentication notes

## [2.0.1] - 2026-02-16

### Deployment & Testing

- **Archia Cloud fully configured** - All 7 agents deployed programmatically
- **agriflow-analytics agent** created via API (/v1/agent/config)
- **26 skills deployed** to Archia Cloud (25 created, 1 existing)
- **Complete pipeline tests** passing (7/7 tests, 100%)

### New Test Suites

- `test_archia_integration.py` - 6 integration tests (Archia API, agents, skills, local graph)
- `test_complete_pipeline.py` - 7 end-to-end tests (routing, execution, tools, DB simulation, API, ML, deployment modes)
- `test_analytics_agent.py` - Analytics agent verification
- `deploy_archia_cloud.py` - Automated deployment script
- `create_analytics_agent.py` - Programmatic agent creation via API

### Documentation

- `docs/ARCHIA_AGENT_SETUP.md` - Agent deployment guide
- `ARCHIA_INTEGRATION_REPORT.md` - Complete integration status report

### Production Readiness

- [x] All agents deployed on Archia Cloud
- [x] All skills deployed (26 total)
- [x] Local LangGraph verified (30 tools, 6 categories)
- [x] Hybrid architecture tested (Archia + Local)
- [x] FastAPI endpoints verified for React frontend
- [x] ML analytics pipeline structure validated
- [ ] Database loading (user action required)

## [2.0.0] - 2026-02-16

### Added - ML Analytics Agent
- **XGBoost & Random Forest** training with 5-fold cross-validation
- **SHAP explainability** for feature importance (TreeExplainer)
- **CCC validation** (Concordance Correlation Coefficient)
- **Isolation Forest** anomaly detection for outlier counties
- **DuckDuckGo web search** for emerging agricultural threats
- **Analytics pipeline** with 6 specialized subagents

### New Tools (13)
- `build_feature_matrix` - Feature engineering from multi-source data
- `train_risk_model` - XGBoost/RF training with cross-validation
- `predict_risk` - Inference with confidence intervals
- `train_crop_model` - Crop dependency modeling
- `predict_crop_yield` - Yield impact predictions
- `get_feature_importance` - SHAP-based feature analysis
- `detect_anomalies` - Isolation Forest outlier detection
- `web_search_risks` - DuckDuckGo threat search
- `run_analytics_pipeline` - Full automated pipeline
- `compute_ccc` - CCC metric computation
- `explain_with_shap` - SHAP explanations for any model

### New Files
- `src/agent/tools/ml_engine.py` - Core ML implementation (~500 lines)
- `src/agent/nodes/analytics_supervisor.py` - Pipeline orchestration (~260 lines)
- `src/mcp_servers/ml_server.py` - MCP wrapper for ML tools
- `archia/agents/agriflow-analytics.toml` - Analytics agent config
- `archia/prompts/agriflow-analytics.md` - Analytics agent prompt
- `archia/tools/agriflow-ml.toml` - ML MCP server config

### Updated
- `src/agent/tools/evaluation.py` - Added CCC + SHAP tools
- `src/agent/tools/prediction.py` - Auto ML/heuristic fallback
- `src/agent/nodes/tool_executor.py` - Added [analytics] routing
- `src/agent/prompts/system.py` - ML analytics workflows
- `src/api/main.py` - Analytics endpoint + examples
- `archia/setup_skills.py` - 6 new analytics skills
- `archia/skills.json` - v2.0.0 with 26 total skills
- `requirements.txt` - xgboost, shap, joblib

### Deprecated
- `agriflow-ml` agent - Merged into agriflow-analytics (marked disabled)

### Fixed
- FEMA state coverage expanded from 10 to 51 states
- SQL query docstrings corrected
- Route optimizer docstrings corrected
- Removed unused ANALYST_MODEL alias from llm.py
- Removed unused json import from prediction.py

### Cleanup (Codebase Audit)
- **Removed redundant files** (~40KB freed):
  - `SETUP.md` (duplicate of docs/SETUP.md)
  - `src/CHANGELOG.md` (duplicate of root CHANGELOG.md)
  - `src/agent/README.md` (info moved to main README)
  - `TEAM_STATUS.md` (hackathon internal, not production)
  - `data/raw/README.md` (empty placeholder)
  - `archia/setup_agents.py` (superseded by setup_skills.py + TOML configs)
- **Removed empty directories**:
  - `src/data_pipeline/` (unused placeholder)
  - `src/features/` (unused placeholder)
  - `src/models/` (unused placeholder, conflicts with models/ at root)
  - `src/visualization/` (unused placeholder)
- **Removed empty __init__.py files** (6 files):
  - `src/__init__.py`
  - `src/api/__init__.py`
  - `src/mcp_servers/__init__.py`
  - `src/agent/prompts/__init__.py`
- **Documentation reorganized**:
  - Created `docs/` folder for all documentation
  - Moved DEPLOYMENT.md and ARCHITECTURE_AUDIT.md to docs/
  - Added docs/SETUP.md, docs/RUNNING_AGENTS.md, docs/QUICK_REFERENCE.md
- **Code metrics** (post-cleanup):
  - Python code: 3,977 lines (from 3,463 core + new analytics)
  - 30 tools across 6 categories
  - 6 active agents (1 deprecated)
  - Documentation: 6 guides in docs/ folder

## [1.0.0] - 2026-02-15

### Added - Initial Release
- LangGraph multi-agent workflow (planner → tools → synthesizer)
- 19 core tools across 5 categories
- Archia integration with 20 skills
- FastAPI REST server
- 3 MCP servers (SQLite, Charts, Routes)

### Core Tools (19)
**Data (7):**
- query_food_atlas, query_food_access, query_nass
- query_weather, query_fema_disasters, query_census_acs
- run_prediction (heuristic)

**SQL (2):**
- list_tables, run_sql_query

**ML (2):**
- compute_evaluation_metrics, compare_scenarios

**Viz (4):**
- create_bar_chart, create_line_chart
- create_scatter_map, create_risk_heatmap

**Route (4):**
- optimize_delivery_route, calculate_distance
- create_route_map, schedule_deliveries

### Agents
- agriflow-planner - Query decomposition
- AGRIFLOW_SYSTEM - SQL + complex analysis
- agriflow-data - USDA/Census/FEMA data
- agriflow-viz - Plotly charts
- agriflow-logistics - Route optimization
- agriflow-ml - Risk prediction (now deprecated)

### Infrastructure
- LangGraph StateGraph with conditional edges
- Multi-model routing (Haiku vs Sonnet 4.5)
- MCP server framework (stdio transport)
- Archia cloud integration
- FastAPI REST endpoints

## [0.1.0] - 2026-02-14

### Added - Project Setup
- Initial project structure
- Frontend mockup (React)
- README and basic documentation
- Git repository initialization
