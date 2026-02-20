# Quick Reference

## Start Agents

```bash
# Local Python
python run_agent.py

# API Server
uvicorn src.api.main:app --port 8000

# Archia Cloud
python archia/setup_skills.py
```

## Common Commands

```bash
# Test graph compilation
python -c "from src.agent.graph import create_agent; create_agent()"

# Check tools
python -c "from src.agent.nodes.tool_executor import ALL_TOOLS; print(len(ALL_TOOLS))"

# Verify Archia skills
python archia/setup_skills.py --verify

# Health check
curl http://localhost:8000/api/health
```

## API Calls

```bash
# Query agent
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which Missouri counties have highest food insecurity?"}'

# Get examples
curl http://localhost:8000/api/examples

# List charts
curl http://localhost:8000/api/charts
```

## Environment Setup

```bash
# Required
export DEFAULT_MODEL=priv-claude-sonnet-4-5-20250929

# Optional
export NASS_API_KEY=your_key
export ARCHIA_TOKEN=your_token
export DB_PATH=data/agriflow.db
export MODEL_DIR=models
```

## Tool Categories

| Category | Count | Model | Cost |
|----------|-------|-------|------|
| data | 9 | Haiku | Low |
| sql | 2 | Sonnet | High |
| ml | 4 | Sonnet | High |
| analytics | 16 | Sonnet | High |
| viz | 15 | Haiku | Low |
| route | 4 | Haiku | Low |

42 unique tools total (some overlap between groups for agent self-sufficiency).

## File Locations

```text
src/agent/graph.py                 LangGraph workflow
src/agent/tools/                   42 tool implementations
src/agent/nodes/tool_executor.py   Routing logic
src/api/main.py                    REST API endpoints
archia/push_agents.sh              Sync agents to Archia cloud
archia/deploy_archia_cloud.py      Full Archia deployment script
archia/create_analytics_agent.py   Create/update analytics agent
scripts/check_environment.py       Verify Python env + imports
scripts/demo_autonomous_behavior.py  Interactive demo runner
scripts/test_ml_pipeline.py        ML pipeline validation (7 sections)
tests/test_analytics_agent.py      Archia agent call test
tests/test_archia_integration.py   6-test integration suite
tests/test_complete_pipeline.py    End-to-end pipeline test
data/agriflow.db                   SQLite database
models/                            Cached ML models
```

## Troubleshooting

**Empty results?**
→ Load data into `data/agriflow.db`

**ModuleNotFoundError?**
→ `pip install -r requirements.txt`

**NASS API errors?**
→ Add NASS_API_KEY to .env

**Archia 401?**
→ Check token, verify workspace

**Graph compilation fails?**
→ Check imports in tool_executor.py

## Example Queries

```text
Data: "Food insecurity rates in Missouri counties"
ML: "Train XGBoost, predict drought risk, show SHAP"
Viz: "Bar chart of top 10 food insecure counties"
Route: "Optimize delivery from Cape Girardeau to Wayne, Pemiscot, Dunklin"
Full: "Full analytics on Missouri drought scenario with web research"
```
