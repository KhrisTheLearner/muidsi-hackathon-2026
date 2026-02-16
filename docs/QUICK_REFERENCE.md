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
| data | 7 | Haiku | Low |
| sql | 2 | Sonnet | High |
| ml | 4 | Sonnet | High |
| analytics | 13 | Sonnet | High |
| viz | 4 | Haiku | Low |
| route | 4 | Haiku | Low |

## File Locations

```text
src/agent/graph.py                 LangGraph workflow
src/agent/tools/                   30 tool implementations
src/agent/nodes/tool_executor.py   Routing logic
src/api/main.py                    REST API endpoints
archia/setup_skills.py             Push skills to cloud
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
