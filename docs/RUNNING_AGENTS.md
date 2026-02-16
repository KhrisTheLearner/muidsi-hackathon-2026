# Running AgriFlow Agents

Three ways to run the agents: local Python, local API, or Archia cloud.

## Option 1: Local Python (Fastest)

Direct LangGraph execution - no server needed.

```python
from src.agent.graph import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent()

result = agent.invoke({
    "messages": [HumanMessage("Which Missouri counties have the highest food insecurity?")],
    "plan": [],
    "current_step": 0,
    "tool_results": {},
    "reasoning_trace": [],
    "final_answer": None,
})

print(result["final_answer"])
```

**Pros:** Fast, full control, easy debugging
**Cons:** No REST API, manual state management

## Option 2: Local FastAPI (Production)

HTTP server with REST endpoints.

```bash
# Start server
uvicorn src.api.main:app --reload --port 8000

# Test health
curl http://localhost:8000/api/health

# Query agent
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Train XGBoost on Missouri data"}'
```

**Endpoints:**
- `POST /api/query` - Send query, get response
- `GET /api/health` - System status
- `GET /api/examples` - Example queries
- `GET /api/charts` - List generated charts
- `GET /api/analytics` - List analytics reports

**Pros:** REST API, stateless, integrates with React frontend
**Cons:** Requires server, port management

## Option 3: Archia Cloud (Team Collaboration)

Managed cloud platform with web UI.

### Setup

```bash
# 1. Get API key from console.archia.app
export ARCHIA_TOKEN="your_token"

# 2. Push skills
python archia/setup_skills.py

# 3. Verify deployment
python archia/setup_skills.py --verify
```

### Usage

**Via API:**
```bash
curl -X POST https://registry.archia.app/v1/responses \
  -H "Authorization: Bearer $ARCHIA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:agriflow-analytics",
    "input": "Train XGBoost model on Missouri counties",
    "stream": false
  }'
```

**Via Web UI:**
1. Go to https://console.archia.app
2. Select agent (e.g., agriflow-analytics)
3. Type query in chat interface
4. View results + tool calls

**Pros:** Web UI, team sharing, managed hosting
**Cons:** Requires internet, Archia Desktop for local MCPs

## Example Queries

### Data Retrieval
```
"Which Missouri counties have the highest food insecurity rates?"
"Show me food desert data for St. Louis County"
"Get weather forecast for Wayne County Missouri"
```

### ML Analytics
```
"Train an XGBoost model on Missouri data and predict drought risk"
"Which features drive food insecurity predictions? Use SHAP"
"Flag counties with unusual food insecurity patterns"
```

### Visualization
```
"Create a bar chart of top 10 food insecure counties"
"Map food desert risk across southeastern Missouri"
"Show a risk heatmap of poverty vs food access"
```

### Route Optimization
```
"Plan delivery route from Cape Girardeau to Wayne, Pemiscot, and Dunklin counties"
"Calculate distance from St. Louis Food Bank to New Madrid County"
"Schedule deliveries starting at 8am with 30min loading time"
```

### Full Pipeline
```
"Run full analytics on Missouri food supply chain under drought scenario.
Include model training, predictions, SHAP, anomalies, and web research."
```

## Monitoring

### Local Logs
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### API Logs
```bash
# Server logs show tool calls and execution times
uvicorn src.api.main:app --log-level debug
```

### Archia Logs
- View in console.archia.app under agent execution history
- Shows tool calls, errors, and response times

## Performance Tips

1. **Pre-train models** - Cache XGBoost to `models/` for instant predictions
2. **Use Haiku for simple tasks** - 10x cheaper than Sonnet
3. **Batch queries** - Run multiple predictions in one call
4. **Cache results** - Store analytics reports for reuse

## Troubleshooting

**"No tools found"**
- Check `from src.agent.nodes.tool_executor import ALL_TOOLS`
- Should return 30 tools

**"Model not found"**
- Train model first with `train_risk_model`
- Check `models/` directory for cached models

**"Empty results"**
- Load data into `data/agriflow.db`
- See SETUP.md for instructions

**Archia 500 errors**
- Check workspace matches API key
- Verify agent exists: `GET /v1/agent`
- Check MCP server is running (for local tools)
