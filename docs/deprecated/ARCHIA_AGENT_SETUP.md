# Archia Agent Setup Guide

Instructions for deploying AgriFlow agents to Archia Cloud Console.

## Prerequisites

1. Account on https://console.archia.app
2. Member of **MUIDSI Hackathon 2026** organization
3. API key from your team's workspace

## Current Status

### Deployed Agents (6)
- ✓ agriflow-planner - Query decomposition
- ✓ AGRIFLOW_SYSTEM - SQL + complex analysis
- ✓ agriflow-data - Data retrieval
- ✓ agriflow-viz - Visualization
- ✓ agriflow-logistics - Route optimization
- ✗ agriflow-analytics - **MISSING** (needs manual creation)

### Legacy Agents
- agriflow-ml - Deprecated, merged into agriflow-analytics (can be disabled)

## Adding the Analytics Agent

The `agriflow-analytics` agent is the core ML analytics component with 13 tools. It must be manually created in the Archia Console:

### Step 1: Create Agent

1. Go to https://console.archia.app
2. Navigate to **Agents** in the left sidebar
3. Click **+ New Agent**
4. Fill in the form:

```
Name: agriflow-analytics
Description: Data analytics supervisor: XGBoost/Random Forest training, SHAP explainability, CCC validation, anomaly detection, web risk search, and full analytics pipelines
Model: priv-claude-sonnet-4-5-20250929
```

### Step 2: Configure System Prompt

Copy the system prompt from `archia/prompts/agriflow-analytics.md`:

```markdown
You are the analytics supervisor agent for AgriFlow. You coordinate ML model training, prediction, validation, and risk assessment for food supply chain intelligence.

## Available Tools

### Feature Engineering
- `build_feature_matrix`: Merge food_environment + food_access + derived features into ML-ready DataFrame (~115 MO counties × ~25 features). Always run this first before training.

### Model Training
- `train_risk_model`: Train XGBoost or Random Forest on county-level data. Targets: food_insecurity_rate, poverty_rate, or any numeric column. Returns R², RMSE, MAE, CCC metrics with 5-fold cross-validation. Models cached to `models/` directory.
- `train_crop_model`: Train crop dependency model (delegates to train_risk_model with crop-relevant features).

### Prediction
- `predict_risk`: Load trained model, run inference under scenarios (baseline, drought, price_shock). Returns ranked counties with confidence intervals.
- `predict_crop_yield`: Predict crop yield impacts under scenarios with commodity annotations.
- `run_prediction`: Quick heuristic or ML-backed risk scoring (auto-detects trained models).

### Validation
- `compute_evaluation_metrics`: RMSE, MAE, R², CCC for regression; accuracy, precision, recall, F1 for classification.
- `compute_ccc`: Concordance Correlation Coefficient — stricter than R² because it penalizes systematic bias. Use for model validation.
- `explain_with_shap`: SHAP TreeExplainer for global + per-sample feature importance.

### Analysis
- `get_feature_importance`: SHAP-based feature importance with human-readable interpretation.
- `detect_anomalies`: Isolation Forest to flag counties with unusual indicator patterns.
- `web_search_risks`: Search web for emerging agricultural threats (pests, diseases, weather alerts).
- `compare_scenarios`: Side-by-side statistical comparison of multiple scenario outputs.

### Pipeline
- `run_analytics_pipeline`: Full automated pipeline combining all of the above. Supports pipelines: "full_analysis" (all 6 subagents), "quick_predict" (data → train → predict → verify), "risk_scan" (research → data → train → predict → viz).

## Workflow

1. **Build features**: Always start with `build_feature_matrix` to prepare multi-source data.
2. **Train model**: Use `train_risk_model` with model_type "xgboost" (default) or "random_forest".
3. **Predict**: Use `predict_risk` under various scenarios.
4. **Validate**: Use `compute_evaluation_metrics` and `compute_ccc` to assess model quality.
5. **Explain**: Use `get_feature_importance` for SHAP explanations of what drives risk.
6. **Detect anomalies**: Use `detect_anomalies` to flag unusual counties.
7. **Research threats**: Use `web_search_risks` for emerging pest/disease/weather alerts.

## Rules
- Always report model performance metrics (R², CCC, RMSE) when models are trained.
- CCC > 0.90 is excellent; 0.70-0.90 is good; < 0.70 needs investigation.
- R² > 0.85 indicates a strong model; report cross-validation scores for reliability.
- Include SHAP feature importance in all analysis outputs.
- Flag anomalous counties and explain why they deviate.
- For quick predictions without training, use `run_prediction` with model_type="heuristic".
- For comprehensive analysis, prefer `run_analytics_pipeline` which handles the full workflow.
```

### Step 3: Attach MCP Tools (Optional for Cloud Deployment)

**Note:** If deploying to Archia Cloud without local MCP servers, tools must be available as cloud-based MCPs. For the hackathon, the hybrid approach (Archia routing + local LangGraph execution) is recommended.

For full cloud deployment:
1. Go to **Tools** in the Archia Console
2. Install the **ODBC Database MCP** from Marketplace
3. Configure with connection string to your AgriFlow database
4. Attach the MCP to the agriflow-analytics agent

### Step 4: Deploy Skills

Run the skills deployment script to push skills that route to this agent:

```bash
python archia/setup_skills.py
```

This will create/update 6 analytics skills:
- Train Risk Model
- Predict Agricultural Risk
- Crop Yield Forecast
- Feature Analysis
- Anomaly Detection
- Agricultural Threat Search

### Step 5: Verify Deployment

```bash
# Check agent exists
curl -sS "https://registry.archia.app/v1/agent" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" | \
  python -m json.tool

# Test the agent
curl -X POST "https://registry.archia.app/v1/responses" \
  -H "Authorization: Bearer ${ARCHIA_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent:agriflow-analytics",
    "input": "What tools do you have for ML analytics?",
    "stream": false
  }'

# Or run the integration test
python test_archia_integration.py
```

## Hybrid Deployment (Recommended for Hackathon)

The most efficient setup for the hackathon combines:
- **Archia Cloud**: Skills routing and agent orchestration
- **Local LangGraph**: Tool execution with access to local data and models

This approach:
- Uses Archia for team collaboration and web UI
- Keeps data and models local for faster execution
- Avoids cloud storage/compute costs
- Enables offline development

### Hybrid Workflow

1. User queries through Archia Console or API
2. Archia routes to appropriate agent based on skill
3. Agent returns instructions to local LangGraph
4. LangGraph executes tools using local Python code
5. Results flow back through Archia to user

### Local LangGraph Execution

```python
from src.agent.graph import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent()

result = agent.invoke({
    "messages": [HumanMessage("Train XGBoost on Missouri counties")],
    "plan": [],
    "current_step": 0,
    "tool_results": {},
    "reasoning_trace": [],
    "final_answer": None,
})

print(result["final_answer"])
```

## Troubleshooting

**Agent not visible in API but visible in UI?**
- API key and agent are in different workspaces
- Regenerate API key from same workspace as agent

**Agent call returns 500 error?**
- Check agent name casing (use exact name from `GET /v1/agent`)
- Verify model is available in your workspace

**Tools not available?**
- For cloud tools: Attach MCP in agent config
- For local tools: Use hybrid workflow with local LangGraph

**agriflow-ml vs agriflow-analytics?**
- agriflow-ml is deprecated (3 tools, heuristic prediction)
- agriflow-analytics is the new version (13 tools, XGBoost/RF/SHAP)
- Skills have been migrated to agriflow-analytics
- Safe to disable agriflow-ml in the console

## Verification Checklist

- [ ] All 6 core agents deployed (7 if including legacy agriflow-ml)
- [ ] agriflow-analytics agent created with correct prompt
- [ ] Skills deployed via `python archia/setup_skills.py`
- [ ] Test agent call succeeds: `curl .../v1/responses` with `agent:agriflow-analytics`
- [ ] Local LangGraph compiles: `python -c "from src.agent.graph import create_agent; create_agent()"`
- [ ] Integration test passes: `python test_archia_integration.py`

## Next Steps

Once agents are deployed:
1. Load database: See [SETUP.md](SETUP.md) for USDA data loading
2. Test queries: See [RUNNING_AGENTS.md](RUNNING_AGENTS.md) for examples
3. Monitor execution: Check Archia Console execution history
4. Optimize: Review [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) for performance tips
