# From Research Notebooks to Agentic Tools

This document explains how the machine learning methodology developed in our Jupyter notebooks was validated, extracted, and embedded into AgriFlow's agentic tool layer — making it accessible to any natural-language query through the LangGraph orchestrator.

---

## Overview

Our ML pipeline went through four stages:

```
Stage 1: Jupyter Exploration  -->  Stage 2: Methodology Validation
Stage 3: Production Tool Code -->  Stage 4: Agentic Integration
```

Each notebook produced validated methodology that was directly translated into production Python code in `src/agent/tools/ml_engine.py`, then exposed through the LangGraph tool layer so any user query can trigger ML inference on demand.

---

## The Four Notebooks

### 1. `notebooks/Suyog.ipynb` — County-Level Composite Risk Score

**Purpose**: Develop and validate a 6-feature composite risk score for 115 Missouri counties.

**Key findings embedded into production code:**

#### Feature Set (6 features, MinMax-normalized):
```python
risk_features = [
    'FOODINSEC_18_20',  # Historical food insecurity baseline
    'FOODINSEC_21_23',  # Current food insecurity (state-level aggregate)
    'POVRATE21',        # County poverty rate
    'PCT_SNAP22',       # SNAP participation rate
    'PCT_WICWOMEN16',   # WIC women participation (STRONGEST predictor)
    'LACCESS_HHNV19',   # Households with low food access + no vehicle
]
```

#### Composite Risk Score Formula:
```python
scaler = MinMaxScaler()
scaled = scaler.fit_transform(risk_df[risk_features])
risk_df['Composite_Risk_Score'] = scaled.mean(axis=1)

# Risk categories by percentile (robust to distribution)
low_thr = np.percentile(scores, 33)   # ~0.33
med_thr = np.percentile(scores, 66)   # ~0.66
risk_df['Risk_Category'] = scores.apply(
    lambda s: 'Low Risk' if s <= low_thr else
              ('High Risk' if s > med_thr else 'Medium Risk')
)
```

#### Validated Results:
- Top high-risk Missouri county: **Carter County** (risk = 0.302)
- Top cluster: High-Risk counties average 14.0% food insecurity, 18.7% poverty
- Color coding: `{'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}`

**Translated into**: `build_feature_matrix()` and `predict_risk()` in `ml_engine.py`

---

### 2. `notebooks/NEW1 2.ipynb` — Multi-Model County Comparison

**Purpose**: Systematically compare 4 regression models on county-level data with interaction feature engineering.

#### 4 Validated Interaction Features (all rank top-15 predictors):
```python
df['POVxSNAP']     = df['POVRATE21'] * df['PCT_SNAP22']
df['POVxLACCESS']  = df['POVRATE21'] * df['LACCESS_HHNV19']
df['FOODxIncome']  = df['FOODINSEC_21_23'] / df['MEDHHINC21']
df['SNAPxLACCESS'] = df['PCT_SNAP22'] * df['LACCESS_HHNV19']
```

These interaction features capture compound effects:
- `POVxSNAP`: Counties where poverty AND food assistance coincide
- `POVxLACCESS`: Poverty + transportation barriers = double vulnerability
- `FOODxIncome`: Food insecurity relative to economic capacity
- `SNAPxLACCESS`: SNAP recipients who also face geographic access barriers

#### Model Benchmark (115 MO counties, 80/20 train/test split):

| Model | R2 | RMSE | MAE | Notes |
|-------|-----|------|-----|-------|
| Gradient Boosting (GBM) | 0.998 | 0.099 | 0.008 | **Production default** |
| Random Forest | 0.987 | — | — | Non-linear baseline |
| Linear Regression | 0.983 | 0.328 | 0.049 | Interpretable baseline |
| SVR (RBF, C=10, eps=0.1) | 0.912 | 0.746 | — | Alternative |

#### K-Means Clustering (k=3, standardized features):
Cluster on: `FOODINSEC_21_23, POVRATE21, PCT_SNAP22, MEDHHINC21, LACCESS_HHNV19`

| Cluster | Profile | Food Insecurity | Poverty | Intervention |
|---------|---------|----------------|---------|--------------|
| 0 | High-Risk (~350 counties) | 14.0% | 18.7% | Direct aid: SNAP, WIC, jobs |
| 1 | Low-Risk (~1500 counties) | 11.1% | 11.0% | Monitoring only |
| 2 | Access-Constrained (~600 counties) | 12.2% | 12.9% | Infrastructure/transport |

#### Top Predictive Features (SHAP analysis):
1. `PCT_WICWOMEN16` (~65% importance) — WIC women participation, strongest single predictor
2. `VLFOODSEC_21_23` (~22%) — Historical extreme food insecurity
3. `FOODINSEC_18_20` (~5%) — Temporal baseline
4. `PCT_SNAP22` — Direct economic hardship indicator
5. `LACCESS_HHNV19` — Critical in interaction features (r~-0.004 direct but top in compound)

**Translated into**: `train_risk_model()` with `model_type` parameter, `get_feature_importance()`, `detect_anomalies()` in `ml_engine.py`

---

### 3. `notebooks/NEW2.ipynb` — Census-Tract Level ML (72k Tracts)

**Purpose**: Scale from county-level to census-tract level for finer-grained vulnerability mapping.

**Scope**: 72,531 US tracts → 72,242 after filtering (removed tracts with Pop2010 < 50)

#### Data Cleaning Pipeline:
```python
# 1. Remove >90% sparse columns (26 COVID-era 2020 columns)
high_missing = df.isnull().mean()[df.isnull().mean() > 0.90].index
df.drop(columns=high_missing, inplace=True)

# 2. Median impute remaining missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 3. Remove micro-populations (ACS noise)
df = df[df['Pop2010'] >= 50]
# Result: 72,531 -> 72,242 tracts, 147 -> 14 features
```

#### Per-Capita Normalization:
```python
pop = df['Pop2010'].replace(0, 1)  # avoid division by zero
df['SNAP_rate']    = (df['TractSNAP']     / pop).clip(upper=1.0)
df['HUNV_rate']    = (df['TractHUNV']     / pop).clip(upper=1.0)
df['Senior_pct']   = (df['TractSeniors']  / pop).clip(upper=1.0)
df['White_pct']    = (df['TractWhite']    / pop).clip(upper=1.0)
df['Black_pct']    = (df['TractBlack']    / pop).clip(upper=1.0)
df['Hispanic_pct'] = (df['TractHispanic'] / pop).clip(upper=1.0)
```

#### 3D Vulnerability Taxonomy:
```python
# Economic Hardship Index (EHI) — dominant predictor, ~70-80% feature importance
df['Economic_Hardship_Index'] = df['SNAP_rate'] + df['HUNV_rate']
# Range: [0, 2], mean = 0.089

# Structural Inequality Index (SII) — systemic/historical disadvantage
df['Structural_Inequality_Index'] = df['Black_pct'] + df['Hispanic_pct'] - df['White_pct']
# Range: [-0.997, +1.022]

# Aging Index (AI) — independent vulnerability dimension
df['Aging_Index'] = df['Senior_pct']
# Range: [0, 0.894], mean = 0.136
```

#### GBM Model Training:
```python
gbm = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=4, random_state=42
)
gbm.fit(X_train, y_train)  # target: SNAP_rate
# Result: R2 = 0.9949, RMSE = 0.0033
```

#### Composite Risk Score:
```python
scaler = MinMaxScaler()
df[['EHI_norm', 'Pred_SNAP_norm']] = scaler.fit_transform(
    df[['Economic_Hardship_Index', 'Predicted_SNAP_rate']]
)
df['Food_Insecurity_Risk'] = (df['EHI_norm'] + df['Pred_SNAP_norm']) / 2
# Score in [0, 1]: 0 = minimal risk, 1 = maximum vulnerability
```

**Translated into**: `build_tract_feature_matrix()` and `compute_food_insecurity_risk()` in `ml_engine.py`

---

### 4. `notebooks/NEW2 1.ipynb` — Tract-Level Validation (Missouri Focus)

**Purpose**: Validate the NEW2 methodology on Missouri's 1,387 census tracts and identify the highest-risk tracts.

**Key validated results**:
- Top high-risk tract: **29510111300** (St. Louis city), `Food_Insecurity_Risk = 0.959`
- GBM model on MO tracts: R2 = 0.9949, RMSE = 0.0033
- EHI is the dominant predictor (~70-80% feature importance in SHAP)
- Intervention decision logic validated:
  - EHI > threshold + SII > threshold → immediate distribution + structural reforms
  - EHI > threshold only → SNAP enrollment + economic assistance
  - AI > threshold → senior nutrition programs + delivery/mobility support

**Translated into**: The `compute_food_insecurity_risk()` tool's Missouri-specific defaults and threshold logic.

---

## From Notebook to Production Tool

The methodology from each notebook was implemented in `src/agent/tools/ml_engine.py` as LangChain `@tool`-decorated functions. Here is the mapping:

| Notebook | Validated Concept | Production Function | Agent Tool |
|----------|-------------------|---------------------|------------|
| Suyog | 6-feature composite risk score | `build_feature_matrix()` | `[analytics]` |
| Suyog | MinMax risk scoring | `predict_risk()` | `[analytics]` |
| NEW1 | 4 interaction features | `build_feature_matrix()` | `[analytics]` |
| NEW1 | GBM/RF/SVR model types | `train_risk_model(model_type=...)` | `[analytics]` |
| NEW1 | K-Means clustering | `train_risk_model()` (embedded) | `[analytics]` |
| NEW1 | SHAP feature importance | `get_feature_importance()` | `[analytics]` |
| NEW1 | Isolation Forest anomalies | `detect_anomalies()` | `[analytics]` |
| NEW2 | Per-capita normalization | `build_tract_feature_matrix()` | `[analytics]` |
| NEW2 | EHI/SII/AI vulnerability taxonomy | `build_tract_feature_matrix()` | `[analytics]` |
| NEW2 | GBM SNAP prediction | `compute_food_insecurity_risk()` | `[analytics]` |
| NEW2 1 | Missouri tract validation | `compute_food_insecurity_risk()` defaults | `[analytics]` |

---

## How the Agent Calls These Tools

When a user asks: *"Train an XGBoost model and predict which counties are most at risk under a drought scenario"*, the LangGraph agent:

1. **Router** classifies this as `[analytics]` (matches "train" + "predict" + "risk")
2. **Planner** decomposes into steps:
   ```
   [data]      Query county food atlas and poverty data for Missouri
   [analytics] Build feature matrix with interaction features
   [analytics] Train XGBoost risk model on county data
   [analytics] Predict risk under drought scenario (30% yield reduction)
   [analytics] Get SHAP feature importance
   [viz]       Create feature importance bar chart and predicted vs actual scatter
   [viz]       Create choropleth map of county risk scores
   ```
3. **Tool Executor** routes each step to the appropriate agent (Haiku for data, Sonnet for analytics)
4. **ML Engine** tools execute the exact same code validated in the notebooks
5. **Chart Validator** fixes any Plotly spec issues
6. **Synthesizer** composes the final answer citing model metrics (R2, RMSE, CCC, AUC)

### Tool Call Chain (Analytics Pipeline)

```
build_feature_matrix(state="MO")
    |-- Loads food_environment from SQLite
    |-- Loads food_access from SQLite
    |-- Computes 4 interaction features
    |-- Returns DataFrame (115 rows x 38 features)
    v
train_risk_model(state="MO", model_type="xgboost")
    |-- Calls build_feature_matrix internally
    |-- Trains XGBRegressor with 5-fold CV
    |-- Computes R2, RMSE, MAE, CCC, AUC
    |-- Generates 3 diagnostic charts:
    |     feature_importance bar chart
    |     actual_vs_predicted scatter (with R2 in title)
    |     roc_curve (binary: above-median = high-risk)
    |-- Saves model to models/ directory
    |-- Returns metrics + chart specs
    v
predict_risk(state="MO", scenario="drought", yield_reduction_pct=30)
    |-- Loads cached model
    |-- Applies scenario modifications to features
    |-- Returns ranked county predictions with confidence intervals
    v
get_feature_importance(state="MO")
    |-- Runs SHAP TreeExplainer on cached model
    |-- Returns top-15 features with importance % and interpretation
```

---

## Key Design Decisions

### Why Translate Notebooks to Agent Tools?

1. **On-demand inference**: The agent can train and run models in response to any query — no pre-built static analysis
2. **Parameterized scenarios**: Users can ask "what if corn yields drop 20%?" and the tool applies that modification dynamically
3. **Explainability built in**: SHAP values are computed automatically with every model, so the agent always explains *why* counties are high-risk
4. **Multi-model support**: The `model_type` parameter lets users compare XGBoost, GBM, Random Forest, and SVR within a single conversation

### Why LangChain `@tool` Decorator?

Each function is decorated with `@tool` from LangChain, which:
- Generates a structured JSON schema for LLM tool calling
- Validates arguments before execution
- Returns results that the LLM can reason about
- Integrates with LangGraph's `ToolNode` for automatic execution

### Synthetic Tool Call Builder

Because the Archia API can time out when tools are bound to LLM requests, we implemented a **synthetic tool call builder** in `tool_executor.py`. For tagged plan steps like `[analytics]`, the system builds tool calls directly from keyword analysis rather than relying on an LLM to generate them. This makes the pipeline more reliable:

```python
# If plan step contains "train", "xgboost", etc.:
# -> automatically calls run_analytics_pipeline or train_risk_model
# without waiting for LLM tool selection
```

---

## Evaluation and Validation

The ML pipeline is validated by `scripts/test_ml_pipeline.py` which runs 7 test sections:

1. Database connectivity (SQLite food atlas loads correctly)
2. Feature matrix construction (shape, no NaN, interaction features present)
3. Model training (R2 > 0.85, RMSE within bounds)
4. Risk prediction (non-negative scores, correct column names)
5. Feature importance (SHAP values present, top feature identified)
6. Anomaly detection (Isolation Forest runs, returns outlier flags)
7. Tract-level risk (EHI/SII/AI computed, risk in [0,1])

**All 7 sections pass** for Missouri data as of the hackathon submission.

---

## Data Notes

### Important: State vs. County Level Variables

Some USDA Food Atlas columns are **state-level aggregates** — the same value is reported for every county within a state:
- `FOODINSEC_21_23` — state average food insecurity rate
- `PCT_SNAP22` — state SNAP participation rate

For **county-level variance**, use:
- `POVRATE21` — county poverty rate
- `PCT_LACCESS_HHNV19` — county % households with low food access and no vehicle
- `PCT_WICWOMEN16` — county WIC women participation rate

The `build_feature_matrix()` function automatically handles this by including the interaction features which reveal county-level variation even when base features are state-level.
