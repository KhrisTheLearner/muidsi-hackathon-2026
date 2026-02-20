"""ML Analytics MCP Server - thin wrapper around agent/tools/ml_engine.py.

Run standalone:  python -m src.mcp_servers.ml_server
External use:    Claude Desktop, Cursor, or any MCP client via stdio

Core logic lives in src/agent/tools/ml_engine.py (single source of truth).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent.tools.ml_engine import (
    build_feature_matrix as _build,
    build_tract_feature_matrix as _build_tract,
    compute_food_insecurity_risk as _tract_risk,
    detect_anomalies as _anomalies,
    get_feature_importance as _importance,
    predict_risk as _predict,
    train_risk_model as _train,
    web_search_risks as _search,
)
from src.agent.nodes.analytics_supervisor import (
    run_analytics_pipeline as _pipeline,
)

mcp = FastMCP("AgriFlow ML Analytics")


@mcp.tool()
def train_and_predict(
    state: str = "MO",
    scenario: str = "drought",
    model_type: str = "xgboost",
    yield_reduction_pct: float = 20.0,
    top_n: int = 15,
):
    """One-shot: build features, train model, predict, evaluate, and analyze."""
    return _pipeline.invoke({
        "state": state,
        "scenario": scenario,
        "model_type": model_type,
        "yield_reduction_pct": yield_reduction_pct,
        "top_n": top_n,
        "pipeline": "full_analysis",
    })


@mcp.tool()
def train_risk_model(
    state: str = "MO",
    model_type: str = "xgboost",
    target_col: str = "FOODINSEC_21_23",
):
    """Train an XGBoost or Random Forest risk prediction model."""
    return _train.invoke({
        "state": state,
        "model_type": model_type,
        "target_col": target_col,
    })


@mcp.tool()
def predict_risk(
    state: str = "MO",
    scenario: str = "baseline",
    yield_reduction_pct: float = 0.0,
    top_n: int = 10,
):
    """Run trained model to predict county-level risk scores."""
    return _predict.invoke({
        "state": state,
        "scenario": scenario,
        "yield_reduction_pct": yield_reduction_pct,
        "top_n": top_n,
    })


@mcp.tool()
def explain_predictions(state: str = "MO", top_n: int = 10):
    """Get SHAP feature importance for a trained model."""
    return _importance.invoke({"state": state, "top_n": top_n})


@mcp.tool()
def detect_anomalies(state: str = "MO", contamination: float = 0.1):
    """Flag counties with unusual patterns using Isolation Forest."""
    return _anomalies.invoke({"state": state, "contamination": contamination})


@mcp.tool()
def search_agricultural_risks(query: str, region: str = "Missouri"):
    """Web search for emerging pests, diseases, and weather threats."""
    return _search.invoke({"query": query, "region": region})


@mcp.tool()
def build_features(state: str = "MO"):
    """Build ML-ready county feature matrix from all data sources."""
    return _build.invoke({"state": state})


@mcp.tool()
def build_tract_features(state: str = ""):
    """Build census-tract feature matrix with per-capita normalization and 3D vulnerability indices.

    Implements NEW2 notebook methodology: SNAP_rate, HUNV_rate, demographic %,
    Economic Hardship Index (EHI), Structural Inequality Index (SII), Aging Index (AI),
    and LILA access binary indicators. Filters Pop2010 < 50.
    """
    return _build_tract.invoke({"state": state})


@mcp.tool()
def compute_tract_risk(state: str = "", top_n: int = 20):
    """Compute Food Insecurity Risk scores for census tracts using GBM predictions.

    Trains GBM (n=500, lr=0.05, depth=4) on SNAP_rate, then computes:
    Food_Insecurity_Risk = (EHI_norm + Pred_SNAP_norm) / 2 ∈ [0, 1]
    Returns top-N most vulnerable tracts ranked by composite risk.
    """
    return _tract_risk.invoke({"state": state, "top_n": top_n})


if __name__ == "__main__":
    mcp.run(transport="stdio")
