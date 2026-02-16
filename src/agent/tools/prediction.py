"""Tool for running ML risk prediction model.

Uses trained XGBoost/Random Forest models when available (from ml_engine.py),
falls back to heuristic scoring if no model is trained. The interface is
stable — callers don't need to know which backend is active.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))


def _compute_risk_score(row: dict, yield_reduction: float = 0.0) -> float:
    """Heuristic risk score — fallback when no trained model exists.

    Combines food insecurity, poverty, low access, and crop dependency
    into a 0-1 risk score. Higher = more vulnerable.
    """
    food_insec = float(row.get("food_insecurity_rate", 0) or 0) / 100
    poverty = float(row.get("poverty_rate", 0) or 0) / 100
    low_access = float(row.get("low_access_pct", 0) or 0) / 100
    snap_pct = float(row.get("snap_pct", 0) or 0) / 100

    base = (
        0.35 * food_insec
        + 0.25 * poverty
        + 0.20 * low_access
        + 0.20 * snap_pct
    )

    scenario_multiplier = 1.0 + (yield_reduction / 100) * 0.5
    return min(round(base * scenario_multiplier, 3), 1.0)


def _has_trained_model(state: str) -> bool:
    """Check if a trained ML model exists for this state."""
    if not MODEL_DIR.exists():
        return False
    return any(MODEL_DIR.glob(f"*_{state}_*_meta.json"))


@tool
def run_prediction(
    state: str = "MO",
    scenario: str = "baseline",
    yield_reduction_pct: float = 0.0,
    top_n: int = 10,
    model_type: str = "auto",
) -> list[dict[str, Any]]:
    """Run the AgriFlow risk prediction model on counties.

    Automatically uses a trained XGBoost/RF model if available, otherwise
    falls back to the heuristic scoring formula. Use train_risk_model from
    the analytics tools to train a real model first for best results.

    Args:
        state: Two-letter state code.
        scenario: Scenario name - "baseline", "drought", "price_shock".
        yield_reduction_pct: Simulated crop yield reduction (0-100).
        top_n: Number of highest-risk counties to return.
        model_type: "auto" (use ML if available), "heuristic", "xgboost", "random_forest".

    Returns:
        List of counties ranked by risk score (highest first).
    """
    if scenario == "drought" and yield_reduction_pct == 0:
        yield_reduction_pct = 20.0
    elif scenario == "price_shock" and yield_reduction_pct == 0:
        yield_reduction_pct = 10.0

    # Try ML model if requested or auto-detect
    use_ml = model_type != "heuristic" and (model_type != "auto" or _has_trained_model(state))

    if use_ml:
        try:
            from src.agent.tools.ml_engine import predict_risk
            ml_type = "xgboost" if model_type in ("auto", "xgboost") else model_type
            results = predict_risk.invoke({
                "state": state,
                "model_type": ml_type,
                "scenario": scenario,
                "yield_reduction_pct": yield_reduction_pct,
                "top_n": top_n,
            })
            if results and "error" not in results[0]:
                # Add model_backend tag so callers know which was used
                for r in results:
                    r["model_backend"] = "ml"
                    r["risk_score"] = r.get("predicted_risk", r.get("risk_score", 0))
                return results
        except Exception:
            pass  # Fall through to heuristic

    # Heuristic fallback
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT FIPS, State, County,
                   FOODINSEC_15_17 as food_insecurity_rate,
                   POVRATE15 as poverty_rate,
                   PCT_LACCESS_POP15 as low_access_pct,
                   PCT_SNAP16 as snap_pct
            FROM food_environment
            WHERE State = ?
            """,
            (state.upper(),),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return [{
            "error": "Database not found or table missing",
            "hint": "Run the data pipeline first: python -m src.data_pipeline.load_atlas",
        }]

    results = []
    for row in rows:
        row_dict = dict(row)
        score = _compute_risk_score(row_dict, yield_reduction_pct)
        results.append({
            "fips": row_dict.get("FIPS"),
            "county": row_dict.get("County"),
            "state": row_dict.get("State"),
            "risk_score": score,
            "scenario": scenario,
            "yield_reduction_pct": yield_reduction_pct,
            "food_insecurity_rate": row_dict.get("food_insecurity_rate"),
            "poverty_rate": row_dict.get("poverty_rate"),
            "model_backend": "heuristic",
        })

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return results[:top_n]
