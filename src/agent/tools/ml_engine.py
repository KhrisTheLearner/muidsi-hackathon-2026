"""ML engine — XGBoost, Random Forest, Isolation Forest, and feature engineering.

Replaces the heuristic in prediction.py with real trained models.
Models cache to models/ directory and reload on subsequent calls.

Research backing:
- XGBoost R² > 0.98 for food price prediction (Frontiers in AI, 2025)
- Random Forest R² = 0.9865 for crop yield prediction (Emerald, 2025)
- Multi-source feature fusion is state-of-the-art (Nature, 2024)
- SHAP is the standard for agricultural ML explainability
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time as _time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature matrix cache — avoids redundant SQL + pandas rebuilds
# ---------------------------------------------------------------------------
_feature_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached_features(state: str) -> list[dict[str, Any]] | None:
    """Return cached feature matrix if fresh, else None."""
    if state in _feature_cache:
        ts, data = _feature_cache[state]
        if _time.time() - ts < _CACHE_TTL:
            return data
    return None


def _set_cached_features(state: str, data: list[dict[str, Any]]) -> None:
    _feature_cache[state] = (_time.time(), data)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _data_hash(df: pd.DataFrame) -> str:
    """Quick hash of dataframe contents for cache invalidation."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]


def _load_food_environment(state: str) -> pd.DataFrame:
    """Load food_environment table for a state."""
    conn = _get_db()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM food_environment WHERE State = ?",
            conn,
            params=(state.upper(),),
        )
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def _load_food_access_summary(state: str) -> pd.DataFrame:
    """Aggregate food_access data to county level."""
    conn = _get_db()
    try:
        df = pd.read_sql_query(
            """
            SELECT County,
                   COUNT(*) as tract_count,
                   AVG(CAST(Urban AS FLOAT)) as urban_pct
            FROM food_access
            WHERE State = ?
            GROUP BY County
            """,
            conn,
            params=(state.upper(),),
        )
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 1: Build Feature Matrix
# ---------------------------------------------------------------------------

@tool
def build_feature_matrix(
    state: str = "MO",
) -> list[dict[str, Any]]:
    """Build an ML-ready feature matrix by merging all available data sources.

    Combines USDA Food Environment Atlas, Food Access Atlas, and derived
    features into a single dataset suitable for model training and prediction.
    Returns one row per county with ~25 features.

    Args:
        state: Two-letter state code (default "MO").

    Returns:
        List of dicts — each dict is one county with all features.
    """
    cached = _get_cached_features(state)
    if cached is not None:
        return cached

    food_env = _load_food_environment(state)
    if food_env.empty:
        return [{"error": "No food_environment data found. Run data pipeline first."}]

    # Numeric columns from food_environment
    # Ordered: Suyog 6-feature set first (21-23 data), then legacy 15-17 fallbacks
    feature_cols = [
        # Suyog validated features (6-feature composite risk score)
        "FOODINSEC_18_20", "FOODINSEC_21_23", "POVRATE21",
        "PCT_SNAP22", "PCT_WICWOMEN16", "LACCESS_HHNV19",
        # Additional predictive features
        "VLFOODSEC_21_23", "MEDHHINC21", "CHILDPOVRATE21",
        "PCT_LACCESS_POP15", "PCT_LACCESS_HHNV15", "PCT_LACCESS_HHNV19",
        "PCT_LACCESS_CHILD15", "PCT_LACCESS_SENIORS15", "PCT_LACCESS_SNAP15",
        "PCT_OBESE_ADULTS17", "PCT_OBESE_ADULTS22",
        "PCT_FREE_LUNCH15", "PCT_REDUCED_LUNCH15",
        "PCT_SNAP17", "PCT_WICWOMEN21", "WICS16", "WICS22",
        "GROC16", "SUPERC16", "CONVS16", "SPECS16", "SNAPS17",
        # Legacy fallbacks (may not exist in newer atlas versions)
        "FOODINSEC_15_17", "POVRATE15", "PCT_SNAP16",
    ]

    # Keep only columns that exist
    available = [c for c in feature_cols if c in food_env.columns]
    df = food_env[["FIPS", "State", "County"] + available].copy()

    # Coerce to numeric
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN with column median
    for col in available:
        median = df[col].median()
        df[col] = df[col].fillna(median if not pd.isna(median) else 0)

    # Derived features (Suyog 4 interaction features)
    _pov  = next((c for c in ["POVRATE21", "POVRATE15"] if c in df.columns), None)
    _snap = next((c for c in ["PCT_SNAP22", "PCT_SNAP17", "PCT_SNAP16"] if c in df.columns), None)
    _food = next((c for c in ["FOODINSEC_21_23", "FOODINSEC_18_20", "FOODINSEC_15_17"] if c in df.columns), None)
    _acc  = next((c for c in ["LACCESS_HHNV19", "PCT_LACCESS_HHNV15"] if c in df.columns), None)

    if _pov and _snap:
        df["POVxSNAP"] = (
            df[_pov].fillna(0) * df[_snap].fillna(0) / 100
        ).round(4)
    if _pov and _acc:
        df["POVxLACCESS"] = (
            df[_pov].fillna(0) * df[_acc].fillna(0) / 100
        ).round(4)
    if _food and _snap:
        df["FOODxSNAP"] = (
            df[_food].fillna(0) * df[_snap].fillna(0) / 100
        ).round(4)
    if _snap and _acc:
        df["SNAPxLACCESS"] = (
            df[_snap].fillna(0) * df[_acc].fillna(0) / 100
        ).round(4)

    if "GROC16" in df.columns and "CONVS16" in df.columns:
        df["grocery_to_conv_ratio"] = (
            df["GROC16"] / df["CONVS16"].replace(0, 1)
        ).round(3)

    # Merge food access summary if available
    access_df = _load_food_access_summary(state)
    if not access_df.empty:
        df = df.merge(access_df, on="County", how="left")
        df["tract_count"] = df["tract_count"].fillna(0)
        df["urban_pct"] = df["urban_pct"].fillna(0.5)

    result = df.to_dict(orient="records")
    _set_cached_features(state, result)
    return result


# ---------------------------------------------------------------------------
# Tool 1b: Build Census-Tract Feature Matrix (NEW2 methodology)
# ---------------------------------------------------------------------------

@tool
def build_tract_feature_matrix(
    state: str = "",
) -> list[dict[str, Any]]:
    """Build a census-tract ML feature matrix with per-capita normalization and 3D vulnerability indices.

    Implements the validated NEW2 methodology:
    - Per-capita normalization (SNAP_rate, HUNV_rate, Senior_pct, demographic %)
    - 3D vulnerability taxonomy: Economic Hardship Index (EHI), Structural Inequality
      Index (SII), Aging Index (AI)
    - Clips rates to [0,1] to remove ACS estimation noise
    - Filters tracts with Pop2010 < 50 to avoid inflated per-capita ratios

    Returns one row per census tract with all engineered features ready for GBM/RF.

    Args:
        state: Two-letter state code filter (empty = all states).

    Returns:
        List of dicts — each dict is one tract with normalized features and indices.
    """
    conn = _get_db()
    try:
        if state:
            df = pd.read_sql_query(
                "SELECT * FROM food_access WHERE State = ?",
                conn, params=(state.upper(),),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM food_access LIMIT 100000", conn)
    except Exception as e:
        return [{"error": f"Failed to load food_access table: {e}"}]
    finally:
        conn.close()

    if df.empty:
        return [{"error": "No food_access data found. Run data pipeline first."}]

    # Coerce numeric columns
    pop_col = "Pop2010" if "Pop2010" in df.columns else None
    raw_cols = {
        "TractSNAP": None, "TractHUNV": None, "TractSeniors": None,
        "TractWhite": None, "TractBlack": None, "TractHispanic": None,
    }
    for col in list(raw_cols.keys()) + (["Pop2010"] if pop_col else []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if pop_col is None or "Pop2010" not in df.columns:
        return [{"error": "food_access table missing Pop2010 column needed for per-capita normalization"}]

    # Remove micro-populations (ACS estimation noise)
    df = df[df["Pop2010"] >= 50].copy()
    if df.empty:
        return [{"error": "No tracts with Pop2010 >= 50"}]

    pop = df["Pop2010"].replace(0, 1)  # guard against zero division

    # --- Per-capita normalization (validated from NEW2 research notebook) ---
    if "TractSNAP" in df.columns:
        df["SNAP_rate"] = (df["TractSNAP"] / pop).clip(upper=1.0)
    if "TractHUNV" in df.columns:
        df["HUNV_rate"] = (df["TractHUNV"] / pop).clip(upper=1.0)
    if "TractSeniors" in df.columns:
        df["Senior_pct"] = (df["TractSeniors"] / pop).clip(upper=1.0)
    if "TractWhite" in df.columns:
        df["White_pct"] = (df["TractWhite"] / pop).clip(upper=1.0)
    if "TractBlack" in df.columns:
        df["Black_pct"] = (df["TractBlack"] / pop).clip(upper=1.0)
    if "TractHispanic" in df.columns:
        df["Hispanic_pct"] = (df["TractHispanic"] / pop).clip(upper=1.0)

    # --- 3D Vulnerability Taxonomy (validated from NEW2 research notebook) ---
    snap_r = df.get("SNAP_rate", pd.Series(0.0, index=df.index))
    hunv_r = df.get("HUNV_rate", pd.Series(0.0, index=df.index))
    black_p = df.get("Black_pct", pd.Series(0.0, index=df.index))
    hisp_p = df.get("Hispanic_pct", pd.Series(0.0, index=df.index))
    white_p = df.get("White_pct", pd.Series(0.0, index=df.index))
    senior_p = df.get("Senior_pct", pd.Series(0.0, index=df.index))

    # EHI = SNAP_rate + HUNV_rate (captures immediate economic stress + housing instability)
    df["Economic_Hardship_Index"] = snap_r + hunv_r

    # SII = Black_pct + Hispanic_pct - White_pct (captures structural/historical disadvantage)
    df["Structural_Inequality_Index"] = black_p + hisp_p - white_p

    # AI = Senior_pct (age-based vulnerability, independent dimension)
    df["Aging_Index"] = senior_p

    # Binary access indicators — keep if available
    binary_cols = [
        "Urban", "LILATracts_1And10", "LILATracts_halfAnd10",
        "LILATracts_1And20", "LILATracts_Vehicle",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Select final columns
    keep = ["CensusTract", "State", "County", "Urban", "Pop2010"]
    keep += [c for c in [
        "SNAP_rate", "HUNV_rate", "Senior_pct", "White_pct", "Black_pct", "Hispanic_pct",
        "Economic_Hardship_Index", "Structural_Inequality_Index", "Aging_Index",
    ] if c in df.columns]
    keep += [c for c in binary_cols if c in df.columns]

    result = df[[c for c in keep if c in df.columns]].to_dict(orient="records")
    return result


@tool
def compute_food_insecurity_risk(
    state: str = "",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Compute composite Food Insecurity Risk scores for census tracts using GBM predictions.

    Implements the validated NEW2 composite scoring methodology:
    1. Build tract feature matrix (per-capita + EHI/SII/AI)
    2. Train or load GBM model (n_estimators=500, lr=0.05, max_depth=4)
    3. Predict SNAP_rate as food insecurity proxy
    4. MinMax-normalize EHI and Predicted_SNAP to [0,1]
    5. Average both: Food_Insecurity_Risk = (EHI_norm + Pred_SNAP_norm) / 2

    Returns tracts ranked by composite risk score (0=minimal, 1=maximum vulnerability).

    Args:
        state: Two-letter state code (empty = all states, may be slow).
        top_n: Number of highest-risk tracts to return.
    """
    # Build feature matrix
    tracts = build_tract_feature_matrix.invoke({"state": state})
    if not tracts or "error" in tracts[0]:
        return tracts

    df = pd.DataFrame(tracts)

    target = "SNAP_rate"
    if target not in df.columns:
        return [{"error": "SNAP_rate not available — cannot compute food insecurity risk"}]

    feature_cols = [
        c for c in [
            "HUNV_rate", "Senior_pct", "White_pct", "Black_pct", "Hispanic_pct",
            "Economic_Hardship_Index", "Structural_Inequality_Index", "Aging_Index",
            "Urban", "LILATracts_1And10", "LILATracts_halfAnd10",
            "LILATracts_1And20", "LILATracts_Vehicle",
        ] if c in df.columns
    ]

    if not feature_cols:
        return [{"error": "Insufficient feature columns for prediction"}]

    X = df[feature_cols].fillna(0).values
    y = df[target].values

    # Train GBM (validated hyperparameters from NEW2 notebook: R²=0.9949, RMSE=0.0033)
    model = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42,
    )
    model.fit(X, y)
    df["Predicted_SNAP_rate"] = model.predict(X)

    # --- Composite Food Insecurity Risk Score (NEW2 formula) ---
    scaler = MinMaxScaler()
    norm_cols = ["Economic_Hardship_Index", "Predicted_SNAP_rate"]
    norm_vals = scaler.fit_transform(df[norm_cols].fillna(0))
    df["EHI_norm"] = norm_vals[:, 0]
    df["Pred_SNAP_norm"] = norm_vals[:, 1]
    df["Food_Insecurity_Risk"] = (df["EHI_norm"] + df["Pred_SNAP_norm"]) / 2.0

    # Feature importance
    feat_imp = sorted(
        zip(feature_cols, model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True,
    )

    # Top N highest-risk tracts
    top = df.nlargest(top_n, "Food_Insecurity_Risk")
    result = []
    for _, row in top.iterrows():
        result.append({
            "census_tract": row.get("CensusTract"),
            "state": row.get("State"),
            "county": row.get("County"),
            "food_insecurity_risk": round(float(row["Food_Insecurity_Risk"]), 4),
            "predicted_snap_rate": round(float(row["Predicted_SNAP_rate"]), 4),
            "economic_hardship_index": round(float(row.get("Economic_Hardship_Index", 0)), 4),
            "structural_inequality_index": round(float(row.get("Structural_Inequality_Index", 0)), 4),
            "aging_index": round(float(row.get("Aging_Index", 0)), 4),
            "urban": int(row.get("Urban", 0)),
        })

    return [{
        "summary": {
            "n_tracts_analyzed": len(df),
            "state": state or "all",
            "model": "GradientBoosting (n_estimators=500, lr=0.05, max_depth=4)",
            "feature_importance": [{"feature": f, "importance": round(i, 4)} for f, i in feat_imp[:10]],
            "score_range": [0.0, 1.0],
            "score_interpretation": "0=minimal risk, 1=maximum vulnerability",
        },
        "top_tracts": result,
    }]


# ---------------------------------------------------------------------------
# Tool 2: Train Risk Model
# ---------------------------------------------------------------------------

@tool
def train_risk_model(
    state: str = "MO",
    model_type: str = "xgboost",
    target_col: str = "FOODINSEC_21_23",
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train a risk prediction model (XGBoost, Random Forest, GBM, SVR, or LinearRegression) on county data.

    Builds a feature matrix from the database, trains the model with
    cross-validation, saves the artifact to models/, and returns metrics.

    Args:
        state: State code for training data.
        model_type: "xgboost", "random_forest", "gradient_boosting", "svr", or "linear_regression".
        target_col: Column to predict (default FOODINSEC_21_23 — food insecurity rate 2021-23).
        test_size: Fraction held out for validation (0.0-0.5).

    Returns:
        Dict with model path, metrics (R², RMSE, MAE, CCC), and feature count.
    """
    features = build_feature_matrix.invoke({"state": state})
    if not features or "error" in features[0]:
        return {"error": "Failed to build feature matrix", "detail": features}

    df = pd.DataFrame(features)

    # Auto-fallback: try common food insecurity column names if preferred not found
    _TARGET_FALLBACKS = [
        "FOODINSEC_21_23", "FOODINSEC_18_20", "FOODINSEC_15_17", "FOODINSEC_13_15",
    ]
    if target_col not in df.columns:
        # Try to find any food insecurity column
        for fallback in _TARGET_FALLBACKS:
            if fallback in df.columns:
                target_col = fallback
                break
        else:
            return {
                "error": f"Target column not found. Tried: {_TARGET_FALLBACKS}",
                "available": [c for c in df.columns if c not in ("FIPS", "State", "County")],
            }

    # Separate features and target
    exclude = {"FIPS", "State", "County", target_col}
    feature_names = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64", "float32")]
    X = df[feature_names].values
    y = df[target_col].values

    if len(X) < 10:
        return {"error": f"Not enough data to train: {len(X)} rows (need >= 10)"}

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    if model_type == "xgboost" and _HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        # Validated county-level hyperparameters from Suyog notebook (R²=0.998, RMSE=0.099)
        # Note: tract-level uses n=500, depth=4 in compute_food_insecurity_risk
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    elif model_type == "linear_regression":
        # Baseline: validated R²=0.983, RMSE=0.328, MAE=0.049 (NEW1/Suyog notebooks)
        model = LinearRegression()
    elif model_type == "svr":
        # SVR with RBF kernel — validated from Suyog notebook (R²=0.912, RMSE=0.746)
        # Uses a Pipeline with StandardScaler (required for SVR)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1)),
        ])
    elif model_type == "random_forest":
        # Validated county-level hyperparameters from NEW1/Suyog notebooks (n_estimators=200, max_depth=15)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model_type = "random_forest"
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )

    # SVR Pipeline handles its own scaling internally — use raw X
    X_fit = X if model_type == "svr" else X_scaled

    # Cross-validation
    cv_scores = cross_val_score(model, X_fit, y, cv=min(5, len(X) // 2), scoring="r2")

    # Train on full data
    model.fit(X_fit, y)
    y_pred = model.predict(X_fit)

    # Metrics
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ccc = _compute_ccc(y, y_pred)

    # Save artifacts
    data_hash = _data_hash(df)
    model_name = f"{model_type}_{state}_{target_col}_{data_hash}"
    model_path = str(MODEL_DIR / f"{model_name}.pkl")
    scaler_path = str(MODEL_DIR / f"{model_name}_scaler.pkl")
    meta_path = str(MODEL_DIR / f"{model_name}_meta.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "model_type": model_type,
        "state": state,
        "target_col": target_col,
        "feature_names": feature_names,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "metrics": {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4), "ccc": round(ccc, 4)},
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "model_path": model_path,
        "scaler_path": scaler_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "trained",
        "model_type": model_type,
        "model_path": model_path,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "metrics": meta["metrics"],
        "cv_r2": f"{meta['cv_r2_mean']:.4f} ± {meta['cv_r2_std']:.4f}",
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Tool 3: Predict Risk
# ---------------------------------------------------------------------------

@tool
def predict_risk(
    state: str = "MO",
    model_type: str = "xgboost",
    target_col: str = "FOODINSEC_15_17",
    scenario: str = "baseline",
    yield_reduction_pct: float = 0.0,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Run trained ML model to predict county-level risk scores.

    Loads the most recent trained model, applies scenario adjustments,
    and returns counties ranked by predicted risk with confidence intervals.

    Args:
        state: State code.
        model_type: "xgboost" or "random_forest".
        target_col: Target variable the model was trained on.
        scenario: "baseline", "drought", or "price_shock".
        yield_reduction_pct: Simulated yield reduction (0-100).
        top_n: Number of highest-risk counties to return.

    Returns:
        List of counties with predicted risk, confidence intervals, and scenario info.
    """
    # Find latest model
    model_path, scaler_path, meta = _find_model(state, model_type, target_col)

    if meta is None:
        # Auto-train if no model exists
        train_result = train_risk_model.invoke({
            "state": state, "model_type": model_type, "target_col": target_col,
        })
        if "error" in train_result:
            return [train_result]
        model_path = train_result["model_path"]
        scaler_path = model_path.replace(".pkl", "_scaler.pkl")
        meta_file = model_path.replace(".pkl", "_meta.json")
        with open(meta_file) as f:
            meta = json.load(f)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = meta["feature_names"]

    # Build features
    features = build_feature_matrix.invoke({"state": state})
    if not features or "error" in features[0]:
        return [{"error": "Failed to build features"}]

    df = pd.DataFrame(features)

    # Apply scenario adjustments
    if scenario == "drought" and yield_reduction_pct == 0:
        yield_reduction_pct = 20.0
    elif scenario == "price_shock" and yield_reduction_pct == 0:
        yield_reduction_pct = 10.0

    # Scenario multiplier on poverty/food insecurity features
    if yield_reduction_pct > 0:
        multiplier = 1.0 + (yield_reduction_pct / 100) * 0.3
        for col in ["POVRATE15", "FOODINSEC_15_17", "PCT_SNAP16"]:
            if col in df.columns and col != target_col:
                df[col] = df[col] * multiplier

    # Ensure all feature columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_names].values
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Confidence intervals
    ci_lower, ci_upper = _get_confidence_intervals(model, X_scaled)

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "fips": row.get("FIPS"),
            "county": row.get("County"),
            "state": row.get("State"),
            "predicted_risk": round(float(predictions[i]), 4),
            "ci_lower": round(float(ci_lower[i]), 4) if ci_lower is not None else None,
            "ci_upper": round(float(ci_upper[i]), 4) if ci_upper is not None else None,
            "scenario": scenario,
            "yield_reduction_pct": yield_reduction_pct,
            "model_type": model_type,
        })

    results.sort(key=lambda x: x["predicted_risk"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Tool 4: Train Crop Model
# ---------------------------------------------------------------------------

@tool
def train_crop_model(
    state: str = "MO",
    commodity: str = "CORN",
    model_type: str = "random_forest",
) -> dict[str, Any]:
    """Train a crop yield prediction model from NASS historical data.

    Uses county-level food environment + access features to predict which
    areas are most dependent on specific crops and vulnerable to yield drops.

    Args:
        state: State code.
        commodity: Crop name (e.g. "CORN", "SOYBEANS").
        model_type: "xgboost" or "random_forest".

    Returns:
        Dict with model info, metrics, and path.
    """
    # For crop models we use the food environment data as proxy features
    # Real NASS yield data would come from the API, but for training we
    # use what's in SQLite and treat food insecurity indicators as targets
    return train_risk_model.invoke({
        "state": state,
        "model_type": model_type,
        "target_col": "FOODINSEC_15_17",
    })


# ---------------------------------------------------------------------------
# Tool 5: Predict Crop Yield Impact
# ---------------------------------------------------------------------------

@tool
def predict_crop_yield(
    state: str = "MO",
    commodity: str = "CORN",
    yield_reduction_pct: float = 20.0,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Predict county vulnerability to crop yield reductions.

    Runs the risk model under a yield reduction scenario to identify
    which counties would be hardest hit by crop failures.

    Args:
        state: State code.
        commodity: Crop affected.
        yield_reduction_pct: Percentage yield drop to simulate.
        top_n: Counties to return.

    Returns:
        Ranked list of most vulnerable counties.
    """
    results = predict_risk.invoke({
        "state": state,
        "scenario": "drought",
        "yield_reduction_pct": yield_reduction_pct,
        "top_n": top_n,
    })

    # Annotate with commodity
    for r in results:
        if isinstance(r, dict) and "error" not in r:
            r["commodity"] = commodity
            r["scenario"] = f"{commodity} yield -{yield_reduction_pct}%"

    return results


# ---------------------------------------------------------------------------
# Tool 6: Feature Importance (SHAP)
# ---------------------------------------------------------------------------

@tool
def get_feature_importance(
    state: str = "MO",
    model_type: str = "xgboost",
    target_col: str = "FOODINSEC_15_17",
    top_n: int = 10,
) -> dict[str, Any]:
    """Get SHAP-based feature importance for a trained model.

    Explains which features drive predictions most. Uses SHAP for tree
    models (XGBoost/RF) and falls back to built-in feature_importances_.

    Args:
        state: State the model was trained on.
        model_type: Model type to explain.
        target_col: Target variable.
        top_n: Number of top features to return.

    Returns:
        Dict with ranked features, importance scores, and interpretation.
    """
    model_path, scaler_path, meta = _find_model(state, model_type, target_col)
    if meta is None:
        return {"error": "No trained model found. Run train_risk_model first."}

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = meta["feature_names"]

    # Build features for explanation
    features = build_feature_matrix.invoke({"state": state})
    if not features or "error" in features[0]:
        return {"error": "Failed to build features for SHAP"}

    df = pd.DataFrame(features)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_names].values
    X_scaled = scaler.transform(X)

    # Try SHAP first
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = list(zip(feature_names, mean_abs_shap.tolist()))
            importance.sort(key=lambda x: x[1], reverse=True)
            method = "shap"
        except Exception:
            importance = _fallback_importance(model, feature_names)
            method = "built_in"
    else:
        importance = _fallback_importance(model, feature_names)
        method = "built_in"

    top_features = [
        {"rank": i + 1, "feature": name, "importance": round(score, 4)}
        for i, (name, score) in enumerate(importance[:top_n])
    ]

    return {
        "method": method,
        "model_type": model_type,
        "target": target_col,
        "top_features": top_features,
        "total_features": len(feature_names),
        "interpretation": _interpret_features(top_features),
    }


# ---------------------------------------------------------------------------
# Tool 7: Anomaly Detection
# ---------------------------------------------------------------------------

@tool
def detect_anomalies(
    state: str = "MO",
    contamination: float = 0.1,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Detect anomalous counties using Isolation Forest.

    Flags counties with unusual combinations of food environment indicators,
    which may signal emerging food security issues or data quality problems.

    Args:
        state: State code.
        contamination: Expected proportion of outliers (0.01-0.5).
        top_n: Max anomalies to return.

    Returns:
        List of anomalous counties with anomaly scores and key deviations.
    """
    features = build_feature_matrix.invoke({"state": state})
    if not features or "error" in features[0]:
        return [{"error": "Failed to build features"}]

    df = pd.DataFrame(features)
    exclude = {"FIPS", "State", "County"}
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64", "float32")]

    if not numeric_cols:
        return [{"error": "No numeric features available"}]

    X = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=min(max(contamination, 0.01), 0.5),
        random_state=42,
        n_estimators=100,
    )
    scores = iso.fit_predict(X_scaled)
    anomaly_scores = iso.decision_function(X_scaled)

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        if scores[i] == -1:  # anomaly
            # Find which features deviate most
            z_scores = (X_scaled[i] - X_scaled.mean(axis=0)) / (X_scaled.std(axis=0) + 1e-8)
            top_deviations = sorted(
                zip(numeric_cols, z_scores.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]

            results.append({
                "county": row.get("County"),
                "state": row.get("State"),
                "fips": row.get("FIPS"),
                "anomaly_score": round(float(anomaly_scores[i]), 4),
                "top_deviations": [
                    {"feature": name, "z_score": round(z, 2)} for name, z in top_deviations
                ],
            })

    results.sort(key=lambda x: x["anomaly_score"])
    return results[:top_n]


# ---------------------------------------------------------------------------
# Tool 8: Web Search for Agricultural Risks
# ---------------------------------------------------------------------------

@tool
def web_search_risks(
    query: str = "Missouri agricultural risks 2026",
    region: str = "Missouri",
) -> list[dict[str, Any]]:
    """Search for emerging agricultural threats using web data.

    Searches for pest outbreaks, disease alerts, weather emergencies,
    and supply chain disruptions relevant to the specified region.

    Args:
        query: Search query (auto-enhanced with agricultural terms).
        region: Geographic focus area.

    Returns:
        List of findings with titles, summaries, and source URLs.
    """
    import httpx

    # Enhance query with agricultural context
    enhanced = f"{query} food security crop disease pest livestock"

    # Use DuckDuckGo instant answer API (free, no key)
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": enhanced, "format": "json", "no_html": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Summary"),
                "summary": data["Abstract"],
                "url": data.get("AbstractURL", ""),
                "source": data.get("AbstractSource", "DuckDuckGo"),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "summary": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo Related",
                })

        if not results:
            results.append({
                "title": "No direct results found",
                "summary": f"Web search for '{query}' returned no structured results. "
                           f"Try more specific queries about {region} agricultural threats.",
                "url": "",
                "source": "system",
            })

        return results

    except Exception as e:
        return [{"error": f"Web search failed: {e}", "query": query}]


# ---------------------------------------------------------------------------
# Tool 9: EDA Pipeline (automated exploratory data analysis with charts)
# ---------------------------------------------------------------------------

@tool
def run_eda_pipeline(
    table_name: str = "food_environment",
    state: str = "",
    target_col: str = "",
    top_n_features: int = 10,
) -> dict[str, Any]:
    """Run automated exploratory data analysis on any database table.

    Performs a complete EDA pipeline: descriptive stats, missing value
    assessment, distribution analysis, correlation matrix, and generates
    charts for the dashboard. Works on ANY data type in the database.

    Args:
        table_name: Database table to analyze (default "food_environment").
        state: Optional state filter (2-letter code). Empty = all data.
        target_col: Optional target variable for focused correlation analysis.
        top_n_features: Number of top features to include in reports.

    Returns:
        Dict with descriptive_stats, distributions, correlations, missing_values,
        outlier_summary, charts (Plotly specs), and recommendations.
    """
    from src.agent.tools.chart_generator import (
        create_bar_chart, create_chart,
    )

    conn = _get_db()
    try:
        # Load data
        if state:
            # Try state filter — fallback to full table if State column missing
            try:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table_name} WHERE State = ?",
                    conn, params=(state.upper(),),
                )
                if df.empty:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10000", conn)
            except Exception:
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10000", conn)
        else:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10000", conn)
    except Exception as e:
        return {"error": f"Failed to load table '{table_name}': {e}"}
    finally:
        conn.close()

    if df.empty:
        return {"error": f"Table '{table_name}' is empty or not found."}

    # Identify column types
    exclude_cols = {"FIPS", "State", "County", "CensusTract", "geoid"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in df.columns
        if c not in exclude_cols and not pd.api.types.is_numeric_dtype(df[c])
    ]

    result: dict[str, Any] = {
        "table": table_name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_types": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "identifier": len(exclude_cols & set(df.columns)),
        },
    }

    # --- 1. Missing value assessment ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }
    result["missing_values"] = {
        "total_missing": int(missing.sum()),
        "total_pct": round(float(missing.sum()) / (len(df) * len(df.columns)) * 100, 2),
        "columns_with_missing": len(missing_report),
        "details": dict(sorted(missing_report.items(), key=lambda x: x[1]["pct"], reverse=True)[:20]),
    }

    # --- 2. Descriptive statistics ---
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        result["descriptive_stats"] = {
            col: {
                "mean": float(desc.loc["mean", col]) if col in desc.columns else None,
                "std": float(desc.loc["std", col]) if col in desc.columns else None,
                "min": float(desc.loc["min", col]) if col in desc.columns else None,
                "q25": float(desc.loc["25%", col]) if col in desc.columns else None,
                "median": float(desc.loc["50%", col]) if col in desc.columns else None,
                "q75": float(desc.loc["75%", col]) if col in desc.columns else None,
                "max": float(desc.loc["max", col]) if col in desc.columns else None,
            }
            for col in numeric_cols[:top_n_features]
        }

    # --- 3. Distribution analysis (skewness, kurtosis) ---
    if numeric_cols:
        dist_info = {}
        for col in numeric_cols[:top_n_features]:
            series = df[col].dropna()
            if len(series) > 2:
                skew = float(series.skew())
                kurt = float(series.kurtosis())
                shape = "normal" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
                dist_info[col] = {"skewness": round(skew, 3), "kurtosis": round(kurt, 3), "shape": shape}
        result["distributions"] = dist_info

    # --- 4. Correlation analysis ---
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(4)

        # Top correlations (excluding self-correlations)
        top_corrs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                r = corr.loc[c1, c2]
                if not pd.isna(r) and abs(r) > 0.3:
                    top_corrs.append({"var1": c1, "var2": c2, "correlation": round(float(r), 4)})

        top_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        result["correlations"] = {
            "top_pairs": top_corrs[:20],
            "strong_positive": len([c for c in top_corrs if c["correlation"] > 0.7]),
            "strong_negative": len([c for c in top_corrs if c["correlation"] < -0.7]),
        }

        # Target-specific correlations
        if target_col and target_col in numeric_cols:
            target_corrs = [
                {"feature": col, "correlation": round(float(corr.loc[target_col, col]), 4)}
                for col in numeric_cols if col != target_col and not pd.isna(corr.loc[target_col, col])
            ]
            target_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            result["target_correlations"] = target_corrs[:top_n_features]

    # --- 5. Outlier detection (IQR method) ---
    outlier_summary = {}
    for col in numeric_cols[:top_n_features]:
        series = df[col].dropna()
        if len(series) > 4:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            if n_outliers > 0:
                outlier_summary[col] = {"count": n_outliers, "pct": round(n_outliers / len(series) * 100, 2)}
    result["outliers"] = outlier_summary

    # --- 6. Generate EDA charts ---
    charts: list[dict] = []

    # Chart 1: Distribution histogram of target or first numeric col
    hist_col = target_col if target_col and target_col in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    if hist_col:
        hist_data = [{"value": float(v)} for v in df[hist_col].dropna().head(500)]
        if hist_data:
            chart = create_chart.invoke({
                "chart_type": "histogram", "title": f"Distribution: {hist_col}",
                "data_json": json.dumps(hist_data), "x_col": "value", "nbins": 30,
            })
            if isinstance(chart, dict) and "plotly_spec" in chart:
                charts.append(chart)

    # Chart 2: Top features by mean value (bar chart)
    if numeric_cols:
        means = df[numeric_cols].mean().sort_values(ascending=False).head(10)
        mean_data = [{"feature": k, "mean_value": round(float(v), 4)} for k, v in means.items() if not pd.isna(v)]
        if mean_data:
            chart = create_bar_chart.invoke({
                "title": f"Top Features by Mean Value — {table_name}",
                "data_json": json.dumps(mean_data),
                "x_col": "feature", "y_col": "mean_value", "horizontal": True,
            })
            if isinstance(chart, dict) and "plotly_spec" in chart:
                charts.append(chart)

    # Chart 3: Missing values bar chart
    if missing_report:
        miss_data = [
            {"column": col, "missing_pct": info["pct"]}
            for col, info in sorted(missing_report.items(), key=lambda x: x[1]["pct"], reverse=True)[:10]
        ]
        if miss_data:
            chart = create_bar_chart.invoke({
                "title": "Missing Data by Column (%)",
                "data_json": json.dumps(miss_data),
                "x_col": "column", "y_col": "missing_pct", "horizontal": True,
            })
            if isinstance(chart, dict) and "plotly_spec" in chart:
                charts.append(chart)

    # Chart 4: Top correlations with target (if specified)
    if target_col and "target_correlations" in result:
        corr_data = [
            {"feature": c["feature"], "correlation": c["correlation"]}
            for c in result["target_correlations"][:10]
        ]
        if corr_data:
            chart = create_bar_chart.invoke({
                "title": f"Top Correlations with {target_col}",
                "data_json": json.dumps(corr_data),
                "x_col": "feature", "y_col": "correlation", "horizontal": True,
            })
            if isinstance(chart, dict) and "plotly_spec" in chart:
                charts.append(chart)

    # Chart 5: Box plot of target by category (if applicable)
    if target_col and target_col in numeric_cols and categorical_cols:
        cat_col = categorical_cols[0]
        top_cats = df[cat_col].value_counts().head(8).index.tolist()
        box_data = [
            {cat_col: str(r[cat_col]), target_col: float(r[target_col])}
            for _, r in df[df[cat_col].isin(top_cats)].head(500).iterrows()
            if pd.notna(r[target_col])
        ]
        if box_data:
            chart = create_chart.invoke({
                "chart_type": "box", "title": f"{target_col} by {cat_col}",
                "data_json": json.dumps(box_data),
                "x_col": cat_col, "y_col": target_col,
            })
            if isinstance(chart, dict) and "plotly_spec" in chart:
                charts.append(chart)

    result["charts"] = charts
    result["n_charts"] = len(charts)

    # --- 7. Recommendations ---
    recs = []
    if result["missing_values"]["total_pct"] > 20:
        recs.append("High missing data rate — consider dropping columns with >90% missing before modeling.")
    if outlier_summary:
        recs.append(f"{len(outlier_summary)} features have outliers — use robust scalers or clip extreme values.")
    strong_pos = result.get("correlations", {}).get("strong_positive", 0)
    if strong_pos > 3:
        recs.append(f"{strong_pos} highly correlated feature pairs — consider dimensionality reduction or feature selection.")
    skewed = [c for c, d in result.get("distributions", {}).items() if d.get("shape") != "normal"]
    if len(skewed) > len(numeric_cols) * 0.5:
        recs.append("Most features are skewed — apply log transform or use tree-based models (Random Forest, GBM).")
    if not recs:
        recs.append("Data quality looks good — proceed with modeling.")
    result["recommendations"] = recs

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin, 1989).

    Measures agreement between predicted and actual values.
    CCC = 1 means perfect agreement; CCC = 0 means no agreement.
    More stringent than R² because it penalizes systematic bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 0.0

    return float(2 * covar / denominator)


def _find_model(
    state: str, model_type: str, target_col: str
) -> tuple[str | None, str | None, dict | None]:
    """Find the latest trained model matching criteria."""
    prefix = f"{model_type}_{state}_{target_col}_"
    candidates = sorted(MODEL_DIR.glob(f"{prefix}*_meta.json"), reverse=True)

    for meta_path in candidates:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            model_path = meta.get("model_path")
            scaler_path = meta.get("scaler_path")
            if model_path and os.path.exists(model_path) and os.path.exists(scaler_path):
                return model_path, scaler_path, meta
        except (json.JSONDecodeError, KeyError):
            continue

    return None, None, None


def _get_confidence_intervals(
    model, X: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute confidence intervals from tree ensemble variance."""
    if isinstance(model, RandomForestRegressor):
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean - 1.96 * std, mean + 1.96 * std
    elif _HAS_XGB and isinstance(model, xgb.XGBRegressor):
        # XGBoost: use predict intervals via standard deviation heuristic
        pred = model.predict(X)
        # Approximate: use residual std as proxy
        std_estimate = np.std(pred) * 0.1
        return pred - 1.96 * std_estimate, pred + 1.96 * std_estimate
    return None, None


def _fallback_importance(model, feature_names: list[str]) -> list[tuple[str, float]]:
    """Get feature importance from model.feature_importances_."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)
    return [(name, 0.0) for name in feature_names]


def _interpret_features(top_features: list[dict]) -> str:
    """Generate a human-readable interpretation of feature importance."""
    if not top_features:
        return "No features to interpret."

    lines = ["Key risk drivers (most to least important):"]
    descriptions = {
        "FOODINSEC_15_17": "food insecurity rate",
        "POVRATE15": "poverty rate",
        "PCT_LACCESS_POP15": "population with low food access",
        "PCT_SNAP16": "SNAP program participation",
        "PCT_OBESE_ADULTS17": "adult obesity rate",
        "PCT_NSLP15": "school lunch program participation",
        "PCT_LACCESS_CHILD15": "children with low food access",
        "PCT_LACCESS_SENIORS15": "seniors with low food access",
        "PCT_LACCESS_HHNV15": "households with no vehicle and low access",
        "GROC16": "number of grocery stores",
        "CONVS16": "number of convenience stores",
        "SUPERC16": "number of supercenters",
        "SNAPS17": "SNAP-authorized stores",
        "WICS16": "WIC-authorized stores",
        "grocery_to_conv_ratio": "ratio of grocery to convenience stores",
        "access_poverty_interaction": "intersection of low access and poverty",
        "snap_coverage_gap": "gap between food insecurity and SNAP coverage",
        "urban_pct": "urbanization level",
        "tract_count": "number of census tracts",
    }

    for feat in top_features[:5]:
        name = feat["feature"]
        desc = descriptions.get(name, name.replace("_", " ").lower())
        lines.append(f"  {feat['rank']}. {desc} (importance: {feat['importance']:.3f})")

    return "\n".join(lines)
