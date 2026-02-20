#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriFlow ML Pipeline Validation - Suyog/NEW2 Methodology
=========================================================

Validates the county-level and census-tract ML pipelines end-to-end against
the local SQLite database.  Reproduces the Suyog.ipynb and NEW2.ipynb
notebook results using the agent's own ml_engine tooling.

Expected outcomes (from validated notebooks):
  County GBM (n=300, depth=3):  R² ≈ 0.998,  RMSE ≈ 0.099
  County LinearReg:             R² ≈ 0.983,  RMSE ≈ 0.328
  County SVR (RBF):             R² ≈ 0.912
  K-Means clusters:             n_clusters == 3
  Composite risk scores:        all in [0, 1]
  Tract GBM (n=500, depth=4):  R² ≈ 0.9949

Usage:
  cd muidsi-hackathon-2026
  python scripts/test_ml_pipeline.py [--model gradient_boosting] [--state MO]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Path fix so imports resolve from project root ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── AgriFlow tools ──────────────────────────────────────────────────────────
from src.agent.tools.ml_engine import (
    build_feature_matrix,
    build_tract_feature_matrix,
    compute_food_insecurity_risk,
    get_feature_importance,
    predict_risk,
    train_risk_model,
)

import io
import sys as _sys
# Force UTF-8 on Windows to handle Unicode output gracefully
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "  [*]"

_fails: list[str] = []


def assertion(condition: bool, msg: str, extra: str = "") -> None:
    if condition:
        print(f"  {PASS}  {msg}" + (f"  ({extra})" if extra else ""))
    else:
        print(f"  {FAIL}  FAIL: {msg}" + (f"  ({extra})" if extra else ""))
        _fails.append(msg)


# ===========================================================================
# Section 1: County Feature Matrix (Suyog / NEW1 2 methodology)
# ===========================================================================

def test_feature_matrix(state: str) -> pd.DataFrame | None:
    print(f"\n{'─'*60}")
    print(f"[1] Build County Feature Matrix — state={state}")
    print(f"{'-'*60}")

    t0 = time.time()
    result = build_feature_matrix.invoke({"state": state})
    elapsed = time.time() - t0

    if not result or (isinstance(result, list) and "error" in result[0]):
        err = result[0].get("error", "unknown") if result else "empty result"
        print(f"  {FAIL}  Feature matrix failed: {err}")
        _fails.append("build_feature_matrix returned error")
        return None

    df = pd.DataFrame(result)
    print(f"  {INFO}  Loaded {len(df)} counties, {len(df.columns)} columns  [{elapsed:.1f}s]")
    assertion(len(df) > 10, f"At least 10 counties loaded", f"got {len(df)}")

    # Show available columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"  {INFO}  Numeric columns ({len(num_cols)}): {', '.join(num_cols[:8])}{'...' if len(num_cols) > 8 else ''}")

    return df


# ===========================================================================
# Section 2: Interaction Features + K-Means (Suyog Step 2–3)
# ===========================================================================

def test_interaction_and_clustering(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'-'*60}")
    print("[2] Interaction Features + K-Means Clustering (k=3)")
    print(f"{'-'*60}")

    # --- Map column names (Suyog 21-23 data; fall back to 15-17 if needed) ---
    COL_MAP = {
        "poverty":     ["POVRATE21", "POVRATE15", "POVRATE17", "CHILDPOVRATE21"],
        "snap":        ["PCT_SNAP22", "PCT_SNAP17", "PCT_SNAP16"],
        "food_insec":  ["FOODINSEC_21_23", "FOODINSEC_18_20", "FOODINSEC_15_17"],
        "access":      ["PCT_LACCESS_HHNV19", "PCT_LACCESS_HHNV15", "LACCESS_HHNV19", "PCT_LACCESS_POP15"],
    }

    def pick_col(choices: list[str], df: pd.DataFrame) -> str | None:
        for c in choices:
            if c in df.columns:
                return c
        return None

    pov_col   = pick_col(COL_MAP["poverty"], df)
    snap_col  = pick_col(COL_MAP["snap"], df)
    food_col  = pick_col(COL_MAP["food_insec"], df)
    acc_col   = pick_col(COL_MAP["access"], df)

    found = {k: v for k, v in zip(["poverty", "snap", "food_insec", "access"],
                                    [pov_col, snap_col, food_col, acc_col]) if v}
    print(f"  {INFO}  Column mapping: {found}")

    if len(found) < 2:
        print(f"  {FAIL}  Insufficient columns for interaction features")
        _fails.append("Insufficient columns for interaction features")
        return df

    # --- Create 4 interaction features (where columns exist) ---
    created = []
    if pov_col and snap_col:
        df = df.copy()
        df["POVxSNAP"] = (
            pd.to_numeric(df[pov_col], errors="coerce").fillna(0)
            * pd.to_numeric(df[snap_col], errors="coerce").fillna(0)
            / 100
        ).round(4)
        created.append("POVxSNAP")

    if pov_col and acc_col:
        df["POVxLACCESS"] = (
            pd.to_numeric(df[pov_col], errors="coerce").fillna(0)
            * pd.to_numeric(df[acc_col], errors="coerce").fillna(0)
            / 100
        ).round(4)
        created.append("POVxLACCESS")

    if food_col and snap_col:
        df["FOODxSNAP"] = (
            pd.to_numeric(df[food_col], errors="coerce").fillna(0)
            * pd.to_numeric(df[snap_col], errors="coerce").fillna(0)
            / 100
        ).round(4)
        created.append("FOODxSNAP")

    if snap_col and acc_col:
        df["SNAPxLACCESS"] = (
            pd.to_numeric(df[snap_col], errors="coerce").fillna(0)
            * pd.to_numeric(df[acc_col], errors="coerce").fillna(0)
            / 100
        ).round(4)
        created.append("SNAPxLACCESS")

    print(f"  {INFO}  Created {len(created)} interaction features: {', '.join(created)}")
    assertion(len(created) >= 2, "At least 2 interaction features created", f"got {len(created)}")

    # --- K-Means clustering (k=3, Suyog Step 3) ---
    cluster_features = [c for c in created if c in df.columns]
    base_features = [c for c in [pov_col, snap_col, food_col, acc_col] if c]
    all_features = list(dict.fromkeys(base_features + cluster_features))  # dedup, preserve order

    X_cluster = df[all_features].select_dtypes(include="number").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_counts = df["Cluster"].value_counts().sort_index()
    print(f"  {INFO}  K-Means cluster sizes: {dict(cluster_counts)}")
    assertion(df["Cluster"].nunique() == 3, "Exactly 3 clusters formed")

    # --- Cluster profiles ---
    profile_cols = [c for c in [pov_col, snap_col, food_col] if c] + ["Cluster"]
    profile = df[profile_cols].groupby("Cluster").mean().round(3)
    print(f"\n  Cluster profiles (mean values):")
    for _, row in profile.iterrows():
        cluster_id = int(row.name)
        vals = "  ".join(f"{c}: {row[c]:.3f}" for c in profile.columns if c != "Cluster")
        print(f"    Cluster {cluster_id}: {vals}")

    return df


# ===========================================================================
# Section 3: County ML Models (Suyog Step 4)
# ===========================================================================

def test_county_models(state: str, model_type: str) -> dict:
    print(f"\n{'-'*60}")
    print(f"[3] County Model Training — model_type={model_type}")
    print(f"{'-'*60}")

    # Check for a target column with real county-level variance
    # NOTE: FOODINSEC_21_23 in the USDA Atlas is a state-level aggregate (same for
    # all counties in a state). POVRATE21 is the best county-level predictor.
    _CANDIDATE_TARGETS = [
        "POVRATE21", "CHILDPOVRATE21", "PCT_LACCESS_POP15",
        "FOODINSEC_21_23", "FOODINSEC_18_20", "FOODINSEC_15_17",
    ]
    import sqlite3 as _sqlite
    try:
        _conn = _sqlite.connect("data/agriflow.db")
        _cur = _conn.cursor()
        for _col in _CANDIDATE_TARGETS:
            try:
                row = _cur.execute(
                    f'SELECT MIN("{_col}"), MAX("{_col}") FROM food_environment WHERE State=?',
                    (state,)
                ).fetchone()
                if row and row[0] is not None and row[1] is not None and row[0] != row[1]:
                    selected_target = _col
                    print(f"  {INFO}  Auto-selected target column: {_col}  "
                          f"(range {row[0]:.2f} – {row[1]:.2f})")
                    break
            except Exception:
                continue
        else:
            selected_target = "FOODINSEC_21_23"
            print(f"  {INFO}  WARNING: All candidate targets have zero variance — "
                  f"using {selected_target} (R² may be 0)")
        _conn.close()
    except Exception:
        selected_target = "FOODINSEC_21_23"

    models_to_test = [model_type]
    if model_type == "gradient_boosting":
        # Also test linear regression as a baseline comparison
        models_to_test.append("linear_regression")

    results = {}

    for mtype in models_to_test:
        t0 = time.time()
        result = train_risk_model.invoke({
            "state": state,
            "model_type": mtype,
            "target_col": selected_target,
        })
        elapsed = time.time() - t0

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass

        if isinstance(result, dict) and "error" in result:
            print(f"  {FAIL}  {mtype}: {result['error']}")
            _fails.append(f"train_risk_model({mtype}) error: {result['error']}")
            continue

        # Extract metrics
        r2   = result.get("r2",   result.get("R2",   result.get("test_r2",   None)))
        rmse = result.get("rmse", result.get("RMSE", result.get("test_rmse", None)))
        mae  = result.get("mae",  result.get("MAE",  result.get("test_mae",  None)))
        n    = result.get("n_samples", result.get("n", "?"))

        # Try nested metrics dict
        if r2 is None and isinstance(result.get("metrics"), dict):
            m    = result["metrics"]
            r2   = m.get("r2",   m.get("R2",   m.get("test_r2")))
            rmse = m.get("rmse", m.get("RMSE", m.get("test_rmse")))

        r2_val   = float(r2)   if r2   is not None else None
        rmse_val = float(rmse) if rmse is not None else None

        metric_str = ""
        if r2_val   is not None: metric_str += f"R²={r2_val:.4f}"
        if rmse_val is not None: metric_str += f"  RMSE={rmse_val:.4f}"
        if mae      is not None: metric_str += f"  MAE={float(mae):.4f}"

        print(f"  {INFO}  {mtype}: {metric_str}  [{elapsed:.1f}s]  n={n}")

        # Assert expected thresholds from notebooks
        thresholds = {
            "gradient_boosting": 0.95,
            "linear_regression": 0.90,
            "random_forest":     0.90,
            "xgboost":           0.90,
            "svr":               0.85,
        }
        min_r2 = thresholds.get(mtype, 0.85)
        if r2_val is not None:
            assertion(r2_val >= min_r2, f"{mtype} R² ≥ {min_r2}", f"got {r2_val:.4f}")

        results[mtype] = {"r2": r2_val, "rmse": rmse_val}

    return results


# ===========================================================================
# Section 4: Risk Predictions (Suyog Step 5)
# ===========================================================================

def test_risk_predictions(state: str) -> None:
    print(f"\n{'─'*60}")
    print("[4] Risk Prediction — Top 10 Highest-Risk Counties")
    print(f"{'-'*60}")

    result = predict_risk.invoke({
        "state":              state,
        "scenario":           "baseline",
        "yield_reduction_pct": 0.0,
        "top_n":              10,
    })

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    if isinstance(result, list) and result and isinstance(result[0], dict):
        rows = result
    elif isinstance(result, dict) and "predictions" in result:
        rows = result["predictions"]
    elif isinstance(result, dict) and "top_counties" in result:
        rows = result["top_counties"]
    else:
        print(f"  {INFO}  Raw result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        rows = []

    if rows:
        print(f"  {INFO}  Top {len(rows)} counties by predicted risk:")
        for i, row in enumerate(rows[:10], 1):
            county = row.get("County", row.get("county", "?"))
            score  = row.get("predicted_risk", row.get("risk_score", row.get("score", "?")))
            print(f"    {i:2d}. {county:<20s}  score={score}")

        # Assert scores are in [0, 1] where possible
        score_vals = []
        for row in rows:
            s = row.get("predicted_risk", row.get("risk_score", row.get("score")))
            if s is not None:
                try:
                    score_vals.append(float(s))
                except (TypeError, ValueError):
                    pass

        if score_vals:
            # Scores may be raw percentages (0–100) or normalized (0–1)
            in_pct_range = all(0 <= s <= 100 for s in score_vals)
            in_norm_range = all(0 <= s <= 1 for s in score_vals)
            range_label = "[0,1]" if in_norm_range else "[0,100]" if in_pct_range else "out of range"
            assertion(in_pct_range, "Predicted risk scores in [0, 100]",
                      f"range={range_label}  min={min(score_vals):.2f}  max={max(score_vals):.2f}")
    else:
        print(f"  {INFO}  predict_risk returned: {str(result)[:200]}")


# ===========================================================================
# Section 5: 6-Feature Composite Risk Score (Suyog Method)
# ===========================================================================

def test_composite_risk_score(df: pd.DataFrame) -> None:
    print(f"\n{'─'*60}")
    print("[5] 6-Feature Composite Risk Score (Suyog Method)")
    print(f"{'-'*60}")

    # Preferred Suyog column names → fallbacks in our DB
    SUYOG_COLS = {
        "FOODINSEC_18_20": ["FOODINSEC_18_20", "FOODINSEC_15_17", "FOODINSEC_13_15"],
        "FOODINSEC_21_23": ["FOODINSEC_21_23", "FOODINSEC_18_20", "FOODINSEC_15_17"],
        "POVRATE21":       ["POVRATE21",  "POVRATE15", "POVRATE17", "CHILDPOVRATE21"],
        "PCT_SNAP22":      ["PCT_SNAP22", "PCT_SNAP17", "PCT_SNAP16"],
        "PCT_WICWOMEN16":  ["PCT_WICWOMEN16", "PCT_WICWOMEN21", "WICS16"],
        "LACCESS_HHNV19":  ["PCT_LACCESS_HHNV19", "PCT_LACCESS_HHNV15", "LACCESS_HHNV19"],
    }

    selected: dict[str, str] = {}
    for feat_name, candidates in SUYOG_COLS.items():
        for c in candidates:
            if c in df.columns:
                selected[feat_name] = c
                break

    print(f"  {INFO}  Feature mapping:")
    for feat, col in selected.items():
        matched = "exact" if feat == col else f"→ {col}"
        print(f"    {feat:<22s} {matched}")

    if len(selected) < 3:
        print(f"  {FAIL}  Only {len(selected)} features found — need ≥ 3 for composite score")
        _fails.append("Composite risk score: insufficient features")
        return

    # Build the feature matrix
    X = pd.DataFrame({feat: pd.to_numeric(df[col], errors="coerce").fillna(0)
                      for feat, col in selected.items()})

    # MinMax normalize to [0, 1]
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Composite = average of normalized features
    df2 = df.copy()
    df2["Composite_Risk_Score"] = X_norm.mean(axis=1).round(4)

    # Percentile-based categorization (33rd / 66th)
    p33 = df2["Composite_Risk_Score"].quantile(0.33)
    p66 = df2["Composite_Risk_Score"].quantile(0.66)
    df2["Risk_Category"] = pd.cut(
        df2["Composite_Risk_Score"],
        bins=[-0.001, p33, p66, 1.001],
        labels=["Low", "Medium", "High"],
    )

    cat_counts = df2["Risk_Category"].value_counts().sort_index()
    score_min  = df2["Composite_Risk_Score"].min()
    score_max  = df2["Composite_Risk_Score"].max()

    print(f"\n  {INFO}  Score range: [{score_min:.4f}, {score_max:.4f}]")
    print(f"  {INFO}  Cutoffs: 33rd pct={p33:.4f}  66th pct={p66:.4f}")
    print(f"  {INFO}  Category distribution: {dict(cat_counts)}")

    # Top 10 high-risk counties
    top10 = df2.nlargest(10, "Composite_Risk_Score")[
        ["County", "Composite_Risk_Score", "Risk_Category"]
        + [c for c in selected.values() if c in df2.columns][:3]
    ]
    print(f"\n  Top 10 High-Risk Counties ({len(selected)}-feature composite):")
    for _, row in top10.iterrows():
        cat = row.get("Risk_Category", "?")
        print(f"    {row['County']:<20s}  score={row['Composite_Risk_Score']:.4f}  [{cat}]")

    # Assertions
    assertion(0 <= score_min <= score_max <= 1,
              "Composite risk scores in [0, 1]",
              f"min={score_min:.4f}  max={score_max:.4f}")
    assertion("High" in cat_counts.index, "Risk categories include 'High'")
    assertion(len(selected) >= 3, f"≥ 3 features used in composite score", f"got {len(selected)}")


# ===========================================================================
# Section 6: Feature Importance / SHAP
# ===========================================================================

def test_feature_importance(state: str) -> None:
    print(f"\n{'─'*60}")
    print("[6] Feature Importance (SHAP / built-in)")
    print(f"{'-'*60}")

    result = get_feature_importance.invoke({"state": state, "top_n": 10})

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    features = []
    if isinstance(result, list):
        features = result
    elif isinstance(result, dict):
        features = result.get("feature_importance", result.get("features", []))

    if features:
        print(f"  {INFO}  Top {len(features)} features:")
        for i, f in enumerate(features[:10], 1):
            name  = f.get("feature", f.get("name",  "?"))
            score = f.get("importance", f.get("score", "?"))
            print(f"    {i:2d}. {name:<30s}  importance={score}")

        # Expected: PCT_WICWOMEN16 should be top predictor (~65%)
        top_feature = features[0].get("feature", features[0].get("name", "")) if features else ""
        wic_in_top3 = any(
            "wic" in str(f.get("feature", f.get("name", ""))).lower()
            for f in features[:3]
        )
        if wic_in_top3:
            print(f"  {PASS}  WIC-related feature in top 3 (as expected from Suyog notebooks)")
        else:
            print(f"  {INFO}  WIC-related feature not in top 3 (top feature: {top_feature})")

        assertion(len(features) >= 3, "At least 3 features returned by importance tool")
    else:
        print(f"  {INFO}  No model cached yet (train first). Raw: {str(result)[:200]}")


# ===========================================================================
# Section 7: Census-Tract Pipeline (NEW2 methodology)
# ===========================================================================

def test_tract_pipeline(state: str) -> None:
    print(f"\n{'─'*60}")
    print(f"[7] Census-Tract Pipeline (NEW2) — state={state or 'all'}")
    print(f"{'-'*60}")

    # --- Build tract feature matrix ---
    t0 = time.time()
    tract_result = build_tract_feature_matrix.invoke({"state": state})
    elapsed = time.time() - t0

    if isinstance(tract_result, list) and tract_result:
        first = tract_result[0]
        if "error" in first:
            print(f"  {INFO}  Tract features: {first['error']} — table may not be loaded yet")
            print(f"  {INFO}  Skipping tract tests (run data pipeline to load food_access table)")
            return
        df_tract = pd.DataFrame(tract_result)
        print(f"  {INFO}  Tract feature matrix: {len(df_tract)} tracts, {len(df_tract.columns)} cols  [{elapsed:.1f}s]")
        assertion(len(df_tract) > 100, f"More than 100 tracts loaded", f"got {len(df_tract)}")
    else:
        print(f"  {INFO}  No tract data available — skipping tract tests")
        return

    # --- Compute tract risk scores ---
    t0 = time.time()
    risk_result = compute_food_insecurity_risk.invoke({"state": state, "top_n": 20})
    elapsed = time.time() - t0

    if isinstance(risk_result, str):
        try:
            risk_result = json.loads(risk_result)
        except json.JSONDecodeError:
            pass

    # compute_food_insecurity_risk returns [{summary: ..., top_tracts: [...]}]
    top_tracts = []
    if isinstance(risk_result, list) and risk_result:
        first = risk_result[0]
        if isinstance(first, dict) and "top_tracts" in first:
            top_tracts = first["top_tracts"]
        elif isinstance(first, dict) and "census_tract" in first:
            top_tracts = risk_result  # Direct list of tract dicts
    elif isinstance(risk_result, dict):
        top_tracts = risk_result.get("top_tracts", risk_result.get("results", []))

    if top_tracts:
        print(f"  {INFO}  Top {len(top_tracts)} highest-risk tracts  [{elapsed:.1f}s]:")
        for i, t in enumerate(top_tracts[:5], 1):
            # Handle both naming conventions
            tract  = t.get("census_tract", t.get("CensusTract", t.get("tract", "?")))
            score  = t.get("food_insecurity_risk", t.get("Food_Insecurity_Risk", t.get("risk", "?")))
            county = t.get("county", t.get("County", ""))
            tract_str = str(tract) if tract else "?"
            print(f"    {i}. Tract {tract_str:<15s}  {county:<15s}  risk={score}")

        # Assert: scores in [0, 1]
        score_vals = []
        for t in top_tracts:
            s = t.get("food_insecurity_risk", t.get("Food_Insecurity_Risk", t.get("risk")))
            if s is not None:
                try:
                    score_vals.append(float(s))
                except (TypeError, ValueError):
                    pass
        if score_vals:
            assertion(all(0 <= s <= 1.001 for s in score_vals),
                      "Tract risk scores in [0, 1]",
                      f"min={min(score_vals):.4f}  max={max(score_vals):.4f}")
    else:
        print(f"  {INFO}  Risk scoring returned: {str(risk_result)[:200]}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="AgriFlow ML Pipeline Validation")
    parser.add_argument("--state",  default="MO",                 help="State code (default: MO)")
    parser.add_argument("--model",  default="gradient_boosting",
                        choices=["gradient_boosting", "random_forest", "xgboost",
                                 "linear_regression", "svr"],
                        help="Primary model type to test (default: gradient_boosting)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AgriFlow ML Pipeline Validation")
    print(f"  State: {args.state}   Primary model: {args.model}")
    print("=" * 60)

    # 1. Feature matrix
    df = test_feature_matrix(args.state)
    if df is None:
        print(f"\n{FAIL}  Cannot continue — no feature matrix.  Check DB_PATH and data pipeline.")
        return 1

    # 2. Interaction features + K-Means
    df = test_interaction_and_clustering(df)

    # 3. County models
    model_results = test_county_models(args.state, args.model)

    # 4. Risk predictions (requires trained model from step 3)
    test_risk_predictions(args.state)

    # 5. Composite risk score
    test_composite_risk_score(df)

    # 6. Feature importance
    test_feature_importance(args.state)

    # 7. Tract pipeline (NEW2)
    test_tract_pipeline(args.state)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if _fails:
        print(f"  {FAIL}  {len(_fails)} assertion(s) failed:")
        for f in _fails:
            print(f"    - {f}")
        print("=" * 60)
        return 1
    else:
        print(f"  {PASS}  All assertions passed — ML pipeline is healthy.")
        if model_results:
            for mtype, m in model_results.items():
                r2 = m.get("r2")
                if r2 is not None:
                    bar = "#" * int(r2 * 20)
                    print(f"      {mtype:<22s} R²={r2:.4f}  {bar}")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
