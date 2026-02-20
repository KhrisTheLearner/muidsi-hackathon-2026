"""Tool for computing evaluation metrics on model predictions.

Directly addresses the 'Evaluation Metrics' judging category by letting
the agent self-assess prediction quality against ground truth data.
Includes CCC (Concordance Correlation Coefficient) and SHAP explainability.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from langchain_core.tools import tool


@tool
def compute_evaluation_metrics(
    predictions_json: str,
    actuals_json: str,
    pred_col: str,
    actual_col: str,
    task_type: str = "regression",
) -> dict[str, Any]:
    """Compute evaluation metrics comparing predictions against actual values.

    Use after running predictions to quantify model accuracy. Supports both
    regression (RMSE, MAE, R-squared) and classification (accuracy, precision,
    recall, F1) tasks.

    Args:
        predictions_json: JSON array of prediction rows.
        actuals_json: JSON array of actual/ground-truth rows.
        pred_col: Column name containing predicted values.
        actual_col: Column name containing actual values.
        task_type: "regression" or "classification".
    """
    preds = json.loads(predictions_json) if isinstance(predictions_json, str) else predictions_json
    actuals = json.loads(actuals_json) if isinstance(actuals_json, str) else actuals_json

    p_vals = [r.get(pred_col) for r in preds if r.get(pred_col) is not None]
    a_vals = [r.get(actual_col) for r in actuals if r.get(actual_col) is not None]

    n = min(len(p_vals), len(a_vals))
    if n == 0:
        return {"error": "No matching prediction/actual pairs found"}

    p_vals, a_vals = p_vals[:n], a_vals[:n]

    if task_type == "regression":
        errors = [p - a for p, a in zip(p_vals, a_vals)]
        sq_errors = [e ** 2 for e in errors]
        abs_errors = [abs(e) for e in errors]
        mean_a = sum(a_vals) / n
        ss_res = sum(sq_errors)
        ss_tot = sum((a - mean_a) ** 2 for a in a_vals)

        # Concordance Correlation Coefficient (Lin, 1989)
        ccc = _concordance_correlation(p_vals, a_vals)

        return {
            "task_type": "regression",
            "n_samples": n,
            "rmse": round(math.sqrt(ss_res / n), 4),
            "mae": round(sum(abs_errors) / n, 4),
            "r_squared": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
            "ccc": round(ccc, 4),
            "mean_error": round(sum(errors) / n, 4),
            "max_error": round(max(abs_errors), 4),
        }
    else:
        tp = fp = tn = fn = 0
        for p, a in zip(p_vals, a_vals):
            if p == 1 and a == 1: tp += 1
            elif p == 1 and a == 0: fp += 1
            elif p == 0 and a == 0: tn += 1
            else: fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "task_type": "classification",
            "n_samples": n,
            "accuracy": round((tp + tn) / n, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }


@tool
def compare_scenarios(
    scenarios_json: str,
    metric_col: str,
    label_col: str = "scenario",
) -> dict[str, Any]:
    """Compare multiple scenario outputs side-by-side.

    Use after running run_prediction with different scenarios to rank outcomes
    and identify which scenario poses the greatest risk.

    Args:
        scenarios_json: JSON array combining results from multiple scenarios,
            each row must have a scenario label and metric value.
        metric_col: Column with the metric to compare (e.g. "risk_score").
        label_col: Column identifying the scenario (default "scenario").
    """
    rows = json.loads(scenarios_json) if isinstance(scenarios_json, str) else scenarios_json

    groups: dict[str, list[float]] = {}
    for r in rows:
        label = str(r.get(label_col, "unknown"))
        val = r.get(metric_col)
        if val is not None:
            groups.setdefault(label, []).append(float(val))

    summary = []
    for label, vals in sorted(groups.items()):
        summary.append({
            "scenario": label,
            "count": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "max": round(max(vals), 4),
            "min": round(min(vals), 4),
            "std_dev": round((sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals))**0.5, 4),
        })

    summary.sort(key=lambda x: x["mean"], reverse=True)

    return {
        "comparison": summary,
        "highest_risk_scenario": summary[0]["scenario"] if summary else "unknown",
        "scenarios_analyzed": len(groups),
    }


# ---------------------------------------------------------------------------
# CCC helper
# ---------------------------------------------------------------------------

def _concordance_correlation(pred: list, actual: list) -> float:
    """Concordance Correlation Coefficient (Lin, 1989).

    Measures agreement between predicted and actual.
    More stringent than Pearson r because it penalizes bias.
    """
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if len(p) < 2:
        return 0.0
    mean_p, mean_a = np.mean(p), np.mean(a)
    var_p, var_a = np.var(p), np.var(a)
    covar = np.mean((p - mean_p) * (a - mean_a))
    denom = var_p + var_a + (mean_p - mean_a) ** 2
    return float(2 * covar / denom) if denom > 0 else 0.0


@tool
def compute_ccc(
    predictions_json: str,
    actuals_json: str,
    pred_col: str,
    actual_col: str,
) -> dict[str, Any]:
    """Compute Concordance Correlation Coefficient between predictions and actuals.

    CCC measures both correlation AND systematic bias. A model that
    consistently over- or under-predicts will have high R² but low CCC.
    Use this as a stricter validation metric than R² alone.

    Args:
        predictions_json: JSON array of prediction rows.
        actuals_json: JSON array of ground-truth rows.
        pred_col: Column with predicted values.
        actual_col: Column with actual values.

    Returns:
        Dict with CCC value, Pearson r, and bias interpretation.
    """
    preds = json.loads(predictions_json) if isinstance(predictions_json, str) else predictions_json
    actuals = json.loads(actuals_json) if isinstance(actuals_json, str) else actuals_json

    p_vals = [float(r[pred_col]) for r in preds if r.get(pred_col) is not None]
    a_vals = [float(r[actual_col]) for r in actuals if r.get(actual_col) is not None]
    n = min(len(p_vals), len(a_vals))
    if n < 2:
        return {"error": "Need at least 2 paired values for CCC"}

    p_vals, a_vals = p_vals[:n], a_vals[:n]
    p_arr, a_arr = np.array(p_vals), np.array(a_vals)

    ccc = _concordance_correlation(p_vals, a_vals)
    pearson_r = float(np.corrcoef(p_arr, a_arr)[0, 1])
    mean_bias = float(np.mean(p_arr - a_arr))

    interpretation = "excellent" if ccc > 0.9 else "good" if ccc > 0.7 else "moderate" if ccc > 0.5 else "poor"

    return {
        "ccc": round(ccc, 4),
        "pearson_r": round(pearson_r, 4),
        "mean_bias": round(mean_bias, 4),
        "n_pairs": n,
        "interpretation": f"{interpretation} agreement (CCC={ccc:.3f})",
    }


@tool
def explain_with_shap(
    model_path: str,
    data_json: str,
    feature_names_json: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Generate SHAP explanations for model predictions.

    SHAP (SHapley Additive exPlanations) shows how each feature
    contributes to individual predictions. Essential for model
    interpretability and trust in agricultural risk assessments.

    Args:
        model_path: Path to saved model (.pkl file).
        data_json: JSON array of feature rows to explain.
        feature_names_json: JSON array of feature column names.
        top_n: Number of top features to return.

    Returns:
        Dict with global feature importance and per-sample explanations.
    """
    try:
        import joblib
        import shap
    except ImportError:
        return {"error": "shap and joblib packages required. pip install shap joblib"}

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    data = json.loads(data_json) if isinstance(data_json, str) else data_json
    feature_names = json.loads(feature_names_json) if isinstance(feature_names_json, str) else feature_names_json

    X = np.array([[row.get(f, 0) for f in feature_names] for row in data])
    if X.shape[0] == 0:
        return {"error": "No data rows provided"}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        return {"error": f"SHAP computation failed: {e}"}

    # Global importance (mean absolute SHAP)
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    top_features = [
        {"feature": name, "mean_abs_shap": round(val, 4)}
        for name, val in global_importance[:top_n]
    ]

    # Per-sample explanation for top 3 rows
    sample_explanations = []
    for i in range(min(3, len(data))):
        sample = sorted(
            zip(feature_names, shap_values[i].tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        sample_explanations.append({
            "sample_index": i,
            "top_contributions": [
                {"feature": name, "shap_value": round(val, 4)} for name, val in sample
            ],
        })

    return {
        "method": "SHAP TreeExplainer",
        "n_samples": len(data),
        "n_features": len(feature_names),
        "global_importance": top_features,
        "sample_explanations": sample_explanations,
    }
