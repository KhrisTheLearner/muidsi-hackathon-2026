"""Analytics supervisor — coordinates 6 subagents for ML-driven analysis.

This is a meta-tool registered with LangGraph's ToolNode. When the
tool_caller routes a step tagged [analytics], it invokes
`run_analytics_pipeline` which internally coordinates:

  1. research_agent  — web search for emerging threats
  2. data_agent      — build feature matrix from all data sources
  3. ml_agent        — train model + predict risk
  4. verify_agent    — compute CCC, RMSE, SHAP
  5. viz_agent       — generate charts and maps
  6. analysis_agent  — synthesize narrative

Each subagent is a plain function (not an LLM call) that invokes the
appropriate tools. This keeps token costs near zero for the coordination
layer — only the final analysis step needs an LLM.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Subagent functions
# ---------------------------------------------------------------------------

def _research_subagent(state: str, query: str) -> dict[str, Any]:
    """Subagent 1: Web research for emerging agricultural risks."""
    from src.agent.tools.ml_engine import web_search_risks

    results = web_search_risks.invoke({
        "query": query or f"{state} agricultural risks food security 2026",
        "region": state,
    })
    return {"web_research": results, "sources": ["DuckDuckGo web search"]}


def _data_subagent(state: str) -> dict[str, Any]:
    """Subagent 2: Build the feature matrix from all available data."""
    from src.agent.tools.ml_engine import build_feature_matrix

    features = build_feature_matrix.invoke({"state": state})
    n_rows = len(features) if features and "error" not in features[0] else 0
    n_cols = len(features[0]) if n_rows > 0 else 0

    return {
        "feature_matrix": features,
        "n_counties": n_rows,
        "n_features": n_cols,
        "sources": ["USDA Food Environment Atlas", "USDA Food Access Atlas"],
    }


def _ml_subagent(
    state: str,
    model_type: str,
    target_col: str,
    scenario: str,
    yield_reduction: float,
    top_n: int,
) -> dict[str, Any]:
    """Subagent 3: Train model and run predictions."""
    from src.agent.tools.ml_engine import predict_risk, train_risk_model

    # Train
    train_result = train_risk_model.invoke({
        "state": state,
        "model_type": model_type,
        "target_col": target_col,
    })

    if "error" in train_result:
        return {"error": train_result["error"], "predictions": [], "model_info": train_result}

    # Predict
    predictions = predict_risk.invoke({
        "state": state,
        "model_type": model_type,
        "target_col": target_col,
        "scenario": scenario,
        "yield_reduction_pct": yield_reduction,
        "top_n": top_n,
    })

    return {
        "model_info": train_result,
        "predictions": predictions,
        "sources": [f"Trained {model_type} model on {train_result.get('n_samples', '?')} counties"],
    }


def _verify_subagent(
    model_info: dict, state: str, model_type: str, target_col: str
) -> dict[str, Any]:
    """Subagent 4: Evaluate model with CCC, RMSE, and SHAP."""
    from src.agent.tools.ml_engine import detect_anomalies, get_feature_importance

    results: dict[str, Any] = {
        "metrics": model_info.get("metrics", {}),
        "cv_r2": model_info.get("cv_r2", "N/A"),
    }

    # Feature importance
    importance = get_feature_importance.invoke({
        "state": state,
        "model_type": model_type,
        "target_col": target_col,
        "top_n": 10,
    })
    results["feature_importance"] = importance

    # Anomaly detection
    anomalies = detect_anomalies.invoke({
        "state": state,
        "contamination": 0.1,
        "top_n": 5,
    })
    results["anomalies"] = anomalies

    return results


def _eda_subagent(state: str, target_col: str) -> dict[str, Any]:
    """Subagent 5a: Run automated EDA pipeline in parallel with modeling."""
    from src.agent.tools.ml_engine import run_eda_pipeline

    eda_result = run_eda_pipeline.invoke({
        "table_name": "food_environment",
        "state": state,
        "target_col": target_col,
        "top_n_features": 10,
    })
    return eda_result if isinstance(eda_result, dict) else {"error": "EDA failed"}


def _viz_subagent(predictions: list[dict], state: str, verify: dict | None = None) -> list[dict]:
    """Subagent 5b: Generate notebook-style Plotly charts from predictions."""
    from src.agent.tools.chart_generator import create_bar_chart, create_chart, create_choropleth_map

    charts = []

    if not predictions or (predictions and "error" in predictions[0]):
        return charts

    y_key = "predicted_risk" if "predicted_risk" in predictions[0] else "risk_score"

    # Chart 1: Horizontal bar — top at-risk counties (NEW1/NEW2 style)
    bar = create_bar_chart.invoke({
        "title": f"Top At-Risk Counties — {state}",
        "data_json": json.dumps(predictions[:15]),
        "x_col": "county",
        "y_col": y_key,
        "color_col": y_key,
        "horizontal": True,
    })
    if isinstance(bar, dict) and "plotly_spec" in bar:
        charts.append(bar)

    # Chart 2: Box plot of risk score distribution
    box = create_chart.invoke({
        "chart_type": "box",
        "title": f"Risk Score Distribution — {state}",
        "data_json": json.dumps(predictions[:50]),
        "y_col": y_key,
    })
    if isinstance(box, dict) and "plotly_spec" in box:
        charts.append(box)

    # Chart 3: Histogram of risk scores (NEW1 composite risk distribution)
    hist = create_chart.invoke({
        "chart_type": "histogram",
        "title": f"Distribution of Predicted Risk Scores — {state}",
        "data_json": json.dumps(predictions),
        "x_col": y_key,
        "nbins": 20,
    })
    if isinstance(hist, dict) and "plotly_spec" in hist:
        charts.append(hist)

    # Chart 4: Choropleth map if FIPS codes available (NEW1 national vulnerability map style)
    has_fips = any(p.get("fips") for p in predictions[:5])
    if has_fips:
        choro_data = [
            {"fips": str(p.get("fips", "")).zfill(5), y_key: p.get(y_key, 0), "county": p.get("county", "")}
            for p in predictions if p.get("fips")
        ]
        choro = create_choropleth_map.invoke({
            "title": f"County Risk Map — {state} (Red = High Risk)",
            "data_json": json.dumps(choro_data),
            "fips_col": "fips",
            "value_col": y_key,
            "text_col": "county",
            "colorscale": "RdYlGn_r",
        })
        if isinstance(choro, dict) and "plotly_spec" in choro:
            charts.append(choro)

    # Chart 5: Feature importance bar chart (NEW1/NEW2 style)
    if verify and isinstance(verify, dict):
        fi = verify.get("feature_importance", {})
        if isinstance(fi, dict) and fi.get("top_features"):
            fi_data = [
                {"feature": f["feature"], "importance": f["importance"]}
                for f in fi["top_features"][:15]
            ]
            fi_chart = create_bar_chart.invoke({
                "title": f"Top Risk Drivers — SHAP Feature Importance ({state})",
                "data_json": json.dumps(fi_data),
                "x_col": "feature",
                "y_col": "importance",
                "color_col": "importance",
                "horizontal": True,
            })
            if isinstance(fi_chart, dict) and "plotly_spec" in fi_chart:
                charts.append(fi_chart)

    # Chart 6: Scatter — predicted vs CI interval (confidence bands)
    has_ci = any(p.get("ci_lower") is not None for p in predictions[:5])
    if has_ci:
        scatter_data = [
            {
                "county": p.get("county", ""),
                y_key: p.get(y_key, 0),
                "ci_lower": p.get("ci_lower", 0),
                "ci_upper": p.get("ci_upper", 0),
            }
            for p in predictions[:20]
        ]
        scatter = create_chart.invoke({
            "chart_type": "scatter",
            "title": f"Predicted Risk with 95% CI — {state}",
            "data_json": json.dumps(scatter_data),
            "x_col": "county",
            "y_col": y_key,
            "text_col": "county",
        })
        if isinstance(scatter, dict) and "plotly_spec" in scatter:
            charts.append(scatter)

    return charts


def _analysis_subagent(
    predictions: list[dict],
    metrics: dict,
    importance: dict,
    anomalies: list,
    web_research: list,
    scenario: str,
    state: str,
    eda: dict | None = None,
) -> str:
    """Subagent 6: Synthesize a narrative analysis from all results.

    This is pure Python — no LLM call. The main synthesizer node will
    reformat this with an LLM for final presentation.
    """
    lines = [f"## Analytics Report — {state} ({scenario} scenario)\n"]

    # EDA summary
    if eda and eda.get("n_rows"):
        lines.append("### Exploratory Data Analysis")
        lines.append(f"- Dataset: {eda['n_rows']} rows × {eda['n_cols']} columns")
        lines.append(f"- Missing data: {eda.get('missing_pct', 0)}%")
        if eda.get("recommendations"):
            for rec in eda["recommendations"]:
                lines.append(f"- {rec}")
        if eda.get("target_correlations"):
            lines.append("\n**Top correlations with target:**")
            for tc in eda["target_correlations"][:5]:
                lines.append(f"  - {tc['feature']}: r={tc['correlation']}")
        lines.append("")

    # Model performance
    if metrics:
        lines.append("### Model Performance")
        for k, v in metrics.items():
            lines.append(f"- **{k.upper()}**: {v}")
        lines.append("")

    # Top predictions
    if predictions and "error" not in predictions[0]:
        lines.append("### Highest-Risk Counties")
        for i, p in enumerate(predictions[:10], 1):
            county = p.get("county", "Unknown")
            score = p.get("predicted_risk", p.get("risk_score", "N/A"))
            ci_lo = p.get("ci_lower", "")
            ci_hi = p.get("ci_upper", "")
            ci_str = f" (95% CI: {ci_lo}–{ci_hi})" if ci_lo and ci_hi else ""
            lines.append(f"{i}. **{county}**: risk score {score}{ci_str}")
        lines.append("")

    # Feature importance
    if isinstance(importance, dict) and "top_features" in importance:
        lines.append("### Key Risk Drivers (SHAP)")
        for feat in importance["top_features"][:5]:
            lines.append(f"- {feat['feature']}: importance {feat['importance']:.3f}")
        if importance.get("interpretation"):
            lines.append(f"\n{importance['interpretation']}")
        lines.append("")

    # Anomalies
    if anomalies and isinstance(anomalies, list) and anomalies and "error" not in anomalies[0]:
        lines.append("### Anomalous Counties")
        for a in anomalies[:5]:
            county = a.get("county", "Unknown")
            devs = ", ".join(f"{d['feature']} (z={d['z_score']})" for d in a.get("top_deviations", [])[:3])
            lines.append(f"- **{county}**: {devs}")
        lines.append("")

    # Web research
    if web_research and isinstance(web_research, list):
        lines.append("### Emerging Threats (Web Research)")
        for r in web_research[:5]:
            if isinstance(r, dict) and r.get("title"):
                url = f" — [{r.get('source', 'link')}]({r.get('url', '')})" if r.get("url") else ""
                lines.append(f"- {r['title'][:120]}{url}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main tool — registered with ToolNode
# ---------------------------------------------------------------------------

@tool
def run_analytics_pipeline(
    state: str = "MO",
    scenario: str = "drought",
    yield_reduction_pct: float = 20.0,
    model_type: str = "xgboost",
    target_col: str = "FOODINSEC_15_17",
    top_n: int = 15,
    search_query: str = "",
    pipeline: str = "full_analysis",
) -> dict[str, Any]:
    """Run the full AgriFlow analytics pipeline with 6 coordinated subagents.

    Orchestrates: web research → data preparation → ML training → prediction →
    verification (CCC/SHAP) → visualization → narrative analysis.

    This is the recommended tool for comprehensive risk assessments. For quick
    predictions, use run_prediction or predict_risk directly.

    Args:
        state: Two-letter state code.
        scenario: "baseline", "drought", or "price_shock".
        yield_reduction_pct: Simulated crop yield drop percentage.
        model_type: "xgboost" or "random_forest".
        target_col: Variable to predict (default food insecurity rate).
        top_n: Number of top-risk counties.
        search_query: Custom web search query (auto-generated if empty).
        pipeline: "full_analysis", "quick_predict", or "risk_scan".

    Returns:
        Dict with predictions, metrics, feature importance, anomalies,
        charts, web research, analysis narrative, and data sources.
    """
    sources: list[str] = []
    analytics_report: dict[str, Any] = {
        "pipeline": pipeline,
        "state": state,
        "scenario": scenario,
        "model_type": model_type,
    }

    # ── Phase 1: Research + Data in PARALLEL ─────────────────
    research = {"web_research": [], "sources": []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if pipeline in ("full_analysis", "risk_scan"):
            futures["research"] = pool.submit(_research_subagent, state, search_query)
        futures["data"] = pool.submit(_data_subagent, state)

        for key, fut in futures.items():
            try:
                result = fut.result(timeout=30)
                if key == "research":
                    research = result
                elif key == "data":
                    data = result
            except Exception:
                if key == "data":
                    data = {"n_counties": 0, "n_features": 0, "sources": []}

    analytics_report["web_research"] = research.get("web_research", [])
    sources.extend(research.get("sources", []))
    analytics_report["n_counties"] = data["n_counties"]
    analytics_report["n_features"] = data["n_features"]
    sources.extend(data.get("sources", []))

    # ── Phase 2: EDA + ML training in PARALLEL ─────────────────
    eda_result: dict[str, Any] = {}
    ml: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        ml_fut = pool.submit(
            _ml_subagent, state, model_type, target_col, scenario,
            yield_reduction_pct, top_n,
        )
        if pipeline in ("full_analysis",):
            eda_fut = pool.submit(_eda_subagent, state, target_col)
        else:
            eda_fut = None

        try:
            ml = ml_fut.result(timeout=120)
        except Exception as e:
            ml = {"error": str(e), "predictions": [], "model_info": {}}

        if eda_fut is not None:
            try:
                eda_result = eda_fut.result(timeout=60)
            except Exception:
                eda_result = {}

    if "error" in ml:
        analytics_report["error"] = ml["error"]
        analytics_report["sources"] = sources
        analytics_report["eda"] = eda_result
        return analytics_report

    analytics_report["predictions"] = ml["predictions"]
    analytics_report["model_info"] = {
        "type": model_type,
        "metrics": ml["model_info"].get("metrics", {}),
        "cv_r2": ml["model_info"].get("cv_r2", "N/A"),
        "n_features": ml["model_info"].get("n_features", 0),
        "model_path": ml["model_info"].get("model_path", ""),
    }
    analytics_report["eda"] = {
        "n_rows": eda_result.get("n_rows", 0),
        "n_cols": eda_result.get("n_cols", 0),
        "missing_pct": eda_result.get("missing_values", {}).get("total_pct", 0),
        "distributions": eda_result.get("distributions", {}),
        "top_correlations": eda_result.get("correlations", {}).get("top_pairs", [])[:5],
        "target_correlations": eda_result.get("target_correlations", [])[:5],
        "outliers": eda_result.get("outliers", {}),
        "recommendations": eda_result.get("recommendations", []),
    }
    sources.extend(ml.get("sources", []))
    if eda_result:
        sources.append("Automated EDA pipeline")

    # ── Phase 3: Verification + Visualization in PARALLEL ──────
    verify = {}
    charts_list: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        verify_fut = pool.submit(
            _verify_subagent, ml["model_info"], state, model_type, target_col,
        )
        if pipeline in ("full_analysis", "risk_scan"):
            # Pass verify results to viz for feature importance chart (may be empty at this point)
            viz_fut = pool.submit(_viz_subagent, ml["predictions"], state, {})
        else:
            viz_fut = None

        try:
            verify = verify_fut.result(timeout=60)
        except Exception:
            verify = {"metrics": {}, "feature_importance": {}, "anomalies": []}

        if viz_fut is not None:
            try:
                charts_list = viz_fut.result(timeout=30)
            except Exception:
                charts_list = []

    # Merge EDA charts into the main chart list
    eda_charts = eda_result.get("charts", [])
    if isinstance(eda_charts, list):
        charts_list.extend(eda_charts)

    analytics_report["evaluation"] = verify.get("metrics", {})
    analytics_report["feature_importance"] = verify.get("feature_importance", {})
    analytics_report["anomalies"] = verify.get("anomalies", [])

    # Add feature importance chart now that verify is done
    if verify.get("feature_importance") and pipeline in ("full_analysis", "risk_scan"):
        try:
            fi_charts = _viz_subagent(ml["predictions"], state, verify)
            # Avoid duplicating non-importance charts — only take new ones
            existing_types = {c.get("chart_type") for c in charts_list}
            for c in fi_charts:
                if c.get("chart_type") not in existing_types:
                    charts_list.append(c)
                elif "Feature Importance" in c.get("title", ""):
                    charts_list.append(c)
        except Exception:
            pass

    analytics_report["charts"] = charts_list

    # ── Phase 4: Analysis narrative (pure Python, fast) ────────
    narrative = _analysis_subagent(
        predictions=ml["predictions"],
        metrics=verify.get("metrics", {}),
        importance=verify.get("feature_importance", {}),
        anomalies=verify.get("anomalies", []),
        web_research=analytics_report.get("web_research", []),
        scenario=scenario,
        state=state,
        eda=analytics_report.get("eda"),
    )
    analytics_report["analysis_report"] = narrative
    analytics_report["sources"] = sources
    analytics_report["confidence"] = _assess_confidence(ml["model_info"])

    return analytics_report


def _assess_confidence(model_info: dict) -> str:
    """Assess overall confidence level based on model metrics."""
    metrics = model_info.get("metrics", {})
    r2 = metrics.get("r2", 0)
    ccc = metrics.get("ccc", 0)

    if r2 > 0.85 and ccc > 0.8:
        return "high"
    elif r2 > 0.7 and ccc > 0.6:
        return "medium"
    return "low"
