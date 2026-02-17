"""Push all AgriFlow skills to Archia via REST API.

Skills define high-level capabilities that route to specialized agents.
Each skill has instructions the agent follows when invoked.

Usage:
    python archia/setup_skills.py          # Create/update all skills
    python archia/setup_skills.py --verify # List existing skills
    python archia/setup_skills.py --clean  # Delete all AgriFlow skills
"""

from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("ARCHIA_BASE_URL", "https://registry.archia.app/v1")
TOKEN = os.getenv("ARCHIA_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# ── Skill definitions ────────────────────────────────────────────────
# Each skill maps to an agent and provides detailed instructions.
# Skills are organized by the 5 routing categories + 1 planning skill.

SKILLS = [
    # ── Planning ──────────────────────────────────────────────────────
    {
        "name": "AgriFlow Query Planner",
        "description": "Decompose complex food supply chain questions into categorized sub-tasks for multi-agent execution",
        "agent": "agriflow-planner",
        "instructions": (
            "You are the AgriFlow query planner. When a user asks a complex question "
            "about food supply chains, food insecurity, crop production, or distribution "
            "logistics, decompose it into concrete sub-tasks.\n\n"
            "Tag each sub-task with a category:\n"
            "- [data] for USDA food atlas, NASS crops, weather, FEMA disasters, Census demographics\n"
            "- [sql] for custom database queries against the AgriFlow SQLite database\n"
            "- [ml] for risk predictions, scenario comparisons, and model evaluation\n"
            "- [viz] for bar charts, line charts, scatter maps, and risk heatmaps\n"
            "- [route] for delivery route optimization, distance calculations, and scheduling\n\n"
            "Rules:\n"
            "1. Always query data BEFORE generating visualizations\n"
            "2. Always optimize routes BEFORE creating route maps\n"
            "3. Use specific tool names and parameters in each sub-task\n"
            "4. Return a JSON array of {category, task, tool, params} objects"
        ),
    },
    # ── Data Retrieval (7 tools) ──────────────────────────────────────
    {
        "name": "Food Insecurity Analysis",
        "description": "Query USDA Food Environment Atlas for county-level food insecurity rates, SNAP participation, poverty, and food access indicators",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_food_atlas` to retrieve county-level food environment data. "
            "Parameters: state (2-letter code, e.g. 'MO'), county (partial name match), "
            "columns (default includes FOODINSEC_15_17, PCT_LACCESS_POP15, POVRATE15, "
            "PCT_SNAP16, PCT_OBESE_ADULTS17). Return raw data without analysis — "
            "the synthesizer handles interpretation. Focus on Missouri counties by default."
        ),
    },
    {
        "name": "Food Desert Mapping",
        "description": "Query USDA Food Access Research Atlas for census-tract food desert classifications, distance to supermarkets, and low-access populations",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_food_access` to retrieve census-tract level food desert data. "
            "Parameters: state (2-letter code), county (partial match), "
            "urban_rural ('Urban' or 'Rural'). Returns food desert flags at 1-mile and "
            "10-mile thresholds, low-access population counts, and vehicle access data. "
            "Useful for identifying underserved areas for food distribution planning."
        ),
    },
    {
        "name": "Crop Production Data",
        "description": "Query USDA NASS Quick Stats for county-level crop yields, production volumes, acreage, and livestock statistics",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_nass` to retrieve agricultural production data from USDA NASS. "
            "Parameters: commodity (e.g. 'CORN', 'SOYBEANS', 'WHEAT'), state (default 'MO'), "
            "year (specific year or empty for recent), stat_type ('YIELD', 'PRODUCTION', "
            "'AREA HARVESTED'), agg_level ('COUNTY' or 'STATE'). "
            "Requires NASS_API_KEY. Results capped at 100 rows."
        ),
    },
    {
        "name": "Weather Forecast",
        "description": "Get 7-day weather forecasts for Missouri counties to assess agricultural and delivery conditions",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_weather` to get weather forecasts. Parameters: county (Missouri county name "
            "like 'Wayne County' or 'Pemiscot County'), or latitude/longitude for precise location. "
            "Returns daily temp_max_f, temp_min_f, precipitation_inches, rain_inches. "
            "Pre-seeded counties: Wayne, Pemiscot, Oregon, Dunklin, Ripley, Shannon, Mississippi, "
            "New Madrid, Boone, Jackson. Uses Open-Meteo (free, no API key)."
        ),
    },
    {
        "name": "Disaster History",
        "description": "Query FEMA disaster declarations for historical floods, storms, droughts, and emergency events by state and year",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_fema_disasters` to retrieve historical disaster data. "
            "Parameters: state ('MO' or 'Missouri'), disaster_type ('DR' for major disasters, "
            "'EM' for emergencies, 'FM' for fire, '' for all), year_start (default 2018), "
            "limit (default 50, max 100). Returns disaster declarations with incident type, "
            "affected counties, and assistance programs. "
            "Free FEMA API, no key required."
        ),
    },
    {
        "name": "Census Demographics",
        "description": "Query US Census ACS for county demographics, income, poverty rates, vehicle access, and unemployment data",
        "agent": "agriflow-data",
        "instructions": (
            "Use `query_census_acs` to retrieve demographic data. "
            "Parameters: state (default 'MO'), year (2019-2022, default 2022), "
            "county_fips (3-digit code, '' for all counties). "
            "Returns: median_household_income, total_population, no_vehicle_pct, "
            "poverty_rate_pct, unemployment_rate_pct, median_age, housing data. "
            "Supported states: MO, IL, AR, KS, TN, KY, OK, IA. Free Census API."
        ),
    },
    # ── ML / Prediction (3 legacy tools) ──────────────────────────────
    {
        "name": "Risk Prediction",
        "description": "Run food insecurity risk predictions for counties under baseline, drought, or price shock scenarios",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `run_prediction` to compute county-level risk scores. "
            "Parameters: state (default 'MO'), scenario ('baseline', 'drought', 'price_shock'), "
            "yield_reduction_pct (0-100, for drought scenarios), top_n (default 10), "
            "model_type ('auto', 'heuristic', 'xgboost', 'random_forest'). "
            "Auto mode uses a trained ML model if available, otherwise falls back to heuristic. "
            "Returns counties ranked by risk_score with the highest-risk counties first. "
            "For comprehensive analysis, use run_analytics_pipeline instead."
        ),
    },
    {
        "name": "Model Evaluation",
        "description": "Compute RMSE, MAE, R-squared, CCC, accuracy, precision, recall, and F1 metrics for prediction evaluation",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `compute_evaluation_metrics` to evaluate prediction quality. "
            "Parameters: predictions_json (JSON array of predicted values), "
            "actuals_json (JSON array of ground truth), pred_col, actual_col, "
            "task_type ('regression' for RMSE/MAE/R²/CCC or 'classification' for accuracy/F1). "
            "Always pair with run_prediction when ground truth data is available. "
            "For standalone CCC computation, use `compute_ccc`."
        ),
    },
    {
        "name": "Scenario Comparison",
        "description": "Compare multiple what-if scenarios side-by-side with statistical summaries to identify highest-risk outcomes",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `compare_scenarios` to compare outputs from multiple prediction runs. "
            "Parameters: scenarios_json (JSON array combining all scenario results), "
            "metric_col (column to compare, e.g. 'risk_score'), "
            "label_col (scenario identifier, default 'scenario'). "
            "Returns per-scenario stats (count, mean, max, min, std_dev) and identifies "
            "the highest_risk_scenario. Run run_prediction for each scenario first, "
            "then combine results for comparison."
        ),
    },
    # ── Analytics (6 new tools) ────────────────────────────────────────
    {
        "name": "Train Risk Model",
        "description": "Train XGBoost or Random Forest risk model on county-level food environment and census data with cross-validation",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `train_risk_model` to train a real ML model. "
            "Parameters: state (default 'MO'), model_type ('xgboost' or 'random_forest'), "
            "target_col (default 'food_insecurity_rate'), test_size (default 0.2). "
            "Automatically builds feature matrix from food_environment + food_access data, "
            "trains with 5-fold cross-validation, and saves model to models/ directory. "
            "Returns R², RMSE, MAE, CCC metrics. Run build_feature_matrix first if you "
            "need to inspect features before training."
        ),
    },
    {
        "name": "Predict Agricultural Risk",
        "description": "Run trained XGBoost/RF model on counties to predict risk scores with confidence intervals under scenarios",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `predict_risk` to run inference with a trained model. "
            "Parameters: state (default 'MO'), model_type ('xgboost' or 'random_forest'), "
            "target_col (default 'food_insecurity_rate'), scenario ('baseline', 'drought', "
            "'price_shock'), yield_reduction_pct (0-100), top_n (default 10). "
            "Requires a trained model (run train_risk_model first). "
            "Returns ranked counties with predicted_risk and confidence intervals."
        ),
    },
    {
        "name": "Feature Analysis",
        "description": "SHAP-based feature importance analysis explaining which factors drive food insecurity risk predictions",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `get_feature_importance` for SHAP explanations. "
            "Parameters: state (default 'MO'), model_type ('xgboost' or 'random_forest'), "
            "target_col (default 'food_insecurity_rate'), top_n (default 10). "
            "Requires a trained model. Returns ranked features with importance scores "
            "and human-readable interpretations (e.g., 'Higher POVRATE15 increases risk'). "
            "Falls back to model.feature_importances_ if SHAP is unavailable."
        ),
    },
    {
        "name": "Anomaly Detection",
        "description": "Flag counties with unusual food insecurity patterns using Isolation Forest outlier detection",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `detect_anomalies` to identify outlier counties. "
            "Parameters: state (default 'MO'), contamination (expected outlier fraction, "
            "default 0.1 = 10%), top_n (max anomalies to return, default 10). "
            "Uses Isolation Forest on standardized features. Returns anomalous counties "
            "with anomaly_score and z-score deviations showing which features are unusual. "
            "High |z-score| features explain why a county is flagged as an outlier."
        ),
    },
    {
        "name": "Agricultural Threat Search",
        "description": "Search the web for emerging agricultural risks including pest outbreaks, crop diseases, and severe weather threats",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `web_search_risks` to find emerging agricultural threats. "
            "Parameters: query (search terms, e.g. 'Missouri corn drought 2026'), "
            "region (default 'Missouri'). "
            "Searches DuckDuckGo for real-time information on pest outbreaks, crop diseases, "
            "livestock health alerts, and severe weather that could impact food supply chains. "
            "Returns structured results with titles, summaries, and source URLs."
        ),
    },
    {
        "name": "Full Analytics Pipeline",
        "description": "Run complete ML analytics pipeline: data collection, model training, prediction, validation, visualization, and analysis",
        "agent": "agriflow-analytics",
        "instructions": (
            "Use `run_analytics_pipeline` for comprehensive automated analysis. "
            "Parameters: state (default 'MO'), scenario ('baseline', 'drought', 'price_shock'), "
            "yield_reduction_pct (0-100), model_type ('xgboost' or 'random_forest'), "
            "target_col (default 'food_insecurity_rate'), top_n (default 10), "
            "search_query (web search terms), pipeline ('full_analysis', 'quick_predict', 'risk_scan'). "
            "Orchestrates 6 subagents: research → data → ML → verification → visualization → analysis. "
            "Returns complete analytics report with predictions, metrics, SHAP importance, "
            "anomalies, web research, charts, and narrative analysis. "
            "Prefer this tool for comprehensive questions requiring multiple data sources and ML."
        ),
    },
    # ── Visualization (4 tools) ───────────────────────────────────────
    {
        "name": "Bar Chart",
        "description": "Create Plotly bar charts for county rankings, metric comparisons, and data distributions",
        "agent": "agriflow-viz",
        "instructions": (
            "Use `create_bar_chart` to generate bar chart visualizations. "
            "Parameters: title (descriptive), data_json (JSON array of records), "
            "x_col (category axis, e.g. 'County'), y_col (value axis, e.g. 'FOODINSEC_15_17'), "
            "color_col (optional, continuous value for color scale), horizontal (boolean). "
            "Data must be queried first using data tools. Pass data as a JSON string. "
            "Dark theme is applied automatically. Returns Plotly JSON spec for frontend rendering."
        ),
    },
    {
        "name": "Line Chart",
        "description": "Create Plotly line charts for time series trends, yield changes, and multi-metric tracking",
        "agent": "agriflow-viz",
        "instructions": (
            "Use `create_line_chart` for trend visualization. "
            "Parameters: title, data_json (JSON array), x_col (date/year column), "
            "y_cols (comma-separated column names for multiple lines). "
            "Supports up to 5 lines with distinct colors. Data must be queried first. "
            "Returns Plotly JSON spec with dark theme."
        ),
    },
    {
        "name": "Geographic Map",
        "description": "Create interactive Plotly scatter maps for food desert overlays, risk assessment, and county-level geographic visualization",
        "agent": "agriflow-viz",
        "instructions": (
            "Use `create_scatter_map` to generate geographic scatter maps. "
            "Parameters: title, data_json (JSON array with coordinates), "
            "lat_col (latitude column), lon_col (longitude column), "
            "size_col (optional, marker size by value), color_col (optional, red=high risk), "
            "text_col (hover label column), zoom (default 6 for Missouri). "
            "Uses Plotly scattermapbox with carto-darkmatter dark map tiles. "
            "Data with lat/lon must be queried first. Returns Plotly JSON spec."
        ),
    },
    {
        "name": "Risk Heatmap",
        "description": "Create risk assessment heatmaps showing county-vs-factor vulnerability matrices",
        "agent": "agriflow-viz",
        "instructions": (
            "Use `create_risk_heatmap` for matrix-style risk visualization. "
            "Parameters: title, data_json (JSON array), x_col (factor column), "
            "y_col (county/entity column), z_col (value/score column). "
            "Uses RdYlGn_r colorscale (red = high risk, green = low risk). "
            "Ideal for comparing multiple risk factors across multiple counties. "
            "Data must be queried and structured first. Returns Plotly JSON spec."
        ),
    },
    # ── Route / Logistics (4 tools) ───────────────────────────────────
    {
        "name": "Route Optimization",
        "description": "Optimize food delivery routes across Missouri using nearest-neighbor TSP with 22 pre-seeded locations",
        "agent": "agriflow-logistics",
        "instructions": (
            "Use `optimize_delivery_route` to compute optimal delivery paths. "
            "Parameters: origin (starting location name, e.g. 'Cape Girardeau Hub'), "
            "destinations (comma-separated location names, e.g. 'Wayne County,Pemiscot County,Dunklin County'), "
            "custom_locations_json (optional JSON array with {name, lat, lon} for unlisted locations). "
            "22 pre-seeded Missouri locations include county seats and distribution hubs: "
            "St. Louis Food Bank, Springfield Hub, Cape Girardeau Hub, Columbia Hub, "
            "Joplin Hub, Sikeston Hub, and 16 county locations. "
            "Uses nearest-neighbor heuristic with haversine distances and 45 mph rural speed. "
            "Returns optimized_route with legs, total_distance_miles, and est_total_drive_minutes."
        ),
    },
    {
        "name": "Distance Calculator",
        "description": "Calculate straight-line distance and estimated drive time between two Missouri locations",
        "agent": "agriflow-logistics",
        "instructions": (
            "Use `calculate_distance` for point-to-point distance calculations. "
            "Parameters: location_a and location_b (pre-seeded location names). "
            "Returns distance_miles (haversine) and est_drive_minutes (at 45 mph average). "
            "Use for quick distance checks before full route optimization."
        ),
    },
    {
        "name": "Route Map",
        "description": "Generate interactive Plotly maps showing delivery routes with numbered stop markers and route lines",
        "agent": "agriflow-logistics",
        "instructions": (
            "Use `create_route_map` to visualize an optimized delivery route. "
            "Parameters: route_json (JSON string from optimize_delivery_route output). "
            "Returns Plotly scattermapbox with route lines connecting stops and "
            "numbered/colored markers at each delivery point. "
            "Always run optimize_delivery_route first, then pass its output here. "
            "Dark map theme (carto-darkmatter) applied automatically."
        ),
    },
    {
        "name": "Delivery Schedule",
        "description": "Generate time-based delivery schedules with loading/unloading windows for each stop",
        "agent": "agriflow-logistics",
        "instructions": (
            "Use `schedule_deliveries` to create a time-based delivery plan. "
            "Parameters: route_json (from optimize_delivery_route), "
            "start_time (default '08:00'), loading_minutes (default 30, at origin), "
            "unloading_minutes (default 20, at each delivery stop). "
            "Returns schedule with arrive/depart times for each stop, "
            "est_end_time, total_stops, and total_distance_miles. "
            "Always run optimize_delivery_route first to get the route_json input."
        ),
    },
    # ── Ingest (5 tools) ──────────────────────────────────────────────
    {
        "name": "List Database Tables",
        "description": "List all tables in agriflow.db with row counts and column counts for exploration and schema discovery",
        "agent": "agriflow-ingest",
        "instructions": (
            "Use `list_db_tables` to enumerate all tables currently in agriflow.db. "
            "No parameters needed. Returns table name, row count, and column count for each table. "
            "Run this first to understand what data is already loaded before fetching new datasets."
        ),
    },
    {
        "name": "Profile Dataset",
        "description": "Fetch a CSV or Excel file from a URL or local path and return an EDA profile without loading it into the database",
        "agent": "agriflow-ingest",
        "instructions": (
            "Use `fetch_and_profile_csv` to preview an external dataset before loading. "
            "Parameters: url_or_path (URL or local file path to CSV or Excel), "
            "state_filter (optional 2-letter state code to filter, default 'MO'), "
            "sample_rows (rows to show in preview, default 5). "
            "Returns shape, column names, dtypes, null counts, numeric summary stats, and a sample. "
            "Use this to validate a dataset is suitable before calling load_dataset."
        ),
    },
    {
        "name": "Load Dataset",
        "description": "Fetch, clean, and load a CSV or Excel dataset from a URL or file path into agriflow.db as a new table",
        "agent": "agriflow-ingest",
        "instructions": (
            "Use `load_dataset` to ingest a new dataset into agriflow.db. "
            "Parameters: url_or_path (URL or local file path), "
            "table_name (destination table name, use convention {source}_{topic}_{year}), "
            "state_filter (optional 2-letter state code, default None = load all states), "
            "replace (whether to replace existing table, default False), "
            "sentinel_value (value to treat as NULL, default -9999.0). "
            "Automatically normalizes state name columns, strips 'County' suffix, "
            "replaces sentinel values with NULL, and creates indexes on State, County, FIPS, Year. "
            "Always profile with fetch_and_profile_csv first. Returns row count loaded and columns."
        ),
    },
    {
        "name": "Explore Table",
        "description": "Run EDA queries on any table in agriflow.db — summary stats, missing value analysis, county samples, or state-filtered previews",
        "agent": "agriflow-ingest",
        "instructions": (
            "Use `run_eda_query` to explore any table in agriflow.db. "
            "Parameters: table (table name), state (2-letter code, default 'MO'), "
            "query_type ('summary' for descriptive stats, 'missing' for null counts, "
            "'sample' for data preview, 'counties' for county list). "
            "Use after loading a new dataset to verify contents and understand its structure. "
            "Useful for diagnosing data quality issues or planning downstream analysis."
        ),
    },
    # ── SQL / Database (2 tools) ──────────────────────────────────────
    {
        "name": "Database Explorer",
        "description": "Discover AgriFlow database schema — list tables, columns, and row counts for query planning",
        "agent": "AGRIFLOW_SYSTEM",
        "instructions": (
            "Use `list_tables` to discover the AgriFlow SQLite database schema. "
            "No parameters needed. Returns all table names with their columns and row counts. "
            "Run this first before writing custom SQL queries to understand available data. "
            "The database contains USDA Food Environment Atlas, Food Access Research Atlas, "
            "and other agricultural datasets."
        ),
    },
    {
        "name": "Custom SQL Query",
        "description": "Execute read-only SQL queries against the AgriFlow database for ad-hoc data analysis",
        "agent": "AGRIFLOW_SYSTEM",
        "instructions": (
            "Use `run_sql_query` for custom data queries. "
            "Parameters: query (SQL SELECT/WITH/PRAGMA/EXPLAIN only — no writes allowed), "
            "limit (default 100, max 500). Auto-appends LIMIT if missing. "
            "Always run list_tables first to understand the schema. "
            "Use for complex joins, aggregations, or filters not covered by the specialized "
            "data tools. Returns results as a list of dictionaries."
        ),
    },
]


def _get(path: str) -> httpx.Response:
    return httpx.get(f"{BASE}{path}", headers=HEADERS, timeout=15)


def _post(path: str, data: dict) -> httpx.Response:
    return httpx.post(f"{BASE}{path}", headers=HEADERS, json=data, timeout=15)


def _delete(path: str) -> httpx.Response:
    return httpx.delete(f"{BASE}{path}", headers=HEADERS, timeout=15)


def setup_all() -> None:
    """Create all skills (delete existing first to ensure clean state)."""
    # Get existing skills to avoid duplicates
    r = _get("/skill")
    existing = {s["name"] for s in r.json().get("skills", [])} if r.status_code == 200 else set()

    created = 0
    skipped = 0
    failed = 0

    for skill in SKILLS:
        name = skill["name"]

        # Delete existing skill with same name to update it
        if name in existing:
            _delete(f"/skill/{name}")

        r = _post("/skill", skill)
        if r.status_code in (200, 201):
            agent = skill.get("agent", "none")
            print(f"  created   {name:30s}  -> {agent}")
            created += 1
        else:
            print(f"  FAILED    {name:30s}  ({r.status_code}: {r.text[:100]})")
            failed += 1

    print(f"\n  {created} created, {failed} failed, {len(SKILLS)} total skills defined.")


def verify() -> None:
    """List all skills on the platform."""
    r = _get("/skill")
    skills = r.json().get("skills", [])

    if not skills:
        print("  No skills found. Run: python archia/setup_skills.py")
        return

    print(f"  {len(skills)} skills on platform:\n")
    for s in skills:
        print(f"  {s['name']:35s}  v{s.get('version', '?'):6s}  src={s.get('source', '?')}")

    # Check against our definitions
    existing_names = {s["name"] for s in skills}
    missing = [s["name"] for s in SKILLS if s["name"] not in existing_names]
    if missing:
        print(f"\n  Missing {len(missing)} skills: {', '.join(missing)}")
    else:
        print(f"\n  All {len(SKILLS)} AgriFlow skills present.")


def clean() -> None:
    """Delete all AgriFlow skills."""
    for skill in SKILLS:
        name = skill["name"]
        r = _delete(f"/skill/{name}")
        status = "deleted" if r.status_code in (200, 204) else f"skip ({r.status_code})"
        print(f"  {status:10s}  {name}")


if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: ARCHIA_TOKEN not set in .env")
        sys.exit(1)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "--setup"

    if cmd == "--verify":
        verify()
    elif cmd == "--clean":
        clean()
    else:
        print("Pushing AgriFlow skills to Archia...\n")
        setup_all()
