"""Chart validator node — validates and fixes Plotly specs before they reach the frontend.

Catches common issues:
- Empty/null data in traces
- Missing or unreadable axis labels (raw column names → human-readable)
- Swapped x/y axes (labels on value axis, numbers on category axis)
- Missing titles, colorbar labels
- Invalid geo coordinates or FIPS codes
- Malformed trace structures that would cause Plotly.js errors
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage

# Chart tool names that produce Plotly specs (must match main.py _CHART_TOOLS)
_CHART_TOOLS = {
    "create_bar_chart", "create_line_chart", "create_scatter_map",
    "create_risk_heatmap", "create_route_map", "create_choropleth_map",
    "create_chart",
}

# ---------------------------------------------------------------------------
# Human-readable column name mapping
# ---------------------------------------------------------------------------
_COLUMN_LABELS: dict[str, str] = {
    # Food Atlas columns
    "foodinsec_13_15": "Food Insecurity Rate (%)",
    "foodinsec_21_23": "Food Insecurity Rate (%)",
    "foodinsec_rate": "Food Insecurity Rate (%)",
    "ch_foodinsec": "Child Food Insecurity Rate (%)",
    "vlfoodsec_13_15": "Very Low Food Security (%)",
    "povrate15": "Poverty Rate (%)",
    "poverty_rate": "Poverty Rate (%)",
    "medhhinc15": "Median Household Income ($)",
    "median_income": "Median Household Income ($)",
    "snapspth16": "SNAP Stores per 1K Pop",
    "snap_participation": "SNAP Participation Rate (%)",
    "pct_snap17": "SNAP Participation (%)",
    "grocpth16": "Grocery Stores per 1K Pop",
    "convspth16": "Convenience Stores per 1K Pop",
    "ffrpth16": "Fast Food per 1K Pop",
    "pct_laccess_pop15": "Low Food Access (%)",
    "laccess_pop15": "Low Access Population",
    "pct_diabetes_adults13": "Diabetes Rate (%)",
    "pct_obese_adults13": "Obesity Rate (%)",
    "recfacpth16": "Recreation Facilities per 1K",
    "pc_ffrsales12": "Fast Food Sales per Capita ($)",
    "pc_fsrsales12": "Full-Service Restaurant Sales per Capita ($)",
    # Census / demographic
    "population": "Population",
    "pop2010": "Population (2010)",
    "unemployment": "Unemployment Rate (%)",
    "pct_no_vehicle": "No Vehicle Access (%)",
    "vehicle_access": "Vehicle Access (%)",
    # NASS / crop columns
    "corn_yield": "Corn Yield (bu/acre)",
    "soybean_yield": "Soybean Yield (bu/acre)",
    "wheat_yield": "Wheat Yield (bu/acre)",
    "production": "Production",
    "yield": "Yield",
    "acreage": "Acreage",
    "value": "Value ($)",
    # Risk / ML columns
    "risk_score": "Risk Score",
    "predicted_risk": "Predicted Risk Score",
    "composite_risk": "Composite Risk Score",
    "food_desert_risk": "Food Desert Risk Score",
    "importance": "Feature Importance",
    "anomaly_score": "Anomaly Score",
    "shap_value": "SHAP Value",
    # Geographic
    "county": "County",
    "county_name": "County",
    "state": "State",
    "fips": "FIPS Code",
    "fipscode": "FIPS Code",
    "fips_code": "FIPS Code",
    "lat": "Latitude",
    "latitude": "Latitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "longitude": "Longitude",
    # Weather
    "temperature": "Temperature (°F)",
    "precipitation": "Precipitation (in)",
    "temp_max": "Max Temperature (°F)",
    "temp_min": "Min Temperature (°F)",
    # Generic
    "year": "Year",
    "date": "Date",
    "name": "Name",
    "category": "Category",
    "count": "Count",
    "rate": "Rate (%)",
    "pct": "Percentage (%)",
    "total": "Total",
}


def _humanize_column(col_name: str) -> str:
    """Convert a raw column name to a human-readable label."""
    if not col_name:
        return col_name
    lower = col_name.lower().strip()
    if lower in _COLUMN_LABELS:
        return _COLUMN_LABELS[lower]

    # Pattern-based matching for known column families with varying suffixes
    import re
    _PATTERNS = [
        (r"^foodinsec[_\d]*$", "Food Insecurity Rate (%)"),
        (r"^vlfoodsec[_\d]*$", "Very Low Food Security (%)"),
        (r"^ch_foodinsec[_\d]*$", "Child Food Insecurity Rate (%)"),
        (r"^povrate\d*$", "Poverty Rate (%)"),
        (r"^medhhinc\d*$", "Median Household Income ($)"),
        (r"^pct_snap\d*$", "SNAP Participation (%)"),
        (r"^pct_laccess[_\w]*$", "Low Food Access (%)"),
        (r"^pct_diabetes[_\w]*$", "Diabetes Rate (%)"),
        (r"^pct_obese[_\w]*$", "Obesity Rate (%)"),
        (r"^grocpth\d*$", "Grocery Stores per 1K Pop"),
        (r"^convspth\d*$", "Convenience Stores per 1K Pop"),
        (r"^ffrpth\d*$", "Fast Food per 1K Pop"),
        (r"^snapspth\d*$", "SNAP Stores per 1K Pop"),
        (r"^recfacpth\d*$", "Recreation Facilities per 1K Pop"),
    ]
    for pattern, label in _PATTERNS:
        if re.match(pattern, lower):
            return label

    # Auto-format: replace underscores, title case, expand abbreviations
    readable = col_name.replace("_", " ").strip()
    # Expand common abbreviations
    abbrevs = {
        "pct": "%", "num": "Number of", "avg": "Average",
        "cnt": "Count", "amt": "Amount", "yr": "Year",
        "mo": "Month", "qty": "Quantity",
    }
    words = readable.split()
    expanded = []
    for w in words:
        wl = w.lower()
        if wl in abbrevs:
            expanded.append(abbrevs[wl])
        elif wl.isdigit() or len(wl) <= 2:
            expanded.append(w.upper())
        else:
            expanded.append(w.capitalize())
    return " ".join(expanded)


def _is_numeric_list(vals: list) -> bool:
    """Check if a list contains mostly numeric values."""
    if not vals:
        return False
    numeric = sum(1 for v in vals[:10] if isinstance(v, (int, float)) and not isinstance(v, bool))
    return numeric / min(len(vals), 10) > 0.6


def _is_string_list(vals: list) -> bool:
    """Check if a list contains mostly string values."""
    if not vals:
        return False
    strings = sum(1 for v in vals[:10] if isinstance(v, str) and not v.replace(".", "").replace("-", "").isdigit())
    return strings / min(len(vals), 10) > 0.6


# ---------------------------------------------------------------------------
# Validation and fix functions
# ---------------------------------------------------------------------------

def validate_and_fix_chart(chart_data: dict) -> dict:
    """Validate a chart spec and fix common issues. Returns the fixed chart."""
    if not isinstance(chart_data, dict) or "plotly_spec" not in chart_data:
        return chart_data

    spec = chart_data["plotly_spec"]
    if not isinstance(spec, dict):
        return chart_data

    traces = spec.get("data", [])
    layout = spec.get("layout", {})
    fixes_applied = []

    # --- Fix 1: Remove empty/all-null traces ---
    valid_traces = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        ttype = trace.get("type", "scatter")

        # Check for empty data
        if ttype in ("bar", "scatter", "histogram"):
            x = trace.get("x", [])
            y = trace.get("y", [])
            if not x and not y:
                fixes_applied.append(f"Removed empty {ttype} trace")
                continue
            # Remove null values
            if x and y and len(x) == len(y):
                pairs = [(xi, yi) for xi, yi in zip(x, y)
                         if xi is not None and yi is not None]
                if pairs:
                    trace["x"], trace["y"] = zip(*pairs)
                    trace["x"] = list(trace["x"])
                    trace["y"] = list(trace["y"])
                elif x:
                    fixes_applied.append(f"Removed all-null {ttype} trace")
                    continue

        elif ttype == "heatmap":
            z = trace.get("z", [])
            if not z or all(not row for row in z):
                fixes_applied.append("Removed empty heatmap trace")
                continue

        elif ttype in ("scattermapbox",):
            lat = trace.get("lat", [])
            lon = trace.get("lon", [])
            if not lat or not lon:
                fixes_applied.append("Removed empty map trace (no lat/lon)")
                continue

        elif ttype == "choroplethmapbox":
            locations = trace.get("locations", [])
            z = trace.get("z", [])
            if not locations or not z:
                fixes_applied.append("Removed empty choropleth trace")
                continue

        elif ttype == "pie":
            labels = trace.get("labels", [])
            values = trace.get("values", [])
            if not labels or not values:
                fixes_applied.append("Removed empty pie trace")
                continue

        valid_traces.append(trace)

    if not valid_traces:
        chart_data["error"] = "All chart traces were empty after validation"
        chart_data["_validation_fixes"] = fixes_applied
        return chart_data

    spec["data"] = valid_traces

    # --- Fix 2: Detect and fix swapped axes on bar charts ---
    for trace in valid_traces:
        ttype = trace.get("type", "scatter")
        orientation = trace.get("orientation", "v")

        if ttype == "bar" and orientation != "h":
            x = trace.get("x", [])
            y = trace.get("y", [])
            if _is_numeric_list(x) and _is_string_list(y):
                # x has numbers, y has strings — they're swapped
                trace["x"], trace["y"] = y, x
                # Also swap axis labels
                xt = layout.get("xaxis", {}).get("title", "")
                yt = layout.get("yaxis", {}).get("title", "")
                if xt and yt:
                    layout.setdefault("xaxis", {})["title"] = yt
                    layout.setdefault("yaxis", {})["title"] = xt
                fixes_applied.append("Swapped bar chart x/y axes (labels were on value axis)")

    # --- Fix 3: Humanize axis labels ---
    for axis_key in ("xaxis", "yaxis"):
        axis = layout.get(axis_key, {})
        title = axis.get("title", "")
        if isinstance(title, dict):
            title_text = title.get("text", "")
        else:
            title_text = title

        if title_text:
            humanized = _humanize_column(title_text)
            if humanized != title_text:
                if isinstance(title, dict):
                    axis["title"]["text"] = humanized
                else:
                    axis["title"] = humanized
                layout[axis_key] = axis
                fixes_applied.append(f"Humanized {axis_key} label: '{title_text}' → '{humanized}'")

    # --- Fix 4: Humanize chart title if it contains raw column names ---
    chart_title = layout.get("title", {})
    if isinstance(chart_title, dict):
        title_text = chart_title.get("text", "")
    else:
        title_text = chart_title

    if title_text:
        # Check for raw column names in title (contain underscores + digits)
        words = title_text.split()
        fixed_words = []
        changed = False
        for w in words:
            clean = w.strip(".,;:()[]")
            if "_" in clean and any(c.isdigit() for c in clean):
                humanized = _humanize_column(clean)
                fixed_words.append(w.replace(clean, humanized))
                changed = True
            else:
                fixed_words.append(w)
        if changed:
            new_title = " ".join(fixed_words)
            if isinstance(chart_title, dict):
                layout["title"]["text"] = new_title
            else:
                layout["title"] = {"text": new_title}
            fixes_applied.append(f"Humanized chart title")

    # --- Fix 5: Humanize colorbar labels ---
    for trace in valid_traces:
        marker = trace.get("marker", {})
        colorbar = marker.get("colorbar", {})
        cb_title = colorbar.get("title", "")
        if isinstance(cb_title, str) and cb_title:
            humanized = _humanize_column(cb_title)
            if humanized != cb_title:
                colorbar["title"] = humanized
                marker["colorbar"] = colorbar
                trace["marker"] = marker
                fixes_applied.append(f"Humanized colorbar: '{cb_title}' → '{humanized}'")

    # --- Fix 6: Validate choropleth FIPS codes ---
    for trace in valid_traces:
        if trace.get("type") == "choroplethmapbox":
            locations = trace.get("locations", [])
            z = trace.get("z", [])
            if locations and z:
                # Ensure FIPS are 5-digit zero-padded strings
                fixed_locs = []
                for loc in locations:
                    s = str(loc).strip()
                    # Remove any decimal (e.g., "29510.0" → "29510")
                    if "." in s:
                        s = s.split(".")[0]
                    fixed_locs.append(s.zfill(5))
                if fixed_locs != locations:
                    trace["locations"] = fixed_locs
                    fixes_applied.append("Fixed FIPS code formatting (zero-padded to 5 digits)")

                # Ensure z values are numeric
                fixed_z = []
                for v in z:
                    if v is None:
                        fixed_z.append(0)
                    elif isinstance(v, (int, float)):
                        fixed_z.append(v)
                    else:
                        try:
                            fixed_z.append(float(v))
                        except (ValueError, TypeError):
                            fixed_z.append(0)
                if fixed_z != z:
                    trace["z"] = fixed_z
                    fixes_applied.append("Converted choropleth z-values to numeric")

    # --- Fix 7: Validate scatter map coordinates ---
    for trace in valid_traces:
        if trace.get("type") == "scattermapbox":
            lat = trace.get("lat", [])
            lon = trace.get("lon", [])
            if lat and lon:
                # Ensure numeric
                try:
                    lat_f = [float(v) for v in lat if v is not None]
                    lon_f = [float(v) for v in lon if v is not None]
                except (ValueError, TypeError):
                    continue

                # Check if lat/lon are swapped (US heuristic)
                if lat_f and lon_f:
                    avg_lat = sum(lat_f) / len(lat_f)
                    avg_lon = sum(lon_f) / len(lon_f)
                    if abs(avg_lat) > 90 and abs(avg_lon) <= 90:
                        trace["lat"], trace["lon"] = lon, lat
                        fixes_applied.append("Swapped lat/lon (were reversed)")
                        # Also fix center
                        mapbox = layout.get("mapbox", {})
                        center = mapbox.get("center", {})
                        if center:
                            old_lat = center.get("lat")
                            old_lon = center.get("lon")
                            if old_lat and old_lon:
                                center["lat"] = old_lon
                                center["lon"] = old_lat
                                mapbox["center"] = center
                                layout["mapbox"] = mapbox

    # --- Fix 8: Ensure chart has a title ---
    if not title_text:
        chart_type = chart_data.get("chart_type", "chart")
        layout["title"] = {"text": chart_data.get("title", f"AgriFlow {chart_type.title()} Chart")}
        fixes_applied.append("Added missing chart title")

    # --- Fix 9: Ensure dark theme is applied ---
    if not layout.get("paper_bgcolor"):
        layout["paper_bgcolor"] = "rgba(11,15,26,0.95)"
    if not layout.get("plot_bgcolor"):
        layout["plot_bgcolor"] = "rgba(11,15,26,0.7)"
    if not layout.get("font"):
        layout["font"] = {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif"}

    spec["layout"] = layout
    chart_data["plotly_spec"] = spec

    if fixes_applied:
        chart_data["_validation_fixes"] = fixes_applied

    return chart_data


def validate_chart_messages(messages: list) -> tuple[list, list[str]]:
    """Validate all chart ToolMessages in a message list.

    Returns (fixed_messages, fix_log) where fix_log describes all fixes applied.
    """
    fix_log = []
    fixed_messages = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            fixed_messages.append(msg)
            continue

        if not hasattr(msg, "name") or msg.name not in _CHART_TOOLS:
            fixed_messages.append(msg)
            continue

        # Parse the chart data
        try:
            content = msg.content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
        except (json.JSONDecodeError, TypeError):
            fixed_messages.append(msg)
            continue

        if not isinstance(data, dict) or "plotly_spec" not in data:
            fixed_messages.append(msg)
            continue

        # Validate and fix
        fixed_data = validate_and_fix_chart(data)
        fixes = fixed_data.pop("_validation_fixes", [])

        if fixes:
            chart_id = fixed_data.get("chart_id", "unknown")
            for fix in fixes:
                fix_log.append(f"Chart {chart_id}: {fix}")

            # Create new ToolMessage with fixed content
            fixed_msg = ToolMessage(
                content=json.dumps(fixed_data),
                name=msg.name,
                tool_call_id=getattr(msg, "tool_call_id", "validated"),
            )
            fixed_messages.append(fixed_msg)
        else:
            fixed_messages.append(msg)

    return fixed_messages, fix_log
