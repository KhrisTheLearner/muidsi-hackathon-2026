"""Chart and map generation tools.

Returns Plotly JSON specs that the React frontend renders with react-plotly.js.
All tool logic is here (single source of truth). MCP servers import from this module.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.tools import tool

_DARK_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(11,15,26,0.95)",
    "plot_bgcolor": "rgba(11,15,26,0.7)",
    "font": {"color": "#e2e8f0", "family": "Inter, system-ui, sans-serif"},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
}


def _humanize(col_name: str) -> str:
    """Convert raw column name to human-readable axis label."""
    if not col_name:
        return col_name
    # Import the central mapping
    from src.agent.nodes.chart_validator import _humanize_column
    return _humanize_column(col_name)


def _parse(data_json: str) -> list[dict]:
    if isinstance(data_json, list):
        return data_json
    try:
        return json.loads(data_json)
    except (json.JSONDecodeError, TypeError):
        return []


def _check_cols(rows: list[dict], *cols: str) -> str | None:
    """Return error message if any required column is missing from data."""
    if not rows:
        return "No data rows provided. Query data before creating charts."
    sample = rows[0]
    missing = [c for c in cols if c and c not in sample]
    if missing:
        available = list(sample.keys())
        return f"Column(s) {missing} not found in data. Available: {available}"
    return None


def _is_numeric(val: Any) -> bool:
    """Check if a value is numeric (int, float, or numeric string)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _check_numeric(rows: list[dict], *cols: str) -> str | None:
    """Validate that columns contain numeric data. Returns error or None.

    Samples up to 5 rows and requires at least 60% numeric values to pass.
    Skips empty/None values in the count.
    """
    if not rows:
        return None
    sample = rows[:5]
    bad_cols = []
    for col in cols:
        if not col:
            continue
        vals = [r.get(col) for r in sample if r.get(col) is not None]
        if not vals:
            continue
        numeric_count = sum(1 for v in vals if _is_numeric(v))
        if numeric_count / len(vals) < 0.6:
            examples = [repr(v) for v in vals[:3]]
            bad_cols.append(f"'{col}' (sample values: {', '.join(examples)})")
    if bad_cols:
        return (f"Expected numeric data in column(s) {', '.join(bad_cols)} "
                f"but found non-numeric values. Check that the correct "
                f"column is assigned to the value/y axis.")
    return None


def _check_fips(rows: list[dict], fips_col: str) -> str | None:
    """Validate that FIPS column contains plausible 5-digit FIPS codes."""
    if not rows or not fips_col:
        return None
    sample = rows[:10]
    bad = []
    for r in sample:
        raw = r.get(fips_col)
        if raw is None:
            continue
        s = str(raw).strip()
        # FIPS codes are 1-5 digit integers (leading zeros may be stripped)
        if not s.replace(".", "").isdigit() or len(s.replace(".", "").lstrip("0") or "0") > 5:
            bad.append(repr(raw))
    if bad and len(bad) > len(sample) * 0.5:
        return (f"Column '{fips_col}' does not appear to contain valid FIPS codes. "
                f"Sample values: {', '.join(bad[:3])}. FIPS codes should be "
                f"numeric county identifiers (e.g., '29510' for St. Louis City).")
    return None


def _check_latlon(rows: list[dict], lat_col: str, lon_col: str) -> str | None:
    """Validate that lat/lon columns contain plausible geographic coordinates."""
    if not rows or not lat_col or not lon_col:
        return None
    issues = []
    sample = rows[:10]
    for label, col, lo, hi in [("Latitude", lat_col, -90, 90),
                                ("Longitude", lon_col, -180, 180)]:
        vals = []
        for r in sample:
            v = r.get(col)
            if v is not None and _is_numeric(v):
                vals.append(float(v))
        if not vals:
            issues.append(f"'{col}' has no numeric values for {label}")
            continue
        out_of_range = [v for v in vals if v < lo or v > hi]
        if out_of_range:
            issues.append(
                f"'{col}' has {label} values out of range [{lo}, {hi}]: "
                f"{out_of_range[:3]}")
    # Check if lat/lon might be swapped (US-centric heuristic)
    if not issues:
        lat_vals = [float(r.get(lat_col)) for r in sample
                    if r.get(lat_col) is not None and _is_numeric(r.get(lat_col))]
        lon_vals = [float(r.get(lon_col)) for r in sample
                    if r.get(lon_col) is not None and _is_numeric(r.get(lon_col))]
        if lat_vals and lon_vals:
            avg_lat = sum(lat_vals) / len(lat_vals)
            avg_lon = sum(lon_vals) / len(lon_vals)
            # If "lat" values look like longitude and vice versa
            if abs(avg_lat) > 90 and abs(avg_lon) <= 90:
                issues.append(
                    f"lat_col '{lat_col}' (avg={avg_lat:.1f}) and lon_col "
                    f"'{lon_col}' (avg={avg_lon:.1f}) appear to be swapped.")
    if issues:
        return "Geographic validation: " + "; ".join(issues)
    return None


def _detect_column_swap(rows: list[dict], label_col: str, value_col: str) -> tuple[str, str]:
    """Detect if label/value columns are swapped and return corrected pair.

    If the label column is all-numeric and the value column is all-strings,
    they are likely swapped. Returns (corrected_label, corrected_value).
    """
    if not rows or not label_col or not value_col:
        return label_col, value_col
    if label_col not in rows[0] or value_col not in rows[0]:
        return label_col, value_col
    sample = rows[:5]
    label_vals = [r.get(label_col) for r in sample if r.get(label_col) is not None]
    value_vals = [r.get(value_col) for r in sample if r.get(value_col) is not None]
    if not label_vals or not value_vals:
        return label_col, value_col
    label_numeric = sum(1 for v in label_vals if _is_numeric(v)) / len(label_vals)
    value_numeric = sum(1 for v in value_vals if _is_numeric(v)) / len(value_vals)
    # If labels are numeric and values are strings, they're swapped
    if label_numeric > 0.8 and value_numeric < 0.2:
        return value_col, label_col
    return label_col, value_col


def _cid() -> str:
    return f"chart_{uuid.uuid4().hex[:8]}"


@tool
def create_bar_chart(
    title: str,
    data_json: str,
    x_col: str,
    y_col: str,
    color_col: str = "",
    horizontal: bool = False,
) -> dict[str, Any]:
    """Create a Plotly bar chart for the frontend to render.

    Use after querying data to visualize rankings, comparisons, or distributions.

    Args:
        title: Chart title.
        data_json: JSON array of row objects, e.g. '[{"county":"Wayne","rate":23.1}]'.
        x_col: Column for categories.
        y_col: Column for values.
        color_col: Optional column for color-coding bars (continuous scale).
        horizontal: If true, horizontal bars (good for ranked county lists).
    """
    rows = _parse(data_json)
    err = _check_cols(rows, x_col, y_col)
    if err:
        return {"error": err}

    # Auto-detect and fix swapped label/value columns
    x_col, y_col = _detect_column_swap(rows, x_col, y_col)

    x_vals = [r.get(x_col) for r in rows]
    y_vals = [r.get(y_col) for r in rows]

    # Validate: y should be numeric, x should be labels
    num_err = _check_numeric(rows, y_col)
    if num_err:
        return {"error": num_err}

    trace: dict[str, Any] = {"type": "bar", "name": _humanize(y_col)}
    if horizontal:
        trace.update({"x": y_vals, "y": x_vals, "orientation": "h"})
    else:
        trace.update({"x": x_vals, "y": y_vals})

    if color_col:
        trace["marker"] = {"color": [r.get(color_col) for r in rows],
                           "colorscale": "RdYlGn_r", "showscale": True,
                           "colorbar": {"title": _humanize(color_col)}}
    else:
        trace["marker"] = {"color": "#3b82f6"}

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": _humanize(y_col) if horizontal else _humanize(x_col)},
              "yaxis": {"title": _humanize(x_col) if horizontal else _humanize(y_col)}}

    return {"chart_id": _cid(), "chart_type": "bar", "title": title,
            "plotly_spec": {"data": [trace], "layout": layout}}


@tool
def create_line_chart(
    title: str,
    data_json: str,
    x_col: str,
    y_cols: str,
) -> dict[str, Any]:
    """Create a Plotly line chart for trends and time series.

    Args:
        title: Chart title.
        data_json: JSON array of row objects.
        x_col: Column for x-axis (dates, years, etc.).
        y_cols: Comma-separated column names for y-axis lines.
    """
    rows = _parse(data_json)
    cols = [c.strip() for c in y_cols.split(",")]
    err = _check_cols(rows, x_col, *cols)
    if err:
        return {"error": err}
    num_err = _check_numeric(rows, *cols)
    if num_err:
        return {"error": num_err}
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]

    traces = []
    for i, col in enumerate(cols):
        traces.append({
            "type": "scatter", "mode": "lines+markers", "name": _humanize(col),
            "x": [r.get(x_col) for r in rows],
            "y": [r.get(col) for r in rows],
            "line": {"color": palette[i % len(palette)], "width": 2},
            "marker": {"size": 6},
        })

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": _humanize(x_col)},
              "yaxis": {"title": _humanize(cols[0]) if len(cols) == 1 else ""},
              "legend": {"orientation": "h", "y": -0.15}}

    return {"chart_id": _cid(), "chart_type": "line", "title": title,
            "plotly_spec": {"data": traces, "layout": layout}}


@tool
def create_scatter_map(
    title: str,
    data_json: str,
    lat_col: str,
    lon_col: str,
    size_col: str = "",
    color_col: str = "",
    text_col: str = "",
    zoom: int = 6,
) -> dict[str, Any]:
    """Create a geographic scatter map (Plotly scattermapbox).

    Use for risk assessment maps, food desert visualization, distribution
    point mapping, and disaster response overlays. Renders as an interactive
    dark-themed map.

    Args:
        title: Map title.
        data_json: JSON array with lat/lon and data columns.
        lat_col: Column with latitude values.
        lon_col: Column with longitude values.
        size_col: Optional column for marker size (proportional).
        color_col: Optional column for marker color (continuous scale, red=high risk).
        text_col: Optional column for hover text labels.
        zoom: Map zoom level (default 6 for state-level view).
    """
    rows = _parse(data_json)
    err = _check_cols(rows, lat_col, lon_col)
    if err:
        return {"error": err}
    geo_err = _check_latlon(rows, lat_col, lon_col)
    if geo_err:
        return {"error": geo_err}
    lats = [r.get(lat_col) for r in rows]
    lons = [r.get(lon_col) for r in rows]

    trace: dict[str, Any] = {
        "type": "scattermapbox", "mode": "markers+text",
        "lat": lats, "lon": lons, "name": title,
    }

    if text_col:
        trace["text"] = [r.get(text_col, "") for r in rows]
        trace["textposition"] = "top center"
        trace["textfont"] = {"size": 10, "color": "#e2e8f0"}

    marker: dict[str, Any] = {"opacity": 0.85}
    if size_col:
        sizes = [r.get(size_col, 10) for r in rows]
        max_s = max(sizes) if sizes else 1
        marker["size"] = [max(8, (s / max_s) * 35) for s in sizes]
    else:
        marker["size"] = 12

    if color_col:
        marker["color"] = [r.get(color_col) for r in rows]
        marker["colorscale"] = "RdYlGn_r"
        marker["showscale"] = True
        marker["colorbar"] = {"title": _humanize(color_col)}
    else:
        marker["color"] = "#ef4444"

    trace["marker"] = marker

    center_lat = sum(float(v) for v in lats if v is not None) / max(len(lats), 1) if lats else 37.5
    center_lon = sum(float(v) for v in lons if v is not None) / max(len(lons), 1) if lons else -90.5

    layout = {
        **_DARK_LAYOUT,
        "title": {"text": title},
        "mapbox": {
            "style": "carto-darkmatter",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": zoom,
        },
        "height": 500,
    }

    return {"chart_id": _cid(), "chart_type": "scatter_map", "title": title,
            "plotly_spec": {"data": [trace], "layout": layout}}


@tool
def create_risk_heatmap(
    title: str,
    data_json: str,
    x_col: str,
    y_col: str,
    z_col: str,
) -> dict[str, Any]:
    """Create a risk assessment heatmap.

    Good for county-vs-factor risk matrices, temporal risk patterns,
    and scenario comparison grids.

    Args:
        title: Chart title.
        data_json: JSON array of row objects.
        x_col: Column for x-axis categories.
        y_col: Column for y-axis categories.
        z_col: Column for cell values (color intensity).
    """
    rows = _parse(data_json)
    err = _check_cols(rows, x_col, y_col, z_col)
    if err:
        return {"error": err}
    num_err = _check_numeric(rows, z_col)
    if num_err:
        return {"error": num_err}
    x_labels = sorted(set(r.get(x_col) for r in rows))
    y_labels = sorted(set(r.get(y_col) for r in rows))
    lookup = {(r.get(x_col), r.get(y_col)): r.get(z_col, 0) for r in rows}
    z_matrix = [[lookup.get((x, y), 0) for x in x_labels] for y in y_labels]

    trace = {
        "type": "heatmap",
        "x": x_labels, "y": y_labels, "z": z_matrix,
        "colorscale": "RdYlGn_r",
        "colorbar": {"title": _humanize(z_col)},
        "hoverongaps": False,
    }

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": _humanize(x_col)}, "yaxis": {"title": _humanize(y_col)}}

    return {"chart_id": _cid(), "chart_type": "heatmap", "title": title,
            "plotly_spec": {"data": [trace], "layout": layout}}


def _load_counties_geojson() -> dict | None:
    """Load and cache the US counties GeoJSON for choropleth maps."""
    if not hasattr(_load_counties_geojson, "_cache"):
        import urllib.request
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                import json as _json
                _load_counties_geojson._cache = _json.loads(resp.read())
        except Exception:
            _load_counties_geojson._cache = None
    return _load_counties_geojson._cache


def _filter_geojson(geojson: dict, fips_set: set) -> dict:
    """Filter GeoJSON features to only matching FIPS codes (reduces payload)."""
    # Get the state prefixes from fips_set to include nearby counties for context
    state_prefixes = {f[:2] for f in fips_set}
    filtered = [f for f in geojson.get("features", [])
                if str(f.get("id", ""))[:2] in state_prefixes]
    return {"type": "FeatureCollection", "features": filtered}


@tool
def create_choropleth_map(
    title: str,
    data_json: str,
    fips_col: str,
    value_col: str,
    text_col: str = "",
    colorscale: str = "RdYlGn_r",
) -> dict[str, Any]:
    """Create a geographic choropleth map colored by county FIPS codes.

    Overlays data values on a real county-level map. Best for geographic
    heatmaps showing risk, poverty, food insecurity, or other metrics
    across counties. Requires FIPS codes in the data.

    Args:
        title: Map title.
        data_json: JSON array of row objects with FIPS codes and values.
        fips_col: Column containing 5-digit county FIPS codes.
        value_col: Column with numeric values to color counties by.
        text_col: Optional column for hover text labels (e.g. county name).
        colorscale: Plotly colorscale name (default "RdYlGn_r" = red=high risk).
    """
    rows = _parse(data_json)
    err = _check_cols(rows, fips_col, value_col)
    if err:
        return {"error": err}
    fips_err = _check_fips(rows, fips_col)
    if fips_err:
        return {"error": fips_err}
    num_err = _check_numeric(rows, value_col)
    if num_err:
        return {"error": num_err}

    fips_vals = []
    for r in rows:
        f = str(r.get(fips_col, "")).zfill(5)
        fips_vals.append(f)

    z_vals = [r.get(value_col, 0) for r in rows]
    hover = [r.get(text_col, f) for r, f in zip(rows, fips_vals)] if text_col else fips_vals

    # Load and filter GeoJSON server-side so browser doesn't need to fetch 30MB
    full_geojson = _load_counties_geojson()
    if full_geojson:
        geojson = _filter_geojson(full_geojson, set(fips_vals))
    else:
        # Fallback: let browser fetch it (may be slow)
        geojson = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

    value_label = _humanize(value_col)
    trace = {
        "type": "choroplethmapbox",
        "geojson": geojson,
        "locations": fips_vals,
        "z": z_vals,
        "text": hover,
        "colorscale": colorscale,
        "colorbar": {"title": value_label, "thickness": 15},
        "marker": {"opacity": 0.7, "line": {"width": 0.5, "color": "rgba(255,255,255,0.3)"}},
        "hovertemplate": "%{text}<br>" + value_label + ": %{z:.2f}<extra></extra>",
        "featureidkey": "id",
    }

    # Center on the data (use Missouri as default)
    layout = {
        **_DARK_LAYOUT,
        "title": {"text": title},
        "mapbox": {
            "style": "carto-darkmatter",
            "center": {"lat": 38.5, "lon": -92.5},
            "zoom": 5.5,
        },
        "height": 500,
    }

    return {"chart_id": _cid(), "chart_type": "choropleth", "title": title,
            "plotly_spec": {"data": [trace], "layout": layout}}


@tool
def create_chart(
    chart_type: str,
    title: str,
    data_json: str,
    x_col: str = "",
    y_col: str = "",
    y_cols: str = "",
    color_col: str = "",
    size_col: str = "",
    text_col: str = "",
    names_col: str = "",
    values_col: str = "",
    z_col: str = "",
    lat_col: str = "",
    lon_col: str = "",
    fips_col: str = "",
    nbins: int = 30,
    horizontal: bool = False,
    colorscale: str = "RdYlGn_r",
) -> dict[str, Any]:
    """Create any Plotly chart type dynamically.

    This is the universal chart tool — use it when you need a chart type not
    covered by the specialized tools (scatter, pie, histogram, box, violin,
    area, funnel, treemap, sunburst, waterfall, indicator/gauge, bubble).
    Also supports all standard types (bar, line, heatmap, scatter_map, choropleth).

    Args:
        chart_type: One of: bar, line, scatter, pie, histogram, box, violin,
                    area, heatmap, scatter_map, choropleth, funnel, treemap,
                    sunburst, waterfall, indicator, bubble,
                    roc_curve, actual_vs_predicted, feature_importance,
                    correlation_matrix.
        title: Chart title.
        data_json: JSON array of row objects.
        x_col: Column for x-axis (bar, line, scatter, histogram, box, area, funnel,
               waterfall, roc_curve=FPR col, actual_vs_predicted=actual col).
        y_col: Column for y-axis (bar, scatter, box, violin, area, funnel, waterfall,
               roc_curve=TPR col, actual_vs_predicted=predicted col).
        y_cols: Comma-separated columns for multi-line charts.
        color_col: Column for color encoding (scatter, bubble, bar, roc_curve=model name).
        size_col: Column for marker size (bubble, scatter_map).
        text_col: Column for labels/hover text.
        names_col: Column for category names (pie, treemap, sunburst, feature_importance=feature name).
        values_col: Column for numeric values (pie, treemap, sunburst, indicator, feature_importance=score).
        z_col: Column for z-axis / cell values (heatmap).
        lat_col: Latitude column (scatter_map).
        lon_col: Longitude column (scatter_map).
        fips_col: FIPS code column (choropleth).
        nbins: Number of bins for histograms (default 30).
        horizontal: Horizontal orientation for bar/funnel.
        colorscale: Plotly colorscale name.
    """
    rows = _parse(data_json)
    if not rows:
        return {"error": "No data rows provided. Query data before creating charts."}

    ct = chart_type.lower().strip()
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
               "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]

    # --- Delegate to specialized tools for standard types ---
    if ct == "bar":
        return create_bar_chart.invoke({
            "title": title, "data_json": data_json,
            "x_col": x_col, "y_col": y_col,
            "color_col": color_col, "horizontal": horizontal,
        })
    if ct == "line":
        return create_line_chart.invoke({
            "title": title, "data_json": data_json,
            "x_col": x_col, "y_cols": y_cols or y_col,
        })
    if ct == "heatmap":
        return create_risk_heatmap.invoke({
            "title": title, "data_json": data_json,
            "x_col": x_col, "y_col": y_col, "z_col": z_col,
        })
    if ct == "scatter_map":
        return create_scatter_map.invoke({
            "title": title, "data_json": data_json,
            "lat_col": lat_col, "lon_col": lon_col,
            "size_col": size_col, "color_col": color_col, "text_col": text_col,
        })
    if ct == "choropleth":
        return create_choropleth_map.invoke({
            "title": title, "data_json": data_json,
            "fips_col": fips_col, "value_col": y_col or values_col,
            "text_col": text_col, "colorscale": colorscale,
        })

    # --- New chart types ---
    if ct == "scatter" or ct == "bubble":
        # Auto-correct swapped axes: scatter/bubble expect numeric on y
        x_col, y_col = _detect_column_swap(rows, x_col, y_col)
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, y_col)
        if num_err:
            return {"error": num_err}
        marker: dict[str, Any] = {"opacity": 0.7}
        if ct == "bubble" and size_col:
            sizes = [r.get(size_col, 10) for r in rows]
            max_s = max(sizes) if sizes else 1
            marker["size"] = [max(6, (s / max_s) * 40) for s in sizes]
        else:
            marker["size"] = 8
        if color_col:
            marker["color"] = [r.get(color_col) for r in rows]
            marker["colorscale"] = colorscale
            marker["showscale"] = True
            marker["colorbar"] = {"title": _humanize(color_col)}
        else:
            marker["color"] = palette[0]
        trace = {
            "type": "scatter", "mode": "markers",
            "x": [r.get(x_col) for r in rows],
            "y": [r.get(y_col) for r in rows],
            "marker": marker,
            "text": [r.get(text_col, "") for r in rows] if text_col else None,
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": _humanize(x_col)}, "yaxis": {"title": _humanize(y_col)}}
        return {"chart_id": _cid(), "chart_type": ct, "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "pie":
        n_col = names_col or x_col
        v_col = values_col or y_col
        # Auto-correct: names should be strings, values should be numeric
        n_col, v_col = _detect_column_swap(rows, n_col, v_col)
        err = _check_cols(rows, n_col, v_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, v_col)
        if num_err:
            return {"error": num_err}
        trace = {
            "type": "pie",
            "labels": [r.get(n_col) for r in rows],
            "values": [r.get(v_col) for r in rows],
            "marker": {"colors": palette[:len(rows)]},
            "textinfo": "label+percent",
            "hole": 0.3,
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title}}
        return {"chart_id": _cid(), "chart_type": "pie", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "histogram":
        col = x_col or y_col
        err = _check_cols(rows, col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, col)
        if num_err:
            return {"error": num_err}
        trace = {
            "type": "histogram",
            "x": [r.get(col) for r in rows],
            "nbinsx": nbins,
            "marker": {"color": palette[0], "line": {"color": "#1e293b", "width": 1}},
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": _humanize(col)}, "yaxis": {"title": "Count"}}
        return {"chart_id": _cid(), "chart_type": "histogram", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "box":
        col = y_col or x_col
        err = _check_cols(rows, col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, col)
        if num_err:
            return {"error": num_err}
        traces = []
        if x_col and x_col != col:
            # Grouped box plot
            groups = sorted(set(r.get(x_col) for r in rows))
            for i, g in enumerate(groups):
                traces.append({
                    "type": "box", "name": str(g),
                    "y": [r.get(col) for r in rows if r.get(x_col) == g],
                    "marker": {"color": palette[i % len(palette)]},
                })
        else:
            traces.append({
                "type": "box", "name": _humanize(col),
                "y": [r.get(col) for r in rows],
                "marker": {"color": palette[0]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "yaxis": {"title": _humanize(col)}}
        return {"chart_id": _cid(), "chart_type": "box", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "violin":
        col = y_col or x_col
        err = _check_cols(rows, col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, col)
        if num_err:
            return {"error": num_err}
        traces = []
        if x_col and x_col != col:
            groups = sorted(set(r.get(x_col) for r in rows))
            for i, g in enumerate(groups):
                traces.append({
                    "type": "violin", "name": str(g),
                    "y": [r.get(col) for r in rows if r.get(x_col) == g],
                    "box": {"visible": True}, "meanline": {"visible": True},
                    "marker": {"color": palette[i % len(palette)]},
                })
        else:
            traces.append({
                "type": "violin", "name": _humanize(col),
                "y": [r.get(col) for r in rows],
                "box": {"visible": True}, "meanline": {"visible": True},
                "marker": {"color": palette[0]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "yaxis": {"title": _humanize(col)}}
        return {"chart_id": _cid(), "chart_type": "violin", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "area":
        err = _check_cols(rows, x_col)
        if err:
            return {"error": err}
        cols = [c.strip() for c in (y_cols or y_col).split(",")]
        num_err = _check_numeric(rows, *cols)
        if num_err:
            return {"error": num_err}
        traces = []
        for i, col in enumerate(cols):
            traces.append({
                "type": "scatter", "mode": "lines", "name": _humanize(col),
                "fill": "tozeroy" if i == 0 else "tonexty",
                "x": [r.get(x_col) for r in rows],
                "y": [r.get(col) for r in rows],
                "line": {"color": palette[i % len(palette)]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": _humanize(x_col)},
                  "yaxis": {"title": _humanize(cols[0]) if len(cols) == 1 else ""}}
        return {"chart_id": _cid(), "chart_type": "area", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "funnel":
        # Auto-correct: funnel labels go on y, values on x
        x_col, y_col = _detect_column_swap(rows, x_col, y_col)
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, y_col)
        if num_err:
            return {"error": num_err}
        trace = {
            "type": "funnel",
            "y": [r.get(x_col) for r in rows],
            "x": [r.get(y_col) for r in rows],
            "marker": {"color": palette[:len(rows)]},
            "textinfo": "value+percent initial",
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title}}
        return {"chart_id": _cid(), "chart_type": "funnel", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "treemap" or ct == "sunburst":
        n_col = names_col or x_col
        v_col = values_col or y_col
        # Auto-correct: labels should be strings, values numeric
        n_col, v_col = _detect_column_swap(rows, n_col, v_col)
        err = _check_cols(rows, n_col, v_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, v_col)
        if num_err:
            return {"error": num_err}
        trace = {
            "type": ct,
            "labels": [r.get(n_col) for r in rows],
            "parents": [""] * len(rows),
            "values": [r.get(v_col) for r in rows],
            "marker": {"colors": palette[:len(rows)]},
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title}}
        return {"chart_id": _cid(), "chart_type": ct, "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "waterfall":
        x_col, y_col = _detect_column_swap(rows, x_col, y_col)
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, y_col)
        if num_err:
            return {"error": num_err}
        trace = {
            "type": "waterfall",
            "x": [r.get(x_col) for r in rows],
            "y": [r.get(y_col) for r in rows],
            "connector": {"line": {"color": "#94a3b8"}},
            "increasing": {"marker": {"color": "#10b981"}},
            "decreasing": {"marker": {"color": "#ef4444"}},
            "totals": {"marker": {"color": "#3b82f6"}},
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": _humanize(x_col)}, "yaxis": {"title": _humanize(y_col)}}
        return {"chart_id": _cid(), "chart_type": "waterfall", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "indicator" or ct == "gauge":
        v_col = values_col or y_col
        num_err = _check_numeric(rows, v_col) if v_col else None
        if num_err:
            return {"error": num_err}
        val = rows[0].get(v_col, 0) if rows else 0
        if not _is_numeric(val):
            return {"error": f"Indicator value column '{v_col}' must contain a numeric value, got {repr(val)}."}
        val = float(val)
        trace = {
            "type": "indicator", "mode": "gauge+number+delta",
            "value": val,
            "title": {"text": _humanize(text_col or v_col)},
            "gauge": {
                "axis": {"range": [0, max(val * 1.5, 100)]},
                "bar": {"color": palette[0]},
                "steps": [
                    {"range": [0, max(val * 0.5, 33)], "color": "rgba(16,185,129,0.3)"},
                    {"range": [max(val * 0.5, 33), max(val, 66)], "color": "rgba(245,158,11,0.3)"},
                    {"range": [max(val, 66), max(val * 1.5, 100)], "color": "rgba(239,68,68,0.3)"},
                ],
            },
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title}, "height": 350}
        return {"chart_id": _cid(), "chart_type": "indicator", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    # ── ROC Curve ────────────────────────────────────────────────────────────
    if ct == "roc_curve":
        fpr_col = x_col
        tpr_col = y_col
        err = _check_cols(rows, fpr_col, tpr_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, fpr_col, tpr_col)
        if num_err:
            return {"error": num_err}

        traces = []
        if color_col and rows and color_col in rows[0]:
            models = list(dict.fromkeys(r.get(color_col) for r in rows))  # preserve order
            for i, model in enumerate(models):
                mrows = [r for r in rows if r.get(color_col) == model]
                fpr_vals = [r.get(fpr_col) for r in mrows]
                tpr_vals = [r.get(tpr_col) for r in mrows]
                # Approximate AUC via trapezoidal integration
                auc = 0.0
                for k in range(1, len(fpr_vals)):
                    dx = float(fpr_vals[k] or 0) - float(fpr_vals[k - 1] or 0)
                    auc += dx * (float(tpr_vals[k] or 0) + float(tpr_vals[k - 1] or 0)) / 2
                auc = abs(auc)
                traces.append({
                    "type": "scatter", "mode": "lines",
                    "name": f"{model} (AUC={auc:.3f})",
                    "x": fpr_vals, "y": tpr_vals,
                    "line": {"color": palette[i % len(palette)], "width": 2.5},
                })
        else:
            fpr_vals = [r.get(fpr_col) for r in rows]
            tpr_vals = [r.get(tpr_col) for r in rows]
            auc = 0.0
            for k in range(1, len(fpr_vals)):
                dx = float(fpr_vals[k] or 0) - float(fpr_vals[k - 1] or 0)
                auc += dx * (float(tpr_vals[k] or 0) + float(tpr_vals[k - 1] or 0)) / 2
            auc = abs(auc)
            traces.append({
                "type": "scatter", "mode": "lines",
                "name": f"ROC Curve (AUC={auc:.3f})",
                "x": fpr_vals, "y": tpr_vals,
                "line": {"color": palette[0], "width": 2.5},
            })
        # Diagonal reference line
        traces.append({
            "type": "scatter", "mode": "lines",
            "name": "Random Classifier",
            "x": [0, 1], "y": [0, 1],
            "line": {"color": "#94a3b8", "width": 1.5, "dash": "dash"},
            "showlegend": True,
        })
        layout = {
            **_DARK_LAYOUT,
            "title": {"text": title},
            "xaxis": {"title": "False Positive Rate (FPR)", "range": [-0.02, 1.02]},
            "yaxis": {"title": "True Positive Rate (TPR)", "range": [-0.02, 1.02]},
            "legend": {"x": 0.55, "y": 0.08, "bgcolor": "rgba(0,0,0,0.4)"},
        }
        return {"chart_id": _cid(), "chart_type": "roc_curve", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    # ── Actual vs Predicted (Regression Diagnostic) ──────────────────────────
    if ct in ("actual_vs_predicted", "residual"):
        actual_col = x_col
        pred_col = y_col
        err = _check_cols(rows, actual_col, pred_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, actual_col, pred_col)
        if num_err:
            return {"error": num_err}

        actuals = [float(r.get(actual_col) or 0) for r in rows]
        preds = [float(r.get(pred_col) or 0) for r in rows]

        # Compute R² inline
        mean_a = sum(actuals) / len(actuals)
        ss_tot = sum((a - mean_a) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for a, p in zip(actuals, preds))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = (ss_res / len(actuals)) ** 0.5

        traces = [
            {
                "type": "scatter", "mode": "markers",
                "name": f"Predictions (R²={r2:.4f}, RMSE={rmse:.4f})",
                "x": actuals, "y": preds,
                "marker": {"color": palette[0], "size": 7, "opacity": 0.75,
                           "line": {"color": "#1e293b", "width": 0.5}},
                "text": [r.get(text_col, "") for r in rows] if text_col else None,
                "hovertemplate": f"Actual: %{{x:.3f}}<br>Predicted: %{{y:.3f}}<extra></extra>",
            },
            {
                "type": "scatter", "mode": "lines",
                "name": "Ideal (y = x)",
                "x": [min(actuals + preds), max(actuals + preds)],
                "y": [min(actuals + preds), max(actuals + preds)],
                "line": {"color": "#10b981", "width": 1.5, "dash": "dash"},
            },
        ]
        layout = {
            **_DARK_LAYOUT,
            "title": {"text": title + f" — R²={r2:.4f}"},
            "xaxis": {"title": _humanize(actual_col)},
            "yaxis": {"title": _humanize(pred_col)},
        }
        return {"chart_id": _cid(), "chart_type": "actual_vs_predicted", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    # ── Feature Importance ────────────────────────────────────────────────────
    if ct == "feature_importance":
        name_col = names_col or text_col or y_col
        imp_col = values_col or x_col
        if not name_col or not imp_col:
            return {"error": "feature_importance requires names_col (feature name) and values_col (importance score)."}
        # Auto-correct if swapped
        name_col, imp_col = _detect_column_swap(rows, name_col, imp_col)
        err = _check_cols(rows, name_col, imp_col)
        if err:
            return {"error": err}
        num_err = _check_numeric(rows, imp_col)
        if num_err:
            return {"error": num_err}

        sorted_rows = sorted(rows, key=lambda r: float(r.get(imp_col) or 0), reverse=True)
        top_rows = sorted_rows[:20]
        importances = [float(r.get(imp_col) or 0) for r in top_rows]
        names = [str(r.get(name_col)) for r in top_rows]
        # Humanize feature names
        names = [_humanize(n) for n in names]

        trace = {
            "type": "bar",
            "name": "Importance",
            "x": importances,
            "y": names,
            "orientation": "h",
            "marker": {
                "color": importances,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Score", "thickness": 12},
            },
            "text": [f"{v:.4f}" for v in importances],
            "textposition": "outside",
        }
        dyn_height = max(320, len(top_rows) * 24 + 100)
        layout = {
            **_DARK_LAYOUT,
            "title": {"text": title},
            "xaxis": {"title": "Importance Score"},
            "yaxis": {"autorange": "reversed"},
            "margin": {"l": 180, "r": 80, "t": 50, "b": 50},
            "height": dyn_height,
        }
        return {"chart_id": _cid(), "chart_type": "feature_importance", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    # ── Correlation Matrix ────────────────────────────────────────────────────
    if ct == "correlation_matrix":
        try:
            import pandas as _pd  # local import to avoid top-level dependency issues
        except ImportError:
            return {"error": "pandas is required for correlation_matrix charts."}

        df_corr = _pd.DataFrame(rows)
        # Keep only numeric columns; optionally filter by y_cols list
        num_df = df_corr.apply(_pd.to_numeric, errors="coerce")
        if y_cols:
            wanted = [c.strip() for c in y_cols.split(",") if c.strip() in num_df.columns]
            if wanted:
                num_df = num_df[wanted]
        num_df = num_df.dropna(axis=1, how="all")
        num_df = num_df[[c for c in num_df.columns if num_df[c].std() > 0]]  # drop zero-variance
        if num_df.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns with variance for a correlation matrix."}

        corr = num_df.corr()
        labels = [_humanize(c) for c in corr.columns]
        z_vals = corr.values.tolist()
        text_vals = [[f"{v:.2f}" for v in row] for row in z_vals]

        trace = {
            "type": "heatmap",
            "x": labels, "y": labels, "z": z_vals,
            "colorscale": "RdBu",
            "zmid": 0,
            "zmin": -1, "zmax": 1,
            "colorbar": {"title": "Pearson r", "thickness": 14},
            "text": text_vals,
            "texttemplate": "%{text}",
            "textfont": {"size": 9},
            "hoverongaps": False,
        }
        n = len(labels)
        dyn_size = max(400, n * 40 + 120)
        layout = {
            **_DARK_LAYOUT,
            "title": {"text": title},
            "xaxis": {"tickangle": -40, "side": "bottom"},
            "yaxis": {"autorange": "reversed"},
            "margin": {"l": 150, "r": 30, "t": 60, "b": 120},
            "height": dyn_size,
        }
        return {"chart_id": _cid(), "chart_type": "correlation_matrix", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    return {"error": f"Unknown chart_type '{chart_type}'. Supported: bar, line, scatter, "
            "pie, histogram, box, violin, area, heatmap, scatter_map, choropleth, "
            "funnel, treemap, sunburst, waterfall, indicator, bubble, "
            "roc_curve, actual_vs_predicted, feature_importance, correlation_matrix."}
