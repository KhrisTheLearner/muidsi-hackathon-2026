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

    x_vals = [r.get(x_col) for r in rows]
    y_vals = [r.get(y_col) for r in rows]

    trace: dict[str, Any] = {"type": "bar", "name": y_col}
    if horizontal:
        trace.update({"x": y_vals, "y": x_vals, "orientation": "h"})
    else:
        trace.update({"x": x_vals, "y": y_vals})

    if color_col:
        trace["marker"] = {"color": [r.get(color_col) for r in rows],
                           "colorscale": "RdYlGn_r", "showscale": True,
                           "colorbar": {"title": color_col}}
    else:
        trace["marker"] = {"color": "#3b82f6"}

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": y_col if horizontal else x_col},
              "yaxis": {"title": x_col if horizontal else y_col}}

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
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]

    traces = []
    for i, col in enumerate(cols):
        traces.append({
            "type": "scatter", "mode": "lines+markers", "name": col,
            "x": [r.get(x_col) for r in rows],
            "y": [r.get(col) for r in rows],
            "line": {"color": palette[i % len(palette)], "width": 2},
            "marker": {"size": 6},
        })

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": x_col},
              "yaxis": {"title": cols[0] if len(cols) == 1 else ""},
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
        marker["colorbar"] = {"title": color_col}
    else:
        marker["color"] = "#ef4444"

    trace["marker"] = marker

    center_lat = sum(lats) / len(lats) if lats else 37.5
    center_lon = sum(lons) / len(lons) if lons else -90.5

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
    x_labels = sorted(set(r.get(x_col) for r in rows))
    y_labels = sorted(set(r.get(y_col) for r in rows))
    lookup = {(r.get(x_col), r.get(y_col)): r.get(z_col, 0) for r in rows}
    z_matrix = [[lookup.get((x, y), 0) for x in x_labels] for y in y_labels]

    trace = {
        "type": "heatmap",
        "x": x_labels, "y": y_labels, "z": z_matrix,
        "colorscale": "RdYlGn_r",
        "colorbar": {"title": z_col},
        "hoverongaps": False,
    }

    layout = {**_DARK_LAYOUT, "title": {"text": title},
              "xaxis": {"title": x_col}, "yaxis": {"title": y_col}}

    return {"chart_id": _cid(), "chart_type": "heatmap", "title": title,
            "plotly_spec": {"data": [trace], "layout": layout}}
