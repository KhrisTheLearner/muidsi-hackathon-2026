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

    fips_vals = []
    for r in rows:
        f = str(r.get(fips_col, "")).zfill(5)
        fips_vals.append(f)

    z_vals = [r.get(value_col, 0) for r in rows]
    hover = [r.get(text_col, f) for r, f in zip(rows, fips_vals)] if text_col else fips_vals

    trace = {
        "type": "choropleth",
        "geojson": "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        "locations": fips_vals,
        "z": z_vals,
        "text": hover,
        "colorscale": colorscale,
        "colorbar": {"title": value_col},
        "marker": {"line": {"color": "rgba(255,255,255,0.2)", "width": 0.5}},
        "hovertemplate": "%{text}<br>" + value_col + ": %{z:.2f}<extra></extra>",
    }

    layout = {
        **_DARK_LAYOUT,
        "title": {"text": title},
        "geo": {
            "scope": "usa",
            "bgcolor": "rgba(11,15,26,0.7)",
            "lakecolor": "rgba(11,15,26,0.7)",
            "landcolor": "rgba(30,40,60,0.5)",
            "showlakes": True,
            "fitbounds": "locations",
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
                    sunburst, waterfall, indicator, bubble.
        title: Chart title.
        data_json: JSON array of row objects.
        x_col: Column for x-axis (bar, line, scatter, histogram, box, area, funnel, waterfall).
        y_col: Column for y-axis (bar, scatter, box, violin, area, funnel, waterfall).
        y_cols: Comma-separated columns for multi-line charts.
        color_col: Column for color encoding (scatter, bubble, bar).
        size_col: Column for marker size (bubble, scatter_map).
        text_col: Column for labels/hover text.
        names_col: Column for category names (pie, treemap, sunburst).
        values_col: Column for numeric values (pie, treemap, sunburst, indicator).
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
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
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
            marker["colorbar"] = {"title": color_col}
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
                  "xaxis": {"title": x_col}, "yaxis": {"title": y_col}}
        return {"chart_id": _cid(), "chart_type": ct, "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "pie":
        n_col = names_col or x_col
        v_col = values_col or y_col
        err = _check_cols(rows, n_col, v_col)
        if err:
            return {"error": err}
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
        trace = {
            "type": "histogram",
            "x": [r.get(col) for r in rows],
            "nbinsx": nbins,
            "marker": {"color": palette[0], "line": {"color": "#1e293b", "width": 1}},
        }
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": col}, "yaxis": {"title": "Count"}}
        return {"chart_id": _cid(), "chart_type": "histogram", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "box":
        col = y_col or x_col
        err = _check_cols(rows, col)
        if err:
            return {"error": err}
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
                "type": "box", "name": col,
                "y": [r.get(col) for r in rows],
                "marker": {"color": palette[0]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "yaxis": {"title": col}}
        return {"chart_id": _cid(), "chart_type": "box", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "violin":
        col = y_col or x_col
        err = _check_cols(rows, col)
        if err:
            return {"error": err}
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
                "type": "violin", "name": col,
                "y": [r.get(col) for r in rows],
                "box": {"visible": True}, "meanline": {"visible": True},
                "marker": {"color": palette[0]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "yaxis": {"title": col}}
        return {"chart_id": _cid(), "chart_type": "violin", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "area":
        err = _check_cols(rows, x_col)
        if err:
            return {"error": err}
        cols = [c.strip() for c in (y_cols or y_col).split(",")]
        traces = []
        for i, col in enumerate(cols):
            traces.append({
                "type": "scatter", "mode": "lines", "name": col, "fill": "tozeroy" if i == 0 else "tonexty",
                "x": [r.get(x_col) for r in rows],
                "y": [r.get(col) for r in rows],
                "line": {"color": palette[i % len(palette)]},
            })
        layout = {**_DARK_LAYOUT, "title": {"text": title},
                  "xaxis": {"title": x_col}, "yaxis": {"title": cols[0] if len(cols) == 1 else ""}}
        return {"chart_id": _cid(), "chart_type": "area", "title": title,
                "plotly_spec": {"data": traces, "layout": layout}}

    if ct == "funnel":
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
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
        err = _check_cols(rows, n_col, v_col)
        if err:
            return {"error": err}
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
        err = _check_cols(rows, x_col, y_col)
        if err:
            return {"error": err}
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
                  "xaxis": {"title": x_col}, "yaxis": {"title": y_col}}
        return {"chart_id": _cid(), "chart_type": "waterfall", "title": title,
                "plotly_spec": {"data": [trace], "layout": layout}}

    if ct == "indicator" or ct == "gauge":
        v_col = values_col or y_col
        val = rows[0].get(v_col, 0) if rows else 0
        trace = {
            "type": "indicator", "mode": "gauge+number+delta",
            "value": val,
            "title": {"text": text_col or v_col},
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

    return {"error": f"Unknown chart_type '{chart_type}'. Supported: bar, line, scatter, "
            "pie, histogram, box, violin, area, heatmap, scatter_map, choropleth, "
            "funnel, treemap, sunburst, waterfall, indicator, bubble."}
