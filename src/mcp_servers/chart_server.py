"""Chart & Map Generation MCP Server - thin wrapper around agent/tools/chart_generator.py.

Run standalone:  python -m src.mcp_servers.chart_server
External use:    Claude Desktop, Cursor, or any MCP client via stdio

Core logic lives in src/agent/tools/chart_generator.py (single source of truth).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent.tools.chart_generator import (
    create_bar_chart as _bar,
    create_line_chart as _line,
    create_risk_heatmap as _heatmap,
    create_scatter_map as _scatter,
)

mcp = FastMCP("AgriFlow Charts")


@mcp.tool()
def create_bar_chart(title: str, data_json: str, x_col: str, y_col: str,
                     color_col: str = "", horizontal: bool = False):
    """Create a Plotly bar chart specification."""
    return _bar.invoke({"title": title, "data_json": data_json, "x_col": x_col,
                        "y_col": y_col, "color_col": color_col, "horizontal": horizontal})


@mcp.tool()
def create_line_chart(title: str, data_json: str, x_col: str, y_cols: str):
    """Create a Plotly line chart for trends and time series."""
    return _line.invoke({"title": title, "data_json": data_json,
                         "x_col": x_col, "y_cols": y_cols})


@mcp.tool()
def create_scatter_map(title: str, data_json: str, lat_col: str, lon_col: str,
                       size_col: str = "", color_col: str = "", text_col: str = "",
                       zoom: int = 6):
    """Create a geographic scatter map (Plotly scattermapbox)."""
    return _scatter.invoke({"title": title, "data_json": data_json, "lat_col": lat_col,
                            "lon_col": lon_col, "size_col": size_col, "color_col": color_col,
                            "text_col": text_col, "zoom": zoom})


@mcp.tool()
def create_risk_heatmap(title: str, data_json: str, x_col: str, y_col: str, z_col: str):
    """Create a risk assessment heatmap."""
    return _heatmap.invoke({"title": title, "data_json": data_json,
                            "x_col": x_col, "y_col": y_col, "z_col": z_col})


if __name__ == "__main__":
    mcp.run(transport="stdio")
