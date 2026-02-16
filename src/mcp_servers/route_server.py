"""Route Optimization & Scheduling MCP Server - thin wrapper around agent/tools/route_optimizer.py.

Run standalone:  python -m src.mcp_servers.route_server
External use:    Claude Desktop, Cursor, or any MCP client via stdio

Core logic lives in src/agent/tools/route_optimizer.py (single source of truth).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent.tools.route_optimizer import (
    calculate_distance as _dist,
    create_route_map as _map,
    optimize_delivery_route as _optimize,
    schedule_deliveries as _schedule,
)

mcp = FastMCP("AgriFlow Routes")


@mcp.tool()
def optimize_delivery_route(origin: str, destinations: str,
                            custom_locations_json: str = ""):
    """Optimize a food delivery route using nearest-neighbor routing."""
    return _optimize.invoke({"origin": origin, "destinations": destinations,
                             "custom_locations_json": custom_locations_json})


@mcp.tool()
def calculate_distance(location_a: str, location_b: str):
    """Calculate straight-line distance and drive time between two locations."""
    return _dist.invoke({"location_a": location_a, "location_b": location_b})


@mcp.tool()
def create_route_map(route_json: str):
    """Generate a Plotly map showing a delivery route with lines and stop markers."""
    return _map.invoke({"route_json": route_json})


@mcp.tool()
def schedule_deliveries(route_json: str, start_time: str = "08:00",
                        loading_minutes: int = 30, unloading_minutes: int = 20):
    """Generate a time-based delivery schedule from a route."""
    return _schedule.invoke({"route_json": route_json, "start_time": start_time,
                             "loading_minutes": loading_minutes,
                             "unloading_minutes": unloading_minutes})


if __name__ == "__main__":
    mcp.run(transport="stdio")
