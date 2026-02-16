"""LangChain tool wrappers for route optimization and scheduling.

Single source of truth — MCP servers import from here.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool

LOCATIONS: dict[str, tuple[float, float]] = {
    "Wayne County": (37.11, -90.46),
    "Pemiscot County": (36.26, -89.79),
    "Oregon County": (36.69, -91.40),
    "Dunklin County": (36.28, -90.05),
    "Ripley County": (36.63, -90.87),
    "Shannon County": (37.15, -91.40),
    "Mississippi County": (36.88, -89.28),
    "New Madrid County": (36.59, -89.53),
    "Boone County": (38.95, -92.33),
    "Jackson County": (39.10, -94.58),
    "Butler County": (36.76, -90.39),
    "Stoddard County": (36.83, -89.89),
    "Scott County": (37.22, -89.57),
    "Iron County": (37.60, -90.69),
    "Springfield Distribution Center": (37.22, -93.29),
    "St. Louis Food Bank": (38.63, -90.20),
    "Cape Girardeau Hub": (37.31, -89.52),
    "Poplar Bluff Hub": (36.76, -90.39),
    "Sikeston Warehouse": (36.88, -89.59),
    "Kennett Distribution": (36.24, -90.06),
    "West Plains Hub": (36.73, -91.85),
    "Farmington Hub": (37.78, -90.42),
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_neighbor(start: str, dests: list[str],
                      coords: dict[str, tuple[float, float]]) -> tuple[list[str], float]:
    route = [start]
    remaining = set(dests) - {start}
    total = 0.0
    current = start
    while remaining:
        nearest = min(remaining, key=lambda d: _haversine(*coords[current], *coords[d]))
        total += _haversine(*coords[current], *coords[nearest])
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return route, round(total, 1)


@tool
def optimize_delivery_route(
    origin: str,
    destinations: str,
    custom_locations_json: str = "",
) -> dict[str, Any]:
    """Optimize a food delivery route using nearest-neighbor routing.

    Calculates the most efficient path from origin through all destinations,
    with distance and drive time estimates for rural Missouri roads (45 mph avg).

    Args:
        origin: Starting location (e.g. "Cape Girardeau Hub").
        destinations: Comma-separated destination names (e.g. "Wayne County, Pemiscot County").
        custom_locations_json: Optional JSON array of extra locations,
            e.g. '[{"name":"My Place","lat":37.0,"lon":-90.0}]'

    Available locations: Wayne County, Pemiscot County, Oregon County,
    Dunklin County, Ripley County, Shannon County, Mississippi County,
    New Madrid County, Boone County, Jackson County, Butler County,
    Stoddard County, Scott County, Iron County, Springfield Distribution Center,
    St. Louis Food Bank, Cape Girardeau Hub, Poplar Bluff Hub,
    Sikeston Warehouse, Kennett Distribution, West Plains Hub, Farmington Hub.
    """
    coords = dict(LOCATIONS)
    if custom_locations_json:
        try:
            for loc in json.loads(custom_locations_json):
                coords[loc["name"]] = (loc["lat"], loc["lon"])
        except (json.JSONDecodeError, KeyError):
            pass

    dest_list = [d.strip() for d in destinations.split(",")]
    missing = [loc for loc in [origin] + dest_list if loc not in coords]
    if missing:
        return {"error": f"Unknown locations: {missing}", "available": sorted(coords.keys())}

    route, total_miles = _nearest_neighbor(origin, dest_list, coords)

    legs = []
    for i in range(len(route) - 1):
        dist = _haversine(*coords[route[i]], *coords[route[i + 1]])
        legs.append({
            "from": route[i], "to": route[i + 1],
            "distance_miles": round(dist, 1),
            "est_drive_minutes": round((dist / 45) * 60),
            "from_lat": coords[route[i]][0], "from_lon": coords[route[i]][1],
            "to_lat": coords[route[i + 1]][0], "to_lon": coords[route[i + 1]][1],
        })

    return {
        "optimized_route": route,
        "total_distance_miles": total_miles,
        "total_stops": len(route),
        "est_total_drive_minutes": sum(leg["est_drive_minutes"] for leg in legs),
        "legs": legs,
        "method": "nearest-neighbor heuristic (45 mph avg rural MO roads)",
    }


@tool
def calculate_distance(location_a: str, location_b: str) -> dict[str, Any]:
    """Calculate straight-line distance and drive time between two locations.

    Args:
        location_a: First location name (from available locations list).
        location_b: Second location name.
    """
    if location_a not in LOCATIONS:
        return {"error": f"Unknown: {location_a}", "available": sorted(LOCATIONS.keys())}
    if location_b not in LOCATIONS:
        return {"error": f"Unknown: {location_b}", "available": sorted(LOCATIONS.keys())}

    dist = _haversine(*LOCATIONS[location_a], *LOCATIONS[location_b])
    return {"from": location_a, "to": location_b,
            "distance_miles": round(dist, 1),
            "est_drive_minutes": round((dist / 45) * 60)}


@tool
def create_route_map(route_json: str) -> dict[str, Any]:
    """Generate a Plotly map showing a delivery route with lines and stop markers.

    Call optimize_delivery_route first, then pass its result here to visualize.

    Args:
        route_json: JSON string from optimize_delivery_route result.
    """
    route_data = json.loads(route_json) if isinstance(route_json, str) else route_json
    legs = route_data.get("legs", [])
    route_order = route_data.get("optimized_route", [])
    if not legs:
        return {"error": "No legs data in route_json"}

    coords = dict(LOCATIONS)

    line_lats: list[float | None] = []
    line_lons: list[float | None] = []
    for leg in legs:
        line_lats.extend([leg["from_lat"], leg["to_lat"], None])
        line_lons.extend([leg["from_lon"], leg["to_lon"], None])

    line_trace = {
        "type": "scattermapbox", "mode": "lines",
        "lat": line_lats, "lon": line_lons,
        "line": {"color": "#3b82f6", "width": 3},
        "name": "Route",
    }

    stop_lats = [coords[s][0] for s in route_order if s in coords]
    stop_lons = [coords[s][1] for s in route_order if s in coords]
    stop_labels = [f"{i+1}. {s}" for i, s in enumerate(route_order) if s in coords]
    n = len(stop_lats)

    marker_trace = {
        "type": "scattermapbox", "mode": "markers+text",
        "lat": stop_lats, "lon": stop_lons,
        "text": stop_labels,
        "textposition": "top center",
        "textfont": {"size": 11, "color": "#e2e8f0"},
        "marker": {
            "size": [18] + [14] * max(0, n - 2) + ([14] if n > 1 else []),
            "color": (["#10b981"] + ["#f59e0b"] * max(0, n - 2) + (["#ef4444"] if n > 1 else []))[:n],
            "opacity": 0.9,
        },
        "name": "Stops",
    }

    center_lat = sum(stop_lats) / len(stop_lats) if stop_lats else 37.0
    center_lon = sum(stop_lons) / len(stop_lons) if stop_lons else -90.0
    total_miles = route_data.get("total_distance_miles", 0)
    total_min = route_data.get("est_total_drive_minutes", 0)

    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(11,15,26,0.95)",
        "font": {"color": "#e2e8f0"},
        "title": {"text": f"Delivery Route - {total_miles} mi, ~{total_min} min"},
        "mapbox": {"style": "carto-darkmatter",
                   "center": {"lat": center_lat, "lon": center_lon}, "zoom": 7},
        "height": 500,
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
        "showlegend": True,
        "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(0,0,0,0.5)"},
    }

    return {"chart_id": f"route_{uuid.uuid4().hex[:8]}", "chart_type": "route_map",
            "title": f"Delivery Route ({len(route_order)} stops)",
            "plotly_spec": {"data": [line_trace, marker_trace], "layout": layout}}


@tool
def schedule_deliveries(
    route_json: str,
    start_time: str = "08:00",
    loading_minutes: int = 30,
    unloading_minutes: int = 20,
) -> dict[str, Any]:
    """Generate a time-based delivery schedule from a route.

    Call optimize_delivery_route first, then pass its result here.

    Args:
        route_json: JSON from optimize_delivery_route.
        start_time: Departure time HH:MM (default "08:00").
        loading_minutes: Loading time at origin (default 30).
        unloading_minutes: Unloading time per stop (default 20).
    """
    route_data = json.loads(route_json) if isinstance(route_json, str) else route_json
    legs = route_data.get("legs", [])
    route_order = route_data.get("optimized_route", [])
    if not legs:
        return {"error": "No legs data"}

    h, m = map(int, start_time.split(":"))
    current = datetime(2026, 1, 1, h, m)

    schedule = [{
        "stop": 1, "location": route_order[0], "activity": "Loading",
        "arrive": current.strftime("%H:%M"),
        "depart": (current + timedelta(minutes=loading_minutes)).strftime("%H:%M"),
        "duration_min": loading_minutes,
    }]
    current += timedelta(minutes=loading_minutes)

    for i, leg in enumerate(legs):
        drive_min = leg["est_drive_minutes"]
        current += timedelta(minutes=drive_min)
        arrive = current.strftime("%H:%M")
        current += timedelta(minutes=unloading_minutes)
        schedule.append({
            "stop": i + 2, "location": leg["to"], "activity": "Delivery",
            "arrive": arrive, "depart": current.strftime("%H:%M"),
            "drive_from_prev_min": drive_min, "unloading_min": unloading_minutes,
            "distance_from_prev_miles": leg["distance_miles"],
        })

    return {
        "schedule": schedule, "start_time": start_time,
        "est_end_time": current.strftime("%H:%M"),
        "total_stops": len(route_order),
        "total_distance_miles": route_data.get("total_distance_miles", 0),
    }
