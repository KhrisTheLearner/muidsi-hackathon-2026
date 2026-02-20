"""Tool for querying weather and drought data."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

# Open-Meteo: free, no API key needed
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# County-seat coordinates for Missouri counties (approximate)
MO_COUNTY_COORDS: dict[str, tuple[float, float]] = {
    "Adair": (40.19, -92.58), "Andrew": (39.98, -94.80), "Atchison": (40.43, -95.41),
    "Audrain": (39.22, -91.85), "Barry": (36.71, -93.83), "Barton": (37.50, -94.35),
    "Bates": (38.25, -94.34), "Benton": (38.30, -93.29), "Bollinger": (37.32, -90.02),
    "Boone": (38.95, -92.33), "Buchanan": (39.77, -94.85), "Butler": (36.76, -90.39),
    "Caldwell": (39.65, -93.98), "Callaway": (38.84, -91.77), "Camden": (38.03, -92.78),
    "Cape Girardeau": (37.31, -89.52), "Carroll": (39.43, -93.50), "Carter": (36.94, -91.00),
    "Cass": (38.65, -94.35), "Cedar": (37.72, -93.83), "Chariton": (39.52, -92.95),
    "Christian": (37.00, -93.19), "Clark": (40.41, -91.73), "Clay": (39.31, -94.42),
    "Clinton": (39.60, -94.41), "Cole": (38.51, -92.17), "Cooper": (38.82, -92.81),
    "Crawford": (37.95, -91.30), "Dade": (37.43, -93.83), "Dallas": (37.68, -93.00),
    "Daviess": (39.89, -93.99), "DeKalb": (39.87, -94.52), "Dent": (37.61, -91.51),
    "Douglas": (36.93, -92.49), "Dunklin": (36.28, -90.05), "Franklin": (38.41, -91.07),
    "Gasconade": (38.39, -91.50), "Gentry": (40.25, -94.41), "Greene": (37.22, -93.29),
    "Grundy": (40.12, -93.57), "Harrison": (40.36, -93.99), "Henry": (38.43, -93.79),
    "Hickory": (37.94, -93.32), "Holt": (40.10, -95.21), "Howard": (39.14, -92.70),
    "Howell": (36.73, -91.85), "Iron": (37.60, -90.69), "Jackson": (39.10, -94.58),
    "Jasper": (37.20, -94.34), "Jefferson": (38.26, -90.53), "Johnson": (38.75, -93.81),
    "Knox": (40.13, -92.15), "Laclede": (37.68, -92.66), "Lafayette": (39.06, -93.78),
    "Lawrence": (36.97, -93.83), "Lewis": (40.10, -91.73), "Lincoln": (39.07, -90.96),
    "Linn": (39.85, -93.11), "Livingston": (39.78, -93.56), "Macon": (39.74, -92.47),
    "Madison": (37.47, -90.32), "Maries": (38.16, -91.72), "Marion": (39.74, -91.53),
    "McDonald": (36.62, -94.35), "Mercer": (40.51, -93.52), "Miller": (38.22, -92.43),
    "Mississippi": (36.88, -89.28), "Moniteau": (38.63, -92.58), "Monroe": (39.49, -92.00),
    "Montgomery": (38.98, -91.50), "Morgan": (38.43, -92.84), "New Madrid": (36.59, -89.53),
    "Newton": (36.88, -94.34), "Nodaway": (40.36, -94.88), "Oregon": (36.69, -91.40),
    "Osage": (38.42, -91.84), "Ozark": (36.65, -92.44), "Pemiscot": (36.26, -89.79),
    "Perry": (37.74, -89.86), "Pettis": (38.73, -93.27), "Phelps": (37.95, -91.77),
    "Pike": (39.34, -91.17), "Platte": (39.38, -94.77), "Polk": (37.62, -93.40),
    "Pulaski": (37.83, -92.20), "Putnam": (40.48, -92.99), "Ralls": (39.52, -91.52),
    "Randolph": (39.44, -92.50), "Ray": (39.35, -94.08), "Reynolds": (37.36, -90.96),
    "Ripley": (36.63, -90.87), "Saline": (39.14, -93.23), "Schuyler": (40.47, -92.52),
    "Scotland": (40.45, -92.15), "Scott": (37.22, -89.57), "Shannon": (37.15, -91.40),
    "Shelby": (39.80, -92.04), "St. Charles": (38.78, -90.50), "St. Clair": (38.03, -93.78),
    "St. Francois": (37.81, -90.37), "St. Louis": (38.63, -90.44),
    "Ste. Genevieve": (37.98, -90.05), "Stoddard": (36.83, -89.89),
    "Stone": (36.75, -93.46), "Sullivan": (40.21, -93.11), "Taney": (36.65, -93.04),
    "Texas": (37.32, -91.97), "Vernon": (37.85, -94.34), "Warren": (38.76, -91.16),
    "Washington": (37.95, -90.88), "Wayne": (37.11, -90.46), "Webster": (37.28, -92.87),
    "Worth": (40.40, -94.44), "Wright": (37.27, -92.47),
    # Independent cities
    "St. Louis City": (38.63, -90.20), "Kansas City": (39.10, -94.58),
}

# City → county lookup for auto-resolution (major MO cities)
MO_CITY_TO_COUNTY: dict[str, str] = {
    "columbia": "Boone", "jefferson city": "Cole", "springfield": "Greene",
    "joplin": "Jasper", "st. joseph": "Buchanan", "cape girardeau": "Cape Girardeau",
    "sedalia": "Pettis", "rolla": "Phelps", "poplar bluff": "Butler",
    "west plains": "Howell", "kirksville": "Adair", "hannibal": "Marion",
    "sikeston": "Scott", "kennett": "Dunklin", "caruthersville": "Pemiscot",
    "dexter": "Stoddard", "farmington": "St. Francois", "warrensburg": "Johnson",
    "marshall": "Saline", "maryville": "Nodaway", "independence": "Jackson",
    "lee's summit": "Jackson", "blue springs": "Jackson", "liberty": "Clay",
    "st. charles": "St. Charles", "o'fallon": "St. Charles",
    "branson": "Taney", "lebanon": "Laclede", "neosho": "Newton",
    "carthage": "Jasper", "nevada": "Vernon", "chillicothe": "Livingston",
    "fulton": "Callaway", "mexico": "Audrain", "moberly": "Randolph",
    "trenton": "Grundy", "brookfield": "Linn", "ava": "Douglas",
    "houston": "Texas", "salem": "Dent", "eminence": "Shannon",
    "van buren": "Carter", "doniphan": "Ripley", "piedmont": "Wayne",
    "greenville": "Wayne", "thayer": "Oregon", "mountain view": "Howell",
    "st. louis": "St. Louis", "kansas city": "Jackson",
}


@tool
def query_weather(
    county: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    days: int = 7,
) -> dict[str, Any]:
    """Get weather forecast for a county or coordinates.

    Uses Open-Meteo (free, no API key) to fetch temperature, precipitation,
    and weather conditions. Useful for assessing drought risk and crop impact.

    Args:
        county: County name (looks up coordinates from built-in table, MO counties).
        latitude: Direct latitude (overrides county lookup).
        longitude: Direct longitude (overrides county lookup).
        days: Forecast days (1-16).

    Returns:
        Dict with daily weather forecast data.
    """
    if latitude is None or longitude is None:
        resolved = None
        if county:
            name = county.replace(" County", "").replace(" county", "").strip()
            # 1) Try exact county match
            if name in MO_COUNTY_COORDS:
                resolved = name
            else:
                # 2) Case-insensitive county match
                for key in MO_COUNTY_COORDS:
                    if key.lower() == name.lower():
                        resolved = key
                        break
            # 3) City-to-county auto-resolution
            if not resolved:
                mapped = MO_CITY_TO_COUNTY.get(name.lower())
                if mapped:
                    resolved = mapped
        if resolved:
            latitude, longitude = MO_COUNTY_COORDS[resolved]
            county = resolved
        else:
            return {
                "error": f"Location '{county}' not found in county or city lookup.",
                "available_counties_sample": list(MO_COUNTY_COORDS.keys())[:20],
                "hint": "Use a Missouri county name (e.g. 'Boone') or a city name (e.g. 'Columbia').",
            }

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,weathercode",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": min(days, 16),
    }

    try:
        resp = httpx.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        return {
            "county": county,
            "latitude": latitude,
            "longitude": longitude,
            "dates": daily.get("time", []),
            "temp_max_f": daily.get("temperature_2m_max", []),
            "temp_min_f": daily.get("temperature_2m_min", []),
            "precipitation_inches": daily.get("precipitation_sum", []),
            "rain_inches": daily.get("rain_sum", []),
        }
    except Exception as e:
        return {"error": f"Weather API failed: {e}"}
