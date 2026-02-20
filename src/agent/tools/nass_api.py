"""Tool for querying USDA NASS Quick Stats API for crop production data."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_core.tools import tool

NASS_API_KEY = os.getenv("NASS_API_KEY", "")
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


@tool
def query_nass(
    commodity: str = "CORN",
    state: str = "MO",
    year: int | None = None,
    stat_type: str = "YIELD",
    agg_level: str = "COUNTY",
) -> list[dict[str, Any]]:
    """Query USDA NASS Quick Stats API for crop production data.

    Returns county or state-level crop yields, production, acreage, and prices
    from the National Agricultural Statistics Service.

    Args:
        commodity: Crop name (e.g. "CORN", "SOYBEANS", "WHEAT").
        state: Two-letter state code (e.g. "MO", "IL").
        year: Specific year. None for most recent available.
        stat_type: Type of statistic - "YIELD", "PRODUCTION", "AREA PLANTED".
        agg_level: Aggregation level - "COUNTY" or "STATE".

    Returns:
        List of dicts with NASS data for matching query.
    """
    if not NASS_API_KEY:
        return [{
            "error": "NASS_API_KEY not set",
            "hint": "Register at https://quickstats.nass.usda.gov/api and add key to .env",
        }]

    params: dict[str, str] = {
        "key": NASS_API_KEY,
        "commodity_desc": commodity.upper(),
        "state_alpha": state.upper(),
        "statisticcat_desc": stat_type.upper(),
        "agg_level_desc": agg_level.upper(),
        "format": "JSON",
    }

    if year:
        params["year"] = str(year)
    else:
        # Default: most recent 3 years
        params["year__GE"] = "2021"

    try:
        resp = httpx.get(NASS_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        # Slim down the response to key fields
        results = []
        for row in data[:100]:  # Cap at 100 rows
            results.append({
                "state": row.get("state_alpha"),
                "county": row.get("county_name"),
                "commodity": row.get("commodity_desc"),
                "year": row.get("year"),
                "stat_type": row.get("statisticcat_desc"),
                "value": row.get("Value"),
                "unit": row.get("unit_desc"),
            })
        return results

    except httpx.HTTPStatusError as e:
        return [{"error": f"NASS API error: {e.response.status_code}", "detail": e.response.text}]
    except Exception as e:
        return [{"error": f"Request failed: {e}"}]
