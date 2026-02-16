"""Tool for querying US Census ACS data - free, no API key needed.

Provides demographics, income, transportation, and housing data that
enriches food insecurity risk models with socioeconomic features.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

CENSUS_API = "https://api.census.gov/data"

# Key ACS variables for food access modeling
ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B25044_003E": "no_vehicle_renter",
    "B25044_010E": "no_vehicle_owner",
    "B17001_002E": "below_poverty_count",
    "B23025_005E": "unemployed",
    "B23025_002E": "labor_force",
    "B01002_001E": "median_age",
    "B25003_003E": "renter_occupied",
    "B25003_001E": "total_housing_units",
}

# Missouri FIPS = 29
STATE_FIPS = {"MO": "29", "IL": "17", "AR": "05", "KS": "20",
              "TN": "47", "KY": "21", "OK": "40", "IA": "19"}


@tool
def query_census_acs(
    state: str = "MO",
    year: int = 2022,
    county_fips: str = "",
) -> list[dict[str, Any]]:
    """Query US Census American Community Survey for socioeconomic data.

    Returns county-level demographics, income, poverty, vehicle access, and
    employment data. Useful for building richer food insecurity risk features.

    Args:
        state: Two-letter state code (e.g. "MO").
        year: ACS year (2019-2022 available, default 2022).
        county_fips: Specific 3-digit county FIPS (e.g. "223" for Wayne).
            Empty for all counties in state.
    """
    state_fips = STATE_FIPS.get(state.upper())
    if not state_fips:
        return [{"error": f"Unknown state: {state}", "supported": list(STATE_FIPS.keys())}]

    var_codes = ",".join(["NAME"] + list(ACS_VARIABLES.keys()))
    geo = f"county:{county_fips or '*'}" if county_fips else "county:*"

    url = f"{CENSUS_API}/{year}/acs/acs5"
    params = {
        "get": var_codes,
        "for": geo,
        "in": f"state:{state_fips}",
    }

    try:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()

        if len(rows) < 2:
            return [{"info": "No data returned"}]

        header = rows[0]
        results = []
        for row in rows[1:]:
            record: dict[str, Any] = {}
            for i, col in enumerate(header):
                val = row[i]
                if col == "NAME":
                    record["county"] = val
                elif col in ACS_VARIABLES:
                    try:
                        record[ACS_VARIABLES[col]] = float(val) if val else None
                    except (ValueError, TypeError):
                        record[ACS_VARIABLES[col]] = None
                elif col == "county":
                    record["county_fips"] = val
                elif col == "state":
                    record["state_fips"] = val

            # Compute derived features
            pop = record.get("total_population") or 1
            pov = record.get("below_poverty_count") or 0
            record["poverty_rate_pct"] = round(pov / pop * 100, 1)

            no_veh = (record.get("no_vehicle_renter") or 0) + (record.get("no_vehicle_owner") or 0)
            housing = record.get("total_housing_units") or 1
            record["no_vehicle_pct"] = round(no_veh / housing * 100, 1)

            labor = record.get("labor_force") or 1
            unemp = record.get("unemployed") or 0
            record["unemployment_rate_pct"] = round(unemp / labor * 100, 1)

            results.append(record)

        return results[:100]
    except Exception as e:
        return [{"error": f"Census API failed: {e}"}]
