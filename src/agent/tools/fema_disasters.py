"""Tool for querying FEMA disaster declarations - free, no API key."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

FEMA_API = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

# Map full state names and common abbreviations to FEMA's 2-letter codes
# All 50 states + DC so the agent can handle any user query
_STATE_MAP = {
    "alabama": "AL", "al": "AL", "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "az": "AZ", "arkansas": "AR", "ar": "AR",
    "california": "CA", "ca": "CA", "colorado": "CO", "co": "CO",
    "connecticut": "CT", "ct": "CT", "delaware": "DE", "de": "DE",
    "district of columbia": "DC", "dc": "DC",
    "florida": "FL", "fl": "FL", "georgia": "GA", "ga": "GA",
    "hawaii": "HI", "hi": "HI", "idaho": "ID", "id": "ID",
    "illinois": "IL", "il": "IL", "indiana": "IN", "in": "IN",
    "iowa": "IA", "ia": "IA", "kansas": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY", "louisiana": "LA", "la": "LA",
    "maine": "ME", "me": "ME", "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "ma": "MA", "michigan": "MI", "mi": "MI",
    "minnesota": "MN", "mn": "MN", "mississippi": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO", "montana": "MT", "mt": "MT",
    "nebraska": "NE", "ne": "NE", "nevada": "NV", "nv": "NV",
    "new hampshire": "NH", "nh": "NH", "new jersey": "NJ", "nj": "NJ",
    "new mexico": "NM", "nm": "NM", "new york": "NY", "ny": "NY",
    "north carolina": "NC", "nc": "NC", "north dakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH", "oklahoma": "OK", "ok": "OK",
    "oregon": "OR", "or": "OR", "pennsylvania": "PA", "pa": "PA",
    "rhode island": "RI", "ri": "RI", "south carolina": "SC", "sc": "SC",
    "south dakota": "SD", "sd": "SD", "tennessee": "TN", "tn": "TN",
    "texas": "TX", "tx": "TX", "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT", "virginia": "VA", "va": "VA",
    "washington": "WA", "wa": "WA", "west virginia": "WV", "wv": "WV",
    "wisconsin": "WI", "wi": "WI", "wyoming": "WY", "wy": "WY",
}


@tool
def query_fema_disasters(
    state: str = "MO",
    disaster_type: str = "",
    year_start: int = 2018,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query FEMA disaster declarations for a state.

    Returns historical disaster data (floods, storms, droughts, fires) useful
    for risk modeling and disaster response planning.

    Args:
        state: State name or 2-letter code (e.g. "MO" or "Missouri").
        disaster_type: Filter: "DR" (major disaster), "EM" (emergency),
            "FM" (fire), or "" for all.
        year_start: Only return disasters from this year onward (default 2018).
        limit: Max results (default 50).
    """
    code = _STATE_MAP.get(state.lower(), state.upper()[:2])

    filt = f"state eq '{code}'"
    if disaster_type:
        filt += f" and declarationType eq '{disaster_type}'"
    if year_start:
        filt += f" and fyDeclared ge {year_start}"

    params: dict[str, Any] = {
        "$filter": filt,
        "$orderby": "declarationDate desc",
        "$top": min(limit, 100),
    }

    try:
        resp = httpx.get(FEMA_API, params=params, timeout=15)
        resp.raise_for_status()
        records = resp.json().get("DisasterDeclarationsSummaries", [])

        results = []
        for r in records:
            results.append({
                "disaster_number": r.get("disasterNumber"),
                "title": r.get("declarationTitle"),
                "type": r.get("declarationType"),
                "date": r.get("declarationDate", "")[:10],
                "state": r.get("state"),
                "county": r.get("designatedArea", "").replace(" (County)", ""),
                "incident_type": r.get("incidentType"),
                "programs_declared": {
                    "individual_assistance": r.get("iaProgramDeclared", False),
                    "public_assistance": r.get("paProgramDeclared", False),
                    "hazard_mitigation": r.get("hmProgramDeclared", False),
                },
            })
        return results or [{"info": f"No disasters found for {code} since {year_start}"}]
    except Exception as e:
        return [{"error": f"FEMA API failed: {e}"}]
