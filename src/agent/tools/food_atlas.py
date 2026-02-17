"""Tool for querying USDA Food Environment Atlas data from SQLite."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")

# Valid columns for food_environment table (whitelist to prevent SQL injection).
# NOTE: FOODINSEC_* and PCT_SNAP* are state-level aggregates (same value per state).
# County-level variation exists in poverty, food access, income, and store count columns.
_VALID_COLS = {
    # Identifiers
    "FIPS", "State", "County",
    # Food insecurity (state-level aggregates - uniform within each state)
    "FOODINSEC_18_20", "FOODINSEC_21_23", "VLFOODSEC_18_20", "VLFOODSEC_21_23",
    "CH_FOODINSEC_20_23",
    # Poverty & income (county-level variation)
    "POVRATE21", "PERPOV17_21", "CHILDPOVRATE21", "DEEPCHILDPOVRATE21",
    "DEEPPOVRATE21", "MEDHHINC21",
    # Low food access - best county-level risk indicators
    "PCT_LACCESS_POP15", "PCT_LACCESS_POP19", "LACCESS_POP15", "LACCESS_POP19",
    "PCT_LACCESS_LOWI15", "PCT_LACCESS_LOWI19", "LACCESS_LOWI15",
    "PCT_LACCESS_CHILD15", "PCT_LACCESS_CHILD19",
    "PCT_LACCESS_SENIORS15", "PCT_LACCESS_SENIORS19",
    "PCT_LACCESS_HHNV15", "PCT_LACCESS_HHNV19",
    "PCT_LACCESS_SNAP15", "PCT_LACCESS_SNAP19",
    # SNAP / assistance (county-level)
    "SNAPS17", "SNAPS23", "SNAPSPTH17", "SNAPSPTH23",
    "PCT_SNAP17", "PCT_SNAP22", "WICS16", "WICS22",
    # Food stores (county-level)
    "GROC16", "GROC20", "GROCPTH16", "GROCPTH20",
    "SUPERC16", "SUPERC20", "SUPERCPTH16", "SUPERCPTH20",
    "CONVS16", "CONVS20", "CONVSPTH16", "CONVSPTH20",
    "SPECS16", "SPECS20", "FFR16", "FFR20", "FSR16", "FSR20",
    # School programs (county-level)
    "PCT_NSLP17", "PCT_NSLP21", "PCT_FREE_LUNCH15", "PCT_REDUCED_LUNCH15",
    "PCT_SBP17", "PCT_SBP21",
    # Health outcomes (county-level)
    "PCT_DIABETES_ADULTS15", "PCT_DIABETES_ADULTS19",
    "PCT_OBESE_ADULTS17", "PCT_OBESE_ADULTS22",
    # Farmers markets
    "FMRKT13", "FMRKT18", "FMRKTPTH13", "FMRKTPTH18",
    # Demographics
    "PCT_HISP20", "PCT_NHBLACK20", "PCT_NHWHITE20",
    "PCT_18YOUNGER20", "PCT_65OLDER20",
    # Geography
    "METRO23", "POPLOSS15",
}


def _sanitize_columns(columns: str) -> str:
    """Validate column names against whitelist. Returns safe SQL column string."""
    if columns == "*":
        return "*"
    requested = [c.strip() for c in columns.split(",")]
    safe = [c for c in requested if c in _VALID_COLS]
    return ",".join(safe) if safe else "*"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def query_food_atlas(
    state: str | None = None,
    county: str | None = None,
    columns: str = "FIPS,State,County,POVRATE21,MEDHHINC21,PCT_LACCESS_POP15,PCT_LACCESS_POP19,PCT_LACCESS_LOWI15,FOODINSEC_21_23,SNAPS23,GROC20,SUPERC20",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query the USDA Food Environment Atlas for county-level food environment data.

    Returns poverty rates, food access indicators, SNAP counts, grocery store
    availability, and other county-level variables. Note: FOODINSEC_* columns
    are state-level aggregates (same value for all counties in a state). For
    county-level risk analysis use POVRATE21, PCT_LACCESS_POP15, PCT_LACCESS_LOWI15.

    Args:
        state: Two-letter state code to filter by (e.g. "MO", "IL"). None for all.
        county: County name to filter by (partial match). None for all in state.
        columns: Comma-separated column names to return.
        limit: Max rows to return.

    Returns:
        List of dicts with requested columns for matching counties.
    """
    conn = _get_db()
    cols = _sanitize_columns(columns)
    query = f"SELECT {cols} FROM food_environment WHERE 1=1"
    params: list[str] = []

    if state:
        query += " AND State = ?"
        params.append(state.upper())
    if county:
        query += " AND County LIKE ?"
        params.append(f"%{county}%")

    query += f" LIMIT {limit}"

    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError as e:
        return [{"error": str(e), "hint": "Run the data pipeline first: python -m src.data_pipeline.load_atlas"}]
    finally:
        conn.close()


@tool
def query_food_access(
    state: str | None = None,
    county: str | None = None,
    urban_rural: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query the USDA Food Access Research Atlas for food desert data.

    Returns census-tract level food desert classifications including distance
    to nearest supermarket, low-access population counts, and vehicle access.

    Args:
        state: Two-letter state code (e.g. "MO").
        county: County name (partial match).
        urban_rural: Filter by "Urban" or "Rural".
        limit: Max rows to return.

    Returns:
        List of dicts with food access data for matching tracts.
    """
    conn = _get_db()
    query = "SELECT * FROM food_access WHERE 1=1"
    params: list[str] = []

    if state:
        query += " AND State = ?"
        params.append(state.upper())
    if county:
        query += " AND County LIKE ?"
        params.append(f"%{county}%")
    if urban_rural:
        query += " AND Urban = ?"
        params.append("1" if urban_rural.lower() == "urban" else "0")

    query += f" LIMIT {limit}"

    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError as e:
        return [{"error": str(e), "hint": "Run the data pipeline first: python -m src.data_pipeline.load_atlas"}]
    finally:
        conn.close()
