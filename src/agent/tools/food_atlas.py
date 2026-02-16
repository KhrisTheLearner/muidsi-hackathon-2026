"""Tool for querying USDA Food Environment Atlas data from SQLite."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")

# Valid columns for food_environment table (whitelist to prevent SQL injection)
_VALID_COLS = {
    "FIPS", "State", "County", "PCT_LACCESS_POP15", "FOODINSEC_15_17",
    "PCT_OBESE_ADULTS17", "POVRATE15", "PCT_SNAP16", "PCT_NSLP15",
    "PCT_FREE_LUNCH15", "PCT_REDUCED_LUNCH15", "PCT_LACCESS_CHILD15",
    "PCT_LACCESS_SENIORS15", "PCT_LACCESS_HHNV15", "PCT_LACCESS_SNAP15",
    "GROC16", "SUPERC16", "CONVS16", "SPECS16", "SNAPS17", "WICS16",
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
    columns: str = "FIPS,State,County,PCT_LACCESS_POP15,FOODINSEC_15_17,PCT_OBESE_ADULTS17,POVRATE15,PCT_SNAP16,PCT_NSLP15",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query the USDA Food Environment Atlas for county-level food environment data.

    Returns food insecurity rates, SNAP participation, poverty rates, food access
    indicators, and other variables from the Food Environment Atlas.

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
