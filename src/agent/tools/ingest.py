"""Data ingestion tools — fetch, profile, clean, and load datasets into agriflow.db.

Designed for the agriflow-ingest agent. Haiku-compatible (all tools are fast and
deterministic; LLM reasoning only needed for table naming and EDA interpretation).
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")

# Mapping full state names -> 2-letter codes
_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_table_name(name: str) -> str:
    """Sanitize a string into a valid SQLite table name."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:60]  # SQLite limit


def _normalize_state_col(df: pd.DataFrame) -> pd.DataFrame:
    """If a State column contains full names, convert to 2-letter codes."""
    if "State" not in df.columns:
        return df
    sample = df["State"].dropna().iloc[0] if len(df) > 0 else ""
    if len(str(sample)) > 2:
        df["State"] = df["State"].map(_STATE_ABBR).fillna(df["State"])
    return df


@tool
def list_db_tables() -> list[dict[str, Any]]:
    """List all tables currently in agriflow.db with row counts and column counts.

    Returns:
        List of dicts with keys: table, rows, columns, indexes.
    """
    conn = _get_db()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        result = []
        for (table,) in tables:
            rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            idxs = [i[1] for i in conn.execute(f"PRAGMA index_list({table})").fetchall()]
            result.append({"table": table, "rows": rows, "columns": len(cols), "indexes": idxs})
        return result
    finally:
        conn.close()


@tool
def fetch_and_profile_csv(
    url_or_path: str,
    state_filter: str | None = "MO",
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Fetch a CSV/Excel file and return an EDA profile without loading to DB.

    Supports URLs (http/https) or local file paths. Use this before load_dataset
    to inspect structure, check column names, and confirm the data is useful.

    Args:
        url_or_path: URL or local path to the CSV or Excel file.
        state_filter: If 'State' column exists, show stats for this state only.
        sample_rows: Number of sample rows to include in the profile.

    Returns:
        Dict with shape, dtypes, missing value counts, numeric ranges, and sample rows.
    """
    try:
        if url_or_path.startswith("http"):
            import requests
            resp = requests.get(url_or_path, timeout=30)
            resp.raise_for_status()
            if url_or_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(resp.content))
            else:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        else:
            path = Path(url_or_path)
            if not path.exists():
                return {"error": f"File not found: {url_or_path}"}
            if path.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return {"error": str(e)}

    df = _normalize_state_col(df)

    total_rows = len(df)
    if state_filter and "State" in df.columns:
        df_state = df[df["State"] == state_filter.upper()]
        state_rows = len(df_state)
    else:
        df_state = df
        state_rows = total_rows

    # Build profile
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    profile: dict[str, Any] = {
        "total_rows": total_rows,
        "state_rows": state_rows,
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "text_columns": len(text_cols),
        "columns": list(df.columns[:50]),  # first 50 to keep response small
        "missing_pct": {
            col: round(df[col].isnull().mean() * 100, 1)
            for col in df.columns[:20]
            if df[col].isnull().any()
        },
        "numeric_ranges": {
            col: {
                "min": round(float(df_state[col].min()), 3),
                "max": round(float(df_state[col].max()), 3),
                "mean": round(float(df_state[col].mean()), 3),
                "null_pct": round(df_state[col].isnull().mean() * 100, 1),
            }
            for col in numeric_cols[:15]
            if col in df_state.columns and not df_state[col].isnull().all()
        },
        "sample": df_state.head(sample_rows).to_dict(orient="records"),
    }

    if "County" in df.columns and state_filter:
        profile["county_count"] = df_state["County"].nunique()

    return profile


@tool
def load_dataset(
    url_or_path: str,
    table_name: str,
    state_filter: str | None = None,
    replace: bool = False,
    sentinel_value: float = -9999.0,
) -> dict[str, Any]:
    """Fetch, clean, and load a CSV/Excel dataset into agriflow.db.

    Automatically:
    - Converts full state names to 2-letter codes
    - Replaces sentinel values (-9999) with NULL
    - Creates indexes on State, County, FIPS/CensusTract if present
    - Appends to existing table or replaces it (use replace=True cautiously)

    Args:
        url_or_path: URL or local path to the CSV or Excel file.
        table_name: Target SQLite table name (will be sanitized automatically).
        state_filter: If set, only load rows where State matches (e.g. "MO").
        replace: If True, drop and recreate table. If False, append.
        sentinel_value: Numeric value to treat as NULL (default -9999).

    Returns:
        Dict with table name, rows loaded, columns, and index names.
    """
    try:
        if url_or_path.startswith("http"):
            import requests
            resp = requests.get(url_or_path, timeout=60)
            resp.raise_for_status()
            if url_or_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(resp.content))
            else:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        else:
            path = Path(url_or_path)
            if not path.exists():
                return {"error": f"File not found: {url_or_path}"}
            if path.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return {"error": f"Fetch failed: {e}"}

    # Normalize state column
    df = _normalize_state_col(df)

    # Filter by state if requested
    if state_filter and "State" in df.columns:
        df = df[df["State"] == state_filter.upper()].copy()

    # Replace sentinel values with NULL
    df = df.replace(sentinel_value, None)
    df = df.replace(-9999, None)

    # Sanitize table name
    safe_name = _safe_table_name(table_name)
    if_exists = "replace" if replace else "append"

    conn = _get_db()
    try:
        df.to_sql(safe_name, conn, if_exists=if_exists, index=False)

        # Create indexes on key join columns if they exist
        indexes_created = []
        for col, idx_name in [
            ("State", f"idx_{safe_name}_state"),
            ("County", f"idx_{safe_name}_county"),
            ("FIPS", f"idx_{safe_name}_fips"),
            ("CensusTract", f"idx_{safe_name}_tract"),
            ("Year", f"idx_{safe_name}_year"),
        ]:
            if col in df.columns:
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {safe_name}({col})")
                    indexes_created.append(idx_name)
                except Exception:
                    pass
        conn.commit()

        final_count = conn.execute(f"SELECT COUNT(*) FROM {safe_name}").fetchone()[0]
        return {
            "table": safe_name,
            "rows_loaded": len(df),
            "total_rows_in_table": final_count,
            "columns": len(df.columns),
            "indexes": indexes_created,
            "status": "success",
        }
    except Exception as e:
        return {"error": f"Load failed: {e}", "table": safe_name}
    finally:
        conn.close()


@tool
def run_eda_query(
    table: str,
    state: str = "MO",
    query_type: str = "summary",
) -> dict[str, Any]:
    """Run a quick EDA query on any table in agriflow.db.

    Args:
        table: Table name in agriflow.db.
        state: State code to filter (e.g. "MO"). Pass "" for all states.
        query_type: One of "summary" (row/col counts + numeric stats),
                    "missing" (null counts per column),
                    "sample" (first 10 rows),
                    "counties" (distinct counties and row counts).

    Returns:
        EDA results dict.
    """
    conn = _get_db()
    try:
        # Verify table exists
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if table not in tables:
            return {"error": f"Table '{table}' not found. Available: {tables}"}

        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        has_state = "State" in cols
        where = f"WHERE State='{state.upper()}'" if state and has_state else ""

        if query_type == "summary":
            total = conn.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
            numeric_stats = {}
            for col in cols[:20]:
                dtype_row = conn.execute(
                    f"SELECT typeof({col}) FROM {table} {where} LIMIT 1"
                ).fetchone()
                if dtype_row and dtype_row[0] in ("real", "integer"):
                    row = conn.execute(
                        f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table} {where}"
                    ).fetchone()
                    if row and row[0] is not None:
                        numeric_stats[col] = {
                            "min": round(row[0], 3),
                            "max": round(row[1], 3),
                            "mean": round(row[2], 3),
                        }
            return {"rows": total, "columns": len(cols), "numeric_stats": numeric_stats}

        elif query_type == "missing":
            missing = {}
            total = conn.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
            for col in cols[:30]:
                null_count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} {where} AND {col} IS NULL"
                    if where else f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                ).fetchone()[0]
                if null_count > 0:
                    missing[col] = {"null_count": null_count, "pct": round(null_count / total * 100, 1)}
            return {"total_rows": total, "missing_by_column": missing}

        elif query_type == "sample":
            rows = conn.execute(f"SELECT * FROM {table} {where} LIMIT 10").fetchall()
            return {"columns": cols, "sample": [dict(r) for r in rows]}

        elif query_type == "counties":
            if "County" not in cols:
                return {"error": "No County column in table"}
            rows = conn.execute(
                f"SELECT County, COUNT(*) as rows FROM {table} {where} GROUP BY County ORDER BY rows DESC LIMIT 20"
            ).fetchall()
            return {"counties": [dict(r) for r in rows]}

        return {"error": f"Unknown query_type: {query_type}"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


@tool
def drop_table(table: str, confirm: str = "") -> dict[str, Any]:
    """Drop a table from agriflow.db. Requires confirm='yes' to execute.

    Args:
        table: Table name to drop.
        confirm: Must be 'yes' to actually drop. Safety guard.

    Returns:
        Status dict.
    """
    if confirm.lower() != "yes":
        return {"error": "Pass confirm='yes' to drop. This is irreversible."}

    conn = _get_db()
    try:
        conn.execute(f"DROP TABLE IF EXISTS {_safe_table_name(table)}")
        conn.commit()
        return {"status": "dropped", "table": table}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()
