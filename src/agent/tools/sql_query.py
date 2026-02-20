"""LangChain tool wrappers for SQLite queries.

Single source of truth — MCP servers import from here.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from langchain_core.tools import tool

DB_PATH = os.getenv("DB_PATH", "data/agriflow.db")
_ALLOWED = ("SELECT", "PRAGMA", "EXPLAIN", "WITH")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def list_tables() -> list[dict[str, Any]]:
    """List all tables in the AgriFlow database with column names and row counts.

    Use this first to discover what data is available before writing SQL queries.
    Returns table names, their columns, and row counts.
    """
    conn = _connect()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        result = []
        for (table_name,) in tables:
            cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
            result.append({
                "table": table_name,
                "columns": [col[1] for col in cols],
                "row_count": count,
            })
        return result
    except sqlite3.OperationalError as e:
        return [{"error": str(e)}]
    finally:
        conn.close()


@tool
def run_sql_query(query: str, limit: int = 100) -> list[dict[str, Any]]:
    """Execute a read-only SQL query against the AgriFlow SQLite database.

    The database contains USDA Food Environment Atlas, Food Access Atlas,
    and other agricultural datasets. Use list_tables first to see available
    tables and columns.

    Only SELECT/WITH/PRAGMA/EXPLAIN queries are allowed. A LIMIT clause
    is auto-appended if not present (max 500 rows).

    Args:
        query: SQL SELECT statement to execute.
        limit: Maximum rows to return (default 100, max 500).
    """
    stripped = query.strip()
    upper = stripped.upper()
    if not any(upper.startswith(p) for p in _ALLOWED):
        return [{"error": "Only SELECT/WITH/PRAGMA/EXPLAIN queries are allowed."}]

    limit = min(limit, 500)
    if "LIMIT" not in upper:
        query = stripped.rstrip("; ") + f" LIMIT {limit}"

    conn = _connect()
    try:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError as e:
        return [{"error": str(e)}]
    finally:
        conn.close()
