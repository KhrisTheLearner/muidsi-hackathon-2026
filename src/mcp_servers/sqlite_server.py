"""SQLite MCP Server - thin wrapper around agent/tools/sql_query.py.

Run standalone:  python -m src.mcp_servers.sqlite_server
External use:    Claude Desktop, Cursor, or any MCP client via stdio

Core logic lives in src/agent/tools/sql_query.py (single source of truth).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent.tools.sql_query import list_tables as _list_tables
from src.agent.tools.sql_query import run_sql_query as _run_sql_query

mcp = FastMCP("AgriFlow SQLite")


@mcp.tool()
def list_tables():
    """List all tables in the AgriFlow database with column names and row counts."""
    return _list_tables.invoke({})


@mcp.tool()
def run_sql_query(query: str, limit: int = 100):
    """Execute a read-only SQL query. Only SELECT/WITH/PRAGMA/EXPLAIN allowed."""
    return _run_sql_query.invoke({"query": query, "limit": limit})


if __name__ == "__main__":
    mcp.run(transport="stdio")
