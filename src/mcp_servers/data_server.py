"""Data MCP Server - wraps live API tools for Archia cloud agents.

Exposes NASS, FEMA, Census ACS, Open-Meteo weather, and web search
as MCP tools so Archia agents can make live API calls.

Run standalone:  python -m src.mcp_servers.data_server
External use:    Archia cloud agents via agriflow-data MCP config
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent.tools.nass_api import query_nass as _nass
from src.agent.tools.fema_disasters import query_fema_disasters as _fema
from src.agent.tools.census_acs import query_census_acs as _census
from src.agent.tools.weather import query_weather as _weather
from src.agent.tools.food_atlas import query_food_atlas as _food_atlas
from src.agent.tools.food_atlas import query_food_access as _food_access
from src.agent.tools.web_search import search_web as _search_web
from src.agent.tools.web_search import search_agricultural_news as _search_ag_news

mcp = FastMCP("AgriFlow Data")


@mcp.tool()
def query_nass(
    commodity: str = "CORN",
    state: str = "MO",
    year: int | None = None,
    stat_type: str = "YIELD",
    agg_level: str = "COUNTY",
):
    """Query USDA NASS Quick Stats API for crop production data.

    Returns county or state-level crop yields, production, acreage, and prices.

    Args:
        commodity: Crop name (e.g. "CORN", "SOYBEANS", "WHEAT").
        state: Two-letter state code (e.g. "MO", "IL").
        year: Specific year (None = most recent 3 years).
        stat_type: "YIELD", "PRODUCTION", "AREA PLANTED", "PRICE RECEIVED".
        agg_level: "COUNTY" or "STATE".
    """
    args: dict = {"commodity": commodity, "state": state, "stat_type": stat_type, "agg_level": agg_level}
    if year is not None:
        args["year"] = year
    return _nass.invoke(args)


@mcp.tool()
def query_fema_disasters(
    state: str = "Missouri",
    disaster_type: str | None = None,
    days_back: int = 365,
    limit: int = 20,
):
    """Query FEMA disaster declarations - free, no API key required.

    Returns recent disaster declarations for a state with dates and types.

    Args:
        state: Full state name or 2-letter code (e.g. "Missouri" or "MO").
        disaster_type: Filter by type e.g. "flood", "drought", "tornado" (None = all).
        days_back: How many days back to search (default 365).
        limit: Max number of records to return (default 20).
    """
    args: dict = {"state": state, "days_back": days_back, "limit": limit}
    if disaster_type is not None:
        args["disaster_type"] = disaster_type
    return _fema.invoke(args)


@mcp.tool()
def query_census_acs(
    state_fips: str = "29",
    variables: list[str] | None = None,
    year: int = 2022,
    geography: str = "county",
):
    """Query US Census ACS demographic and economic data - free, no API key.

    Provides income, population, vehicle access, poverty, and housing data.

    Args:
        state_fips: State FIPS code (Missouri = "29", Illinois = "17").
        variables: ACS variable codes (None = default food-access set).
        year: ACS year (2019-2022 supported).
        geography: "county" or "tract".
    """
    args: dict = {"state_fips": state_fips, "year": year, "geography": geography}
    if variables is not None:
        args["variables"] = variables
    return _census.invoke(args)


@mcp.tool()
def query_weather(
    county: str = "Wayne",
    state: str = "MO",
    forecast_days: int = 7,
):
    """Get current weather forecast for a Missouri county - free, no API key.

    Returns temperature, precipitation, wind speed, and weather conditions.
    Also flags drought risk based on precipitation deficit.

    Args:
        county: County name (e.g. "Wayne", "Cape Girardeau", "St. Louis").
        state: State code (default "MO").
        forecast_days: Days to forecast (1-16, default 7).
    """
    return _weather.invoke({"county": county, "state": state, "forecast_days": forecast_days})


@mcp.tool()
def query_food_atlas(
    state: str = "MO",
    columns: list[str] | None = None,
    limit: int = 50,
):
    """Query USDA Food Environment Atlas data from local SQLite database.

    Returns county-level food insecurity, poverty, food access, and store data.

    Args:
        state: Two-letter state code (default "MO").
        columns: Column names to return (None = all key columns).
        limit: Max rows to return (default 50).
    """
    args: dict = {"state": state, "limit": limit}
    if columns is not None:
        args["columns"] = columns
    return _food_atlas.invoke(args)


@mcp.tool()
def query_food_access(
    state: str = "MO",
    min_laccess_pct: float = 20.0,
    sort_by: str = "PCT_LACCESS_POP15",
    limit: int = 20,
):
    """Query counties with high food desert risk - limited grocery access.

    Returns counties where a significant share of residents lack food access.

    Args:
        state: Two-letter state code (default "MO").
        min_laccess_pct: Minimum % of population with low food access (default 20%).
        sort_by: Column to sort by (default "PCT_LACCESS_POP15").
        limit: Max counties to return (default 20).
    """
    return _food_access.invoke({
        "state": state,
        "min_laccess_pct": min_laccess_pct,
        "sort_by": sort_by,
        "limit": limit,
    })


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 8,
):
    """Search the web for any topic using DuckDuckGo — returns real results.

    Use for current news, scientific articles, USDA policy updates, market data,
    crop disease outbreaks, FEMA declarations, and anything not in the database.

    Args:
        query: Search query (be specific for better results).
        max_results: Number of results to return (default 8, max 15).

    Returns:
        List of dicts with title, summary (snippet), url, and source.
    """
    return _search_web.invoke({"query": query, "max_results": max_results})


@mcp.tool()
def search_agricultural_news(
    topic: str,
    region: str = "Missouri",
    year: int = 2026,
    max_results: int = 10,
):
    """Search for current agricultural news, disease alerts, and crop reports.

    Specialized search combining agricultural context automatically. Ideal for:
    - Crop diseases (tar spot, gray leaf spot, soybean rust, sudden death syndrome)
    - Pest outbreaks (corn rootworm, aphids, soybean looper, fall armyworm)
    - Drought conditions and weather impacts on crops
    - USDA program updates (SNAP, WIC, crop insurance, FSA loans)
    - Commodity prices and market outlook
    - Livestock disease (avian flu, swine fever)

    Args:
        topic: The agricultural topic (e.g., "corn disease outbreak", "soybean prices").
        region: Geographic focus (default "Missouri").
        year: Year for time-relevance (default 2026).
        max_results: Number of results (default 10).
    """
    return _search_ag_news.invoke({
        "topic": topic,
        "region": region,
        "year": year,
        "max_results": max_results,
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
