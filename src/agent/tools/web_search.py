"""General-purpose web search tool using DuckDuckGo.

Uses the `duckduckgo_search` library (DDGS) which returns real web results
with titles, URLs, and snippets — unlike the DuckDuckGo instant-answer API
which only returns Wikipedia abstracts.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


def _ddgs_search(query: str, max_results: int = 8, region: str = "us-en") -> list[dict[str, Any]]:
    """Run a DuckDuckGo web search and return structured results.

    Falls back to the DuckDuckGo instant-answer API if DDGS is unavailable.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region=region, max_results=max_results))

        results = []
        for r in raw:
            results.append({
                "title":   r.get("title", ""),
                "summary": r.get("body", r.get("snippet", "")),
                "url":     r.get("href", r.get("link", "")),
                "source":  "DuckDuckGo",
            })
        return results

    except Exception:
        # Fallback: instant-answer API (limited but always available)
        try:
            import httpx
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            if data.get("Abstract"):
                results.append({
                    "title":   data.get("Heading", "Summary"),
                    "summary": data["Abstract"],
                    "url":     data.get("AbstractURL", ""),
                    "source":  data.get("AbstractSource", "DuckDuckGo"),
                })
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title":   topic.get("Text", "")[:80],
                        "summary": topic.get("Text", ""),
                        "url":     topic.get("FirstURL", ""),
                        "source":  "DuckDuckGo",
                    })
            return results
        except Exception as e:
            return [{"error": f"Web search unavailable: {e}", "query": query}]


@tool
def search_web(
    query: str,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Search the web for any topic and return titles, snippets, and URLs.

    Use this to find current news, reports, scientific articles, alerts,
    or any information that may not be in the database. Ideal for:
    - Current crop disease outbreaks and pest alerts
    - Recent FEMA disaster declarations
    - USDA policy updates and announcements
    - Corn, soybean, wheat market news and price trends
    - Agricultural research findings
    - Supply chain disruption news
    - Weather and drought forecasts
    - Food security program updates

    Args:
        query: Search query (be specific for better results).
        max_results: Number of results to return (default 8, max 15).

    Returns:
        List of dicts with title, summary (snippet), url, and source.
    """
    max_results = min(max_results, 15)
    return _ddgs_search(query, max_results=max_results)


@tool
def search_agricultural_news(
    topic: str,
    region: str = "Missouri",
    year: int = 2026,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search for current agricultural news, disease alerts, and crop reports.

    Specialized search combining agricultural context automatically. Use this
    for domain-specific queries about:
    - Crop diseases (tar spot, gray leaf spot, soybean rust, sudden death syndrome)
    - Pest outbreaks (corn rootworm, aphids, soybean looper, fall armyworm)
    - Drought conditions and weather impacts on crops
    - USDA program updates (SNAP, WIC, crop insurance, FSA loans)
    - Commodity prices and market outlook
    - Food distribution center news
    - Livestock disease (avian flu, swine fever)

    Args:
        topic: The agricultural topic (e.g., "corn disease outbreak", "soybean prices").
        region: Geographic focus (default "Missouri").
        year: Year for time-relevance (default 2026).
        max_results: Number of results (default 10).

    Returns:
        List of dicts with title, summary, url, source fields.
    """
    # Build contextual query
    query = f"{topic} {region} {year} agriculture"
    results = _ddgs_search(query, max_results=max_results)

    if not results or (len(results) == 1 and "error" in results[0]):
        # Try without region if too specific
        results = _ddgs_search(f"{topic} agriculture {year}", max_results=max_results)

    # Tag results with topic metadata
    for r in results:
        r["topic"] = topic
        r["region"] = region
    return results
