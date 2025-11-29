from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd
import yfinance as yf
from langchain_community.tools.tavily_search import TavilySearchResults

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; skip if unavailable
    pass


def _get_date_range(days: int = 35) -> Tuple[datetime, datetime]:
    """Return UTC date window for yfinance queries."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start, end


def fetch_price_history(ticker: str) -> str:
    """
    Fetch the last ~1 month of daily OHLCV data and return as Markdown.

    Returns a string so it can be dropped directly into the report while also
    keeping the analyst logic free to compute indicators separately.
    """
    start, end = _get_date_range()
    try:
        df = yf.download(
            ticker,
            start=start.date(),
            end=end.date(),
            progress=False,
        )
    except Exception as exc:  # yfinance occasionally raises on bad tickers
        return f"No price data found for {ticker} (error: {exc})"

    if df.empty:
        return f"No price data found for {ticker}."

    df = df.tail(22)  # roughly one trading month
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep_cols]
    return df.to_markdown(index=False, floatfmt=".2f")


def search_news(ticker: str, max_results: int = 5) -> List[str]:
    """
    Run a Tavily search for the ticker. Gracefully handle API/key failures.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ["TAVILY_API_KEY not set; skipping web search."]

    query = (
        f"{ticker} stock news analyst rating thailand outlook "
        f"{datetime.now(timezone.utc).year}"
    )

    tool = TavilySearchResults(api_key=api_key, max_results=max_results)

    try:
        results = tool.run(query)
    except Exception as exc:
        return [f"No news found (error: {exc})"]

    if not results:
        return ["No news found."]

    formatted = []
    for item in results:
        if isinstance(item, dict):
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            content = item.get("content") or ""
            snippet = content[:280] + ("..." if len(content) > 280 else "")
            formatted.append(f"{title} — {url} — {snippet}")
        else:
            formatted.append(str(item))

    return formatted
