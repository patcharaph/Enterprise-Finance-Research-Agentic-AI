from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
import ta
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AgentState
from tools import fetch_price_history, search_news

try:
    from dotenv import load_dotenv

    # Explicitly load local .env and allow override so system envs don't block
    load_dotenv(".env", override=True)
except Exception:
    # Optional dependency for local .env loading
    pass


def _get_llm(model: str = "meta-llama/llama-3.1-70b-instruct", temperature: float = 0.2) -> ChatOpenAI:
    """
    Centralized LLM factory so we can swap models easily.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required for OpenRouter access.")

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", model),
        temperature=temperature,
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def _load_prices(ticker: str) -> pd.DataFrame:
    """
    Pulls about one month of daily data for indicator calculations.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=60)
    df = yf.download(
        ticker,
        start=start.date(),
        end=end.date(),
        progress=False,
        auto_adjust=False,  # explicitly disable to avoid default shifts
    )
    return df.tail(60) if not df.empty else df


def researcher_node(state: AgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    price_history = fetch_price_history(ticker)
    search_results = search_news(ticker)
    return {
        "price_history": price_history,
        "search_results": search_results,
    }


def analyst_node(state: AgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    df = _load_prices(ticker)
    if df.empty or len(df) < 35:
        return {
            "technical_indicators": {},
            "trend_signal": "NO_DATA",
        }

    close = df["Close"]

    indicators: Dict[str, float] = {}

    try:
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        indicators["RSI_14"] = float(rsi_series.dropna().iloc[-1])
    except Exception:
        indicators["RSI_14"] = math.nan

    try:
        macd_calc = ta.trend.MACD(close=close)
        indicators["MACD"] = float(macd_calc.macd().dropna().iloc[-1])
        indicators["MACD_SIGNAL"] = float(macd_calc.macd_signal().dropna().iloc[-1])
    except Exception:
        indicators["MACD"] = math.nan
        indicators["MACD_SIGNAL"] = math.nan

    try:
        indicators["SMA_20"] = float(
            ta.trend.SMAIndicator(close=close, window=20).sma_indicator().dropna().iloc[-1]
        )
        indicators["SMA_50"] = float(
            ta.trend.SMAIndicator(close=close, window=50).sma_indicator().dropna().iloc[-1]
        )
    except Exception:
        indicators["SMA_20"] = math.nan
        indicators["SMA_50"] = math.nan

    # .item() avoids pandas FutureWarning on float conversion
    latest_price = float(close.iloc[-1].item())
    sma20 = indicators.get("SMA_20")
    sma50 = indicators.get("SMA_50")
    rsi = indicators.get("RSI_14")

    trend_signal = "Neutral"
    if all(not math.isnan(x) for x in [latest_price, sma20, sma50, rsi or math.nan]):
        if latest_price > sma20 and latest_price > sma50 and (rsi is not None and rsi >= 55):
            trend_signal = "Bullish bias"
        elif latest_price < sma20 and latest_price < sma50 and (rsi is not None and rsi <= 45):
            trend_signal = "Bearish bias"
        elif latest_price > sma20:
            trend_signal = "Uptrend"
        elif latest_price < sma20:
            trend_signal = "Downtrend"

    indicators["LAST_CLOSE"] = latest_price

    return {
        "technical_indicators": indicators,
        "trend_signal": trend_signal,
    }


def writer_node(state: AgentState) -> Dict[str, Any]:
    llm = _get_llm()
    news_lines = state.get("search_results") or []
    critic_feedback = state.get("critic_feedback") or ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an investment writer creating concise bilingual reports "
                    "(Thai with English technical terms). "
                    "Sections required: Market Overview, Technicals, Risks. "
                    "Cite at least one news item verbatim from provided search_results. "
                    "Keep tone professional and succinct."
                ),
            ),
            (
                "human",
                (
                    "Ticker: {ticker}\n"
                    "Trend signal: {trend_signal}\n"
                    "Technical indicators: {technical_indicators}\n"
                    "Price history (markdown table):\n{price_history}\n"
                    "News (search_results):\n{search_results}\n"
                    "Existing draft (if any): {draft_report}\n"
                    "Critic feedback (if any): {critic_feedback}\n\n"
                    "Write the updated report in Thai with English technical terms. "
                    "If critic_feedback is provided, address it explicitly."
                ),
            ),
        ]
    )

    response = llm.invoke(
        prompt.format_messages(
            ticker=state["ticker"],
            trend_signal=state.get("trend_signal") or "Unknown",
            technical_indicators=state.get("technical_indicators") or {},
            price_history=state.get("price_history") or "N/A",
            search_results="\n".join(news_lines),
            draft_report=state.get("draft_report") or "None",
            critic_feedback=critic_feedback or "None",
        )
    )

    return {
        "draft_report": response.content.strip(),
        # Clear critic feedback once addressed
        "critic_feedback": "",
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    llm = _get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Act as a strict quality critic. "
                    "Validate structure (Market Overview, Technicals, Risks), "
                    "consistency with technical_indicators, and news grounding. "
                    "Return JSON with keys: review_status ('APPROVE' or 'REJECT') "
                    "and critic_feedback in Thai. Reject if any section is missing or "
                    "claims contradict indicators."
                ),
            ),
            (
                "human",
                (
                    "Draft:\n{draft_report}\n\n"
                    "Technical indicators: {technical_indicators}\n"
                    "Trend signal: {trend_signal}\n"
                    "News: {search_results}\n"
                    "Remember: respond ONLY with JSON."
                ),
            ),
        ]
    )

    result = llm.invoke(
        prompt.format_messages(
            draft_report=state.get("draft_report") or "",
            technical_indicators=state.get("technical_indicators") or {},
            trend_signal=state.get("trend_signal") or "Unknown",
            search_results=state.get("search_results") or [],
        )
    )

    review_status = "REJECT"
    critic_feedback = "ไม่ผ่าน เนื่องจากไม่สามารถตรวจสอบคุณภาพได้"

    try:
        parsed = json.loads(result.content)
        review_status = parsed.get("review_status", review_status)
        critic_feedback = parsed.get("critic_feedback", critic_feedback)
    except Exception:
        # fallback heuristic if LLM output is not JSON
        text = result.content.lower()
        if "approve" in text and "reject" not in text:
            review_status = "APPROVE"
            critic_feedback = ""

    revision_count = state.get("revision_count", 0) or 0
    if review_status == "REJECT":
        revision_count += 1

    return {
        "review_status": review_status,
        "critic_feedback": critic_feedback,
        "revision_count": revision_count,
    }
