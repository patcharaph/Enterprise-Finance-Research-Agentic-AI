from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import ta
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AgentState
from tools import fetch_price_history, search_news

try:
    from dotenv import load_dotenv

    load_dotenv(".env", override=True)
except Exception:
    # Optional dependency for local .env loading
    pass

# Encourage LangSmith traces when a key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))

REQUIRED_DRAFT_KEYS = [
    "market_overview_thai",
    "technical_analysis_summary_thai",
    "risk_factors_thai",
    "key_takeaway_thai",
    "cited_sources",
]

DEFAULT_DRAFT: Dict[str, object] = {
    "market_overview_thai": "",
    "technical_analysis_summary_thai": "",
    "risk_factors_thai": "",
    "key_takeaway_thai": "",
    "cited_sources": [],
}


def _get_llm(model: str = "meta-llama/llama-3.1-70b-instruct", temperature: float = 0.2) -> ChatOpenAI:
    """Centralized LLM factory."""
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
    """Pull recent prices for indicator calculations."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=90)
    df = yf.download(
        ticker,
        start=start.date(),
        end=end.date(),
        progress=False,
        auto_adjust=False,
    )
    return df.tail(90) if not df.empty else df


def _coerce_json_response(content: str, existing: Dict[str, object] | None = None) -> Dict[str, object]:
    """
    Parse LLM output into the strict draft schema.

    Best-effort parsing to keep the graph moving even if the model adds
    extra text around the JSON.
    """
    existing = existing or {}
    cleaned = content.strip()

    parsed: Dict[str, object] = {}
    try:
        parsed_obj = json.loads(cleaned)
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj
    except Exception:
        # Try to salvage JSON substring
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed_obj = json.loads(cleaned[start : end + 1])
                if isinstance(parsed_obj, dict):
                    parsed = parsed_obj
            except Exception:
                parsed = {}

    draft = {**DEFAULT_DRAFT, **existing, **parsed}
    # Normalize cited_sources to list[str]
    raw_sources = draft.get("cited_sources", [])
    if isinstance(raw_sources, list):
        draft["cited_sources"] = [str(item) for item in raw_sources]
    else:
        draft["cited_sources"] = [str(raw_sources)]
    return draft


def _validate_draft_structure(draft: Dict[str, object] | None) -> List[str]:
    errors: List[str] = []
    if not isinstance(draft, dict):
        return ["draft_report_json is not a JSON object."]

    for key in REQUIRED_DRAFT_KEYS:
        if key not in draft:
            errors.append(f"Missing key: {key}")
        elif draft[key] in (None, "", [], {}):
            errors.append(f"Empty content for key: {key}")
    if "cited_sources" in draft and not isinstance(draft["cited_sources"], list):
        errors.append("cited_sources must be a list of strings.")
    return errors


def researcher_node(state: AgentState) -> Dict[str, Any]:
    ticker = state["ticker"]
    return {
        "price_history": fetch_price_history(ticker),
        "search_results": search_news(ticker),
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

    latest_price = float(close.iloc[-1].item())
    sma20 = indicators.get("SMA_20")
    sma50 = indicators.get("SMA_50")
    rsi = indicators.get("RSI_14")

    trend_signal = "Neutral"
    if all(not math.isnan(x) for x in [latest_price, sma20 or math.nan, sma50 or math.nan, rsi or math.nan]):
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
    prior_draft = state.get("draft_report_json") or {}
    human_feedback = state.get("human_feedback") or ""
    critic_feedback = state.get("critic_feedback") or ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an investment writer for SET50 equities. "
                    "Write in Thai but keep English technical terms (RSI, MACD, SMA). "
                    "Respond ONLY with strict JSON following the schema and do not wrap in markdown. "
                    "Always incorporate provided feedback explicitly."
                ),
            ),
            (
                "human",
                (
                    "Schema:\n"
                    "{{\"market_overview_thai\": \"...\", \"technical_analysis_summary_thai\": \"...\", "
                    "\"risk_factors_thai\": \"...\", \"key_takeaway_thai\": \"...\", "
                    "\"cited_sources\": [\"Title 1\", \"Title 2\"]}}\n\n"
                    "Ticker: {ticker}\n"
                    "Trend signal: {trend_signal}\n"
                    "Technical indicators: {technical_indicators}\n"
                    "Price history (markdown table):\n{price_history}\n"
                    "News (search_results):\n{search_results}\n"
                    "Previous draft (if any): {previous_draft}\n"
                    "Human feedback to address: {human_feedback}\n"
                    "Critic feedback to address: {critic_feedback}\n\n"
                    "Rules:\n"
                    "- Output only JSON (no pre/post text).\n"
                    "- Cite at least one news headline verbatim in cited_sources when available.\n"
                    "- Mention how feedback was addressed inside the relevant Thai sections.\n"
                ),
            ),
        ]
    )

    formatted_news = "\n".join(news_lines) if news_lines else "No news context"

    response = llm.invoke(
        prompt.format_messages(
            ticker=state["ticker"],
            trend_signal=state.get("trend_signal") or "Unknown",
            technical_indicators=state.get("technical_indicators") or {},
            price_history=state.get("price_history") or "N/A",
            search_results=formatted_news,
            previous_draft=json.dumps(prior_draft, ensure_ascii=False) if prior_draft else "None",
            human_feedback=human_feedback or "None",
            critic_feedback=critic_feedback or "None",
        )
    )

    draft_json = _coerce_json_response(response.content, existing=prior_draft)
    return {
        "draft_report_json": draft_json,
        "critic_feedback": "",
        "review_status": "REJECT",  # reset until the critic approves
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    draft = state.get("draft_report_json")
    revision_count = state.get("revision_count", 0) or 0

    structural_errors = _validate_draft_structure(draft)
    if structural_errors:
        return {
            "review_status": "REJECT",
            "critic_feedback": " ; ".join(structural_errors),
            "revision_count": revision_count + 1,
        }

    llm = _get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Act as a strict quality critic for an investment draft. "
                    "Check: JSON validity and required keys, alignment with technical indicators, "
                    "and whether human feedback is acknowledged. "
                    "Respond only with JSON: {{\"review_status\": \"APPROVE\"|\"REJECT\", \"critic_feedback\": \"...\"}}. "
                    "Reject if anything is missing, contradictory, or ignores feedback."
                ),
            ),
            (
                "human",
                (
                    "Draft JSON:\n{draft_report_json}\n\n"
                    "Technical indicators: {technical_indicators}\n"
                    "Trend signal: {trend_signal}\n"
                    "Price history (for sanity):\n{price_history}\n"
                    "News context: {search_results}\n"
                    "Human feedback to satisfy: {human_feedback}\n"
                    "Current revision count: {revision_count}\n"
                ),
            ),
        ]
    )

    result = llm.invoke(
        prompt.format_messages(
            draft_report_json=json.dumps(draft, ensure_ascii=False),
            technical_indicators=state.get("technical_indicators") or {},
            trend_signal=state.get("trend_signal") or "Unknown",
            price_history=state.get("price_history") or "",
            search_results=state.get("search_results") or [],
            human_feedback=state.get("human_feedback") or "None",
            revision_count=revision_count,
        )
    )

    review_status = "REJECT"
    critic_feedback = "ไม่ผ่าน: ต้องแก้ไขเพื่อให้ตรงตามข้อมูลและ feedback"

    try:
        parsed = json.loads(result.content)
        if isinstance(parsed, dict):
            review_status = parsed.get("review_status", review_status)
            critic_feedback = parsed.get("critic_feedback", critic_feedback)
    except Exception:
        text = result.content.lower()
        if "approve" in text and "reject" not in text:
            review_status = "APPROVE"
            critic_feedback = ""

    if review_status not in ("APPROVE", "REJECT"):
        review_status = "REJECT"

    if review_status == "REJECT":
        revision_count += 1

    return {
        "review_status": review_status,
        "critic_feedback": critic_feedback,
        "revision_count": revision_count,
    }
