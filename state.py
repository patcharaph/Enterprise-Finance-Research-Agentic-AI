from __future__ import annotations

from typing import Dict, List, Literal, NotRequired, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state that flows through the LangGraph pipeline.

    Fields are optional on the TypedDict to let each node add its own data
    without requiring the entire payload to be present upfront.
    """

    # Input
    ticker: str

    # Research data
    price_history: str
    search_results: List[str]

    # Analysis data
    technical_indicators: Dict[str, float]
    trend_signal: str

    # Drafting
    draft_report: str
    critic_feedback: str
    review_status: Literal["APPROVE", "REJECT"]

    # Control
    revision_count: NotRequired[int]
