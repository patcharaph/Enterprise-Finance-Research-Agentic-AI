from __future__ import annotations

from typing import Dict, List, Literal, TypedDict

# Python <3.11 compatibility for PEP 655 markers
try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired  # type: ignore


class AgentState(TypedDict, total=False):
    """
    Shared state that flows through the LangGraph pipeline.

    Fields are optional on the TypedDict to let each node add its own data
    without requiring the entire payload to be present upfront.
    """

    # Input / control
    ticker: str
    revision_count: NotRequired[int]

    # Research data
    price_history: str
    search_results: List[str]

    # Analysis data
    technical_indicators: Dict[str, float]
    trend_signal: str

    # Drafting & review
    draft_report_json: Dict[str, object]
    human_feedback: str
    critic_feedback: str
    review_status: Literal["APPROVE", "REJECT"]
