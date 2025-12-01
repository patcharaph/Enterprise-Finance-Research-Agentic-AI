from __future__ import annotations

import uuid
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph
from state import AgentState

# Load environment for API keys (OpenRouter, Tavily, LangSmith, etc.)
load_dotenv(".env", override=True)

st.set_page_config(
    page_title="Enterprise Finance Research (SET50) - HITL",
    layout="wide",
)


def _init_session(max_revisions: int) -> None:
    """Initialize or reset session state."""
    st.session_state.graph = build_graph(max_revisions=max_revisions, checkpointer=MemorySaver())
    st.session_state.thread_id = f"ui-{uuid.uuid4()}"
    st.session_state.latest_snapshot = None
    st.session_state.progress_log: List[str] = []
    st.session_state.status = "idle"


def _graph_config() -> Dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _ensure_session_defaults() -> None:
    """Guarantee session keys exist to avoid AttributeError on first render."""
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("thread_id", f"ui-{uuid.uuid4()}")
    st.session_state.setdefault("latest_snapshot", None)
    st.session_state.setdefault("progress_log", [])
    st.session_state.setdefault("status", "idle")


def _run_until_interrupt(input_state: AgentState | None = None) -> None:
    """
    Drive the graph forward until it hits the interrupt_before=['critic'] barrier or finishes.
    Streams node updates so the UI can display progress.
    """
    graph = st.session_state.graph
    config = _graph_config()
    last_node = None
    progress_placeholder = st.empty()

    for update in graph.stream(input_state, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            st.session_state.status = "awaiting_review"
            progress_placeholder.info("Paused for human review before Critic.")
            break

        node, payload = next(iter(update.items()))
        last_node = node
        st.session_state.progress_log.append(node)
        progress_placeholder.info(f"Running node: {node}")
        # Persist latest values for display
        if isinstance(payload, dict):
            if st.session_state.latest_snapshot is None:
                st.session_state.latest_snapshot = payload
            else:
                st.session_state.latest_snapshot.update(payload)

    snapshot = graph.get_state(config)
    st.session_state.latest_snapshot = snapshot.values
    if not snapshot.next:
        st.session_state.status = "finished"
        progress_placeholder.success("Flow completed.")


def _render_progress() -> None:
    if not st.session_state.progress_log:
        st.caption("Progress will appear here once you start a run.")
        return
    last_node = st.session_state.progress_log[-1]
    st.write(f"**Current node:** {last_node}")
    st.progress(min(len(st.session_state.progress_log) / 6, 0.99))


def _render_draft_review(latest: AgentState) -> None:
    st.subheader("Draft Report (Pending Human Review)")
    draft = latest.get("draft_report_json") or {}
    st.json(draft)

    with st.form("human_review_form", clear_on_submit=False):
        feedback = st.text_area(
            "Human Feedback",
            value=latest.get("human_feedback") or "",
            placeholder="e.g., เพิ่มรายละเอียด sentiment จากข่าว, เน้น risk ของเงินบาท",
        )
        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("Approve Draft ✅")
        request_changes = col2.form_submit_button("Request Changes 🔁")

        if approve or request_changes:
            message = feedback.strip()
            if not message:
                message = "อนุมัติให้ส่งต่อ (no additional feedback)." if approve else "ขอปรับแก้เพิ่มเติมก่อนส่งต่อ."
            st.session_state.graph.update_state(_graph_config(), {"human_feedback": message})
            st.session_state.status = "running"
            _run_until_interrupt(None)


def _render_final(latest: AgentState) -> None:
    st.success("Workflow finished.")
    draft = latest.get("draft_report_json") or {}
    st.markdown("### Final Report")
    st.markdown(f"**ตลาดโดยรวม:**\n\n{draft.get('market_overview_thai', '')}")
    st.markdown(f"**เทคนิคอล:**\n\n{draft.get('technical_analysis_summary_thai', '')}")
    st.markdown(f"**ความเสี่ยง:**\n\n{draft.get('risk_factors_thai', '')}")
    st.markdown(f"**Key takeaway:**\n\n{draft.get('key_takeaway_thai', '')}")
    sources = draft.get("cited_sources") or []
    if sources:
        st.markdown("**แหล่งอ้างอิง:**")
        for src in sources:
            st.markdown(f"- {src}")
    st.markdown("---")
    st.markdown("**Price History (last month):**")
    st.code(latest.get("price_history") or "N/A")
    st.markdown(f"**Trend signal:** {latest.get('trend_signal', 'Unknown')}")
    st.markdown(f"**Revision count:** {latest.get('revision_count', 0)}")
    st.markdown(f"**Critic status:** {latest.get('review_status', 'N/A')}")


def main() -> None:
    _ensure_session_defaults()
    st.title("Enterprise Finance Research (LangGraph + HITL)")
    st.caption("StateGraph with human-in-the-loop pause before Critic. SET50 focus.")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="AOT.BK", help="Use SET tickers with .BK suffix (e.g., PTT.BK).")
        max_revisions = st.slider("Max automated revisions", 1, 5, 2)
        if st.button("Start Research"):
            if not ticker:
                st.error("Please provide a ticker.")
            else:
                _init_session(max_revisions)
                st.session_state.status = "running"
                initial_state: AgentState = {"ticker": ticker, "revision_count": 0}
                _run_until_interrupt(initial_state)

    st.markdown("### Real-time Progress")
    _render_progress()

    latest: AgentState = st.session_state.latest_snapshot or {}
    status = st.session_state.status

    if status == "awaiting_review" and latest:
        _render_draft_review(latest)
    elif status == "finished" and latest:
        _render_final(latest)
    elif status == "idle":
        st.info("Enter a ticker and click **Start Research** to begin.")


if __name__ == "__main__":
    main()
