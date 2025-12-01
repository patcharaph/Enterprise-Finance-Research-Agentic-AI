from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import analyst_node, critic_node, researcher_node, writer_node
from state import AgentState


def _route_after_critic(state: AgentState, max_revisions: int) -> str:
    """
    Decide where to go after the critic.

    - APPROVE -> end
    - REJECT with remaining quota -> writer
    - Otherwise -> end
    """
    status = state.get("review_status")
    revisions = state.get("revision_count", 0) or 0
    if status == "APPROVE":
        return "end"
    if status == "REJECT" and revisions < max_revisions:
        return "writer"
    return "end"


def build_graph(max_revisions: int = 2, checkpointer: Optional[MemorySaver] = None):
    """
    Construct the LangGraph workflow with cyclic topology and HITL interrupt.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("researcher")

    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")

    workflow.add_conditional_edges(
        "critic",
        lambda state: _route_after_critic(state, max_revisions),
        {
            "writer": "writer",
            "end": END,
        },
    )

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["critic"],
    )


def run_workflow(ticker: str, max_revisions: int = 2) -> Dict[str, Any]:
    """
    Convenience helper to execute the graph end-to-end.
    """
    graph = build_graph(max_revisions=max_revisions, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"cli-{ticker}-{uuid4()}"}}
    state: AgentState = {"ticker": ticker, "revision_count": 0}

    # Run until the graph finishes, auto-supplying neutral human feedback when paused.
    graph.invoke(state, config=config, stream_mode="values")
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return snapshot.values  # finished

        # Assume HITL pause before critic; continue with neutral acknowledgement.
        graph.update_state(
            config,
            {"human_feedback": snapshot.values.get("human_feedback", "AUTO: no human feedback provided")},
        )
        graph.invoke(None, config=config, stream_mode="values")
