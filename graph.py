from __future__ import annotations

from typing import Any, Dict

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


def build_graph(max_revisions: int = 2):
    """
    Construct the LangGraph workflow with the required cyclic topology.
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

    return workflow.compile()


def run_workflow(ticker: str, max_revisions: int = 2) -> Dict[str, Any]:
    """
    Convenience helper to execute the graph end-to-end.
    """
    graph = build_graph(max_revisions=max_revisions)
    initial_state: AgentState = {"ticker": ticker, "revision_count": 0}
    return graph.invoke(initial_state)
