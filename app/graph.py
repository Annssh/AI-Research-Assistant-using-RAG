from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.agents.chat import chat_node
from app.agents.critic import critic_node
from app.agents.researcher import researcher_node
from app.agents.supervisor import supervisor_node
from app.agents.writer import writer_node
from app.state import ResearchState

RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0)


def route_from_start(state: ResearchState) -> str:
    """A question about a finished report goes to the chat agent; anything else starts research."""
    messages = state.get("messages", [])
    if state.get("draft") and messages and isinstance(messages[-1], HumanMessage):
        return "chat_agent"
    return "supervisor"


def route_from_supervisor(state: ResearchState) -> str:
    return state["next_agent"]


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)

    # Nodes
    builder.add_node("supervisor", supervisor_node, retry_policy=RETRY)
    builder.add_node("researcher", researcher_node, retry_policy=RETRY)
    builder.add_node("writer", writer_node, retry_policy=RETRY)
    builder.add_node("critic", critic_node, retry_policy=RETRY)
    builder.add_node("chat_agent", chat_node, retry_policy=RETRY)

    # Entry: research or chat?
    builder.add_conditional_edges(
        START,
        route_from_start,
        {"supervisor": "supervisor", "chat_agent": "chat_agent"},
    )

    # Research loop (unchanged)
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "writer": "writer", "critic": "critic", "FINISH": END},
    )
    for worker in ["researcher", "writer", "critic"]:
        builder.add_edge(worker, "supervisor")

    # Chat: answer once, then stop
    builder.add_edge("chat_agent", END)

    return builder.compile(checkpointer=checkpointer)