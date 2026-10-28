import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # Input
    topic: str

    # Researcher output (append, don't overwrite)
    research_notes: Annotated[list[str], operator.add]
    sources: Annotated[list[str], operator.add]
    images: Annotated[list[dict], operator.add]

    # Writer output
    draft: str
    revision_count: int

    # Critic output
    critique: str
    approved: bool
    needs_more_research: bool

    # Supervisor / routing
    next_agent: str
    last_agent: str

    # Follow-up chat history (add_messages = the standard chat reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    images: Annotated[list[dict], operator.add]
    doc_sources: Annotated[list[str], operator.add]
