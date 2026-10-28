from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_RESEARCH_ROUNDS, MAX_REVISIONS, get_llm
from app.state import ResearchState

MAX_DRAFTS = MAX_REVISIONS + 1  # first draft + allowed rewrites


class Route(BaseModel):
    # The LLM now makes only ONE kind of decision, so only two options exist
    next_agent: Literal["researcher", "writer"]
    reason: str = Field(
        description="One short sentence explaining the decision")


SUPERVISOR_PROMPT = """You are the supervisor of a research team. The critic has just REJECTED the latest draft.
Decide who should act next:

- researcher: the feedback asks for information that is NOT in the current research notes
  (new facts, statistics, official sources, missing subtopics).
- writer: the feedback can be fixed by rewriting with the information already available
  (structure, clarity, wording, citation formatting, removing content).

The critic's 'needs more research' flag is a hint, but read the feedback itself and decide."""


def _go(agent: str, reason: str) -> dict:
    print(f"🧭 Supervisor → {agent} ({reason})")
    return {"next_agent": agent}


def supervisor_node(state: ResearchState) -> dict:
    last = state.get("last_agent")
    drafts = state.get("revision_count", 0)
    research_rounds = len(state.get("research_notes", []))

    # Defensive check: if research exists, some worker has run, so last_agent must be set
    if research_rounds > 0 and not last:
        raise ValueError(
            "Bug: a worker node is not returning 'last_agent'. "
            "Check the return statements in researcher.py, writer.py and critic.py, "
            "and that 'last_agent' is defined in state.py."
        )

    # ---------- Guardrails: workflow rules enforced by code ----------
    if state.get("approved"):
        return _go("FINISH", "draft approved")

    if last == "writer":
        return _go("critic", "every new draft must be reviewed")

    if last == "critic" and drafts >= MAX_DRAFTS:
        return _go("FINISH", f"reached the {MAX_DRAFTS}-draft limit, keeping best effort")

    if research_rounds == 0:
        return _go("researcher", "no research yet")

    if last == "researcher":
        return _go("writer", "new research is ready to write up")

    # ---------- Judgement: only reached after the critic rejects a draft ----------
    if research_rounds >= MAX_RESEARCH_ROUNDS:
        return _go("writer", "research limit reached, revising with existing notes")

    status = (
        f"Topic: {state['topic']}\n"
        f"Research rounds so far: {research_rounds} (maximum {MAX_RESEARCH_ROUNDS})\n"
        f"Critic says more research needed: {state.get('needs_more_research', False)}\n"
        f"Critic feedback:\n{state.get('critique')}"
    )
    router = get_llm().with_structured_output(Route)
    decision: Route = router.invoke(
        [("system", SUPERVISOR_PROMPT), ("human", status)])  # type: ignore
    return _go(decision.next_agent, decision.reason)
