from pydantic import BaseModel, Field

from app.config import APPROVAL_SCORE, get_llm
from app.state import ResearchState


# The exact shape we want the LLM's answer in.
# The Field descriptions are sent to the LLM, so they act as instructions too.
class Review(BaseModel):
    score: int = Field(
        description="Overall quality from 1 (poor) to 10 (excellent)")
    needs_more_research: bool = Field(
        description="True only if the main problem is missing information that "
                    "requires new web searches. False for writing or structure problems."
    )
    feedback: str = Field(
        description="Specific, actionable instructions for improving the report"
    )


CRITIC_PROMPT = f"""You are a strict but fair editor reviewing a research report.
Judge it on:
1. Accuracy: every claim is supported by the research notes
2. Completeness: it covers the important aspects of the topic
3. Clarity and structure: easy to read, logically organised
4. Citations: claims are cited with [n] numbers

Scoring:
- A score of {APPROVAL_SCORE} or higher means the report is ready to publish.
- Do not withhold approval over minor wording or formatting preferences alone.

Feedback rules:
- Only ask for changes the writer can make using the EXISTING research notes.
- If a problem can only be fixed with information that is not in the notes
  (new sources, statistics, official citations), set needs_more_research to true
  and state exactly what should be searched for.
- Be specific: say exactly what to change, not just what is wrong.
- Images are optional. Never request new images. If an embedded image clearly does not match its section, ask the writer to remove it.
- The References list is generated automatically from verified sources. Never ask for
  references to be added, changed or reformatted."""



def critic_node(state: ResearchState) -> dict:
    print("🧐 Critic: reviewing the draft...")

    # with_structured_output makes the LLM return a Review object instead of text
    reviewer = get_llm().with_structured_output(Review)

    notes = "\n\n".join(state["research_notes"])
    review: Review = reviewer.invoke([
        ("system", CRITIC_PROMPT),
        ("human", f"Topic: {state['topic']}\n\n"
         f"Research notes:\n{notes}\n\n"
         f"Report to review:\n{state['draft']}"),
    ])  # type: ignore

    approved = review.score >= APPROVAL_SCORE  # code enforces the rule, not the LLM
    print(
        f"   Score: {review.score}/10 → {'✅ approved' if approved else '❌ needs work'}")
    if not approved:
        print(f"   More research needed: {review.needs_more_research}")

    return {
        "approved": approved,
        "needs_more_research": review.needs_more_research,
        "critique": "" if approved else review.feedback,
        "last_agent": "critic",
    }
