from langchain_core.messages import SystemMessage

from app.config import get_llm
from app.rag.retrieval import format_source, retrieve
from app.rag.store import visible_doc_ids
from app.state import ResearchState

from langchain_core.runnables import RunnableConfig

MAX_HISTORY = 10

CHAT_PROMPT = """You are a helpful research assistant. The user has a finished research report
and is asking follow-up questions.

Use these sources:
1. The report and research notes below.
2. Passages from the user's own documents (labelled [DOC: file, p. N]), when provided.

Rules:
- Be clear and concise. Use Markdown when it helps.
- Cite report facts with the report's [n] numbers, and document passages as (file, p. N).
- If none of the sources contain the answer, say so honestly. You may add general background
  only if you are confident, and label it clearly as not coming from the sources.
- If the sources contain relevant information, use it, even if it only partly answers
  the question, and say what is missing.
- For comparisons ("A vs B", "better / worse"), state the direction exactly as the
  sources do. Double-check you have not reversed it.

=== REPORT ===
{report}

=== RESEARCH NOTES ===
{notes}

=== PASSAGES FROM THE USER'S DOCUMENTS ===
{passages}"""


def _conversation_context(messages, turns: int = 4) -> str:
    """The last few messages before the current question, for resolving 'it' and 'that'."""
    recent = messages[-(turns + 1):-1]
    return "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}" for m in recent
    )


def chat_node(state: ResearchState, config: RunnableConfig) -> dict:
    question = state["messages"][-1].content
    doc_ids = visible_doc_ids(config.get("configurable", {}).get("user_id"))

    passages = "none"
    if doc_ids:
        chunks = retrieve(question, context=_conversation_context(state["messages"]), #type: ignore
                          doc_ids=doc_ids)
        if chunks:
            passages = "\n\n".join(f"[DOC: {format_source(c)}]\n{c['text']}" for c in chunks)
    system = CHAT_PROMPT.format(
        report=state["draft"],
        notes="\n\n".join(state.get("research_notes", [])),
        passages=passages,
    )
    history = state["messages"][-MAX_HISTORY:]

    # The "chat_answer" tag lets us stream ONLY this call, not the query rewriter's JSON
    answer_llm = get_llm().with_config(tags=["chat_answer"])
    response = answer_llm.invoke([SystemMessage(content=system), *history])
    return {"messages": [response], "last_agent": "chat"}
