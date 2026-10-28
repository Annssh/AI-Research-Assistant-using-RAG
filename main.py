import re
import time
from pathlib import Path
from langchain_core.messages import AIMessageChunk, HumanMessage

from langgraph.errors import GraphRecursionError

from app.graph import build_graph
from app.storage import get_checkpointer, new_thread_id, save_thread


def make_initial_state(topic: str) -> dict:
    return {
        "topic": topic,
        "research_notes": [],
        "sources": [],
        "images": [],
        "doc_sources": [],
        "draft": "",
        "revision_count": 0,
        "critique": "",
        "approved": False,
        "needs_more_research": False,
        "next_agent": "",
        "last_agent": "",
        "messages": [],
    }


def run_research(graph, topic: str, thread_id: str, on_step=None, on_token=None, user_id: str | None = None) -> dict:
    """
    Run the graph for one thread.
    on_step(node, update, elapsed) is called after each node finishes.
    on_token(text) is called for every token the writer generates.
    """
    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 25,
    }

    final_state = {}
    start = time.time()

    for mode, chunk in graph.stream(
        make_initial_state(topic),
        config=config,
        stream_mode=["updates", "messages", "values"],
    ):
        if mode == "updates" and on_step:
            for node_name, update in chunk.items():
                on_step(node_name, update or {}, time.time() - start)

        elif mode == "messages" and on_token:
            token, metadata = chunk
            # Only stream the writer's tokens. The critic's and supervisor's
            # structured-output JSON would look like noise.
            if (metadata.get("langgraph_node") == "writer"
                    and isinstance(token.content, str) and token.content):
                on_token(token.content)

        elif mode == "values":
            final_state = chunk

    return final_state


def load_research(graph, thread_id: str) -> dict:
    """Load a past run's final state from the database."""
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values


def ask_followup(graph, thread_id: str, question: str, on_token=None, user_id: str | None = None) -> str:
    """Ask a question about a finished report. The thread remembers the conversation."""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}, "recursion_limit": 10}
    final_state = {}

    # Send ONLY the new message. LangGraph merges it into the thread's saved state.
    for mode, chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages" and on_token:
            token, metadata = chunk
            # AIMessageChunk = a live token piece, so the finished message isn't printed twice
            if ("chat_answer" in metadata.get("tags", [])
                    and isinstance(token, AIMessageChunk) and token.content):
                on_token(token.content)
        elif mode == "values":
            final_state = chunk

    return final_state["messages"][-1].content


def status_of(state: dict) -> str:
    return "approved" if state.get("approved") else "best effort"


def save_report(topic: str, report: str) -> Path:
    Path("reports").mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    path = Path("reports") / f"{slug}.md"
    path.write_text(report, encoding="utf-8")
    return path


def print_step(node_name: str, update: dict, elapsed: float) -> None:
    if node_name != "supervisor":
        print(f"   ⏱  {node_name} finished at {elapsed:.0f}s\n")


def print_token(token: str) -> None:
    print(token, end="", flush=True)


if __name__ == "__main__":
    topic = input("🔬 What should I research? ").strip()
    if not topic:
        print("Please enter a topic.")
        raise SystemExit(1)

    graph = build_graph(get_checkpointer())
    thread_id = new_thread_id()
    save_thread(thread_id, topic, "running")
    print(f"🧵 Thread ID: {thread_id}")

    try:
        final_state = run_research(graph, topic, thread_id,
                                   on_step=print_step, on_token=print_token)
    except GraphRecursionError:
        save_thread(thread_id, topic, "failed")
        print("⚠️  The agents took too many steps without finishing. Try a narrower topic.")
        raise SystemExit(1)
    except KeyboardInterrupt:
        save_thread(thread_id, topic, "failed")
        print("\n🛑 Stopped by user.")
        raise SystemExit(1)
    except Exception as e:
        save_thread(thread_id, topic, "failed")
        print(f"❌ Something went wrong: {type(e).__name__}: {e}")
        raise SystemExit(1)

    save_thread(thread_id, topic, status_of(final_state))
    path = save_report(topic, final_state["draft"])
    print("=" * 60)
    print(f"✅ Done! Drafts written: {final_state['revision_count']}, "
          f"approved: {final_state['approved']}")
    print(f"📄 Report saved to: {path}")
    print("\n💬 Ask follow-up questions about the report (press Enter on an empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            break
        print("Assistant: ", end="")
        ask_followup(graph, thread_id, question, on_token=print_token)
        print()
