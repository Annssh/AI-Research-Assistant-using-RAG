import json
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

from app.config import get_llm
from app.rag.retrieval import format_source

from langchain_core.runnables import RunnableConfig

from app.rag.store import visible_doc_ids
from app.rag.tools import make_search_documents_tool

from app.state import ResearchState
from app.tools import web_search

MAX_IMAGE_CANDIDATES = 12
MAX_IMAGES = 6

RESEARCHER_PROMPT = """You are a meticulous research assistant.
Your job is to gather accurate facts about a topic using your tools.

Rules:
- Run 2-3 focused web searches, each covering a different angle of the topic. Never more than 4.
- If the topic involves a system, process, model or architecture, make one web search
  specifically aimed at finding a diagram (for example: "<topic> architecture diagram").
- Only use facts found in your tool results. Never invent facts.
- When finished, reply with 5-10 concise bullet points of key findings.
- End each bullet with its source: a URL in square brackets like [https://...],
  or a document label exactly as the tool gave it, like [DOC: book.pdf, p. 21]."""

DOCS_RULES = """

You also have the user's own document library, through the search_documents tool:
- ALWAYS call search_documents first, with 1-2 focused queries about the topic.
- Prefer facts from the user's documents. Use web search for anything they don't cover
  and for recent developments."""


def _search_results(messages):
    """Yield each web search result (from the ToolMessages) as a Python dict."""
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content)  # type: ignore
        except (json.JSONDecodeError, TypeError):
            continue  # document-search results are plain text, so they are skipped here
        if isinstance(data, dict):
            yield data


def _extract_urls(messages) -> list[str]:
    urls = []
    for data in _search_results(messages):
        urls += [r["url"] for r in data.get("results", []) if "url" in r]
    return list(dict.fromkeys(urls))


def _extract_doc_sources(messages) -> list[str]:
    """Read citations from the document tool's structured artifact, not from its text."""
    sources = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "search_documents" and msg.artifact:
            sources += [format_source(chunk) for chunk in msg.artifact]
    return list(dict.fromkeys(sources))


def _is_working_image(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    try:
        with httpx.stream("GET", url, timeout=5, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"}) as response:
            content_type = response.headers.get("content-type", "")
            return response.status_code == 200 and content_type.startswith("image/")
    except Exception:
        return False


def _extract_images(messages) -> list[dict]:
    images, seen = [], set()
    for data in _search_results(messages):
        for img in data.get("images") or []:
            url = img.get("url") if isinstance(img, dict) else img
            description = img.get(
                "description", "") if isinstance(img, dict) else ""
            if url and url not in seen:
                seen.add(url)
                images.append({"url": url, "description": description})

    candidates = images[:MAX_IMAGE_CANDIDATES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_is_working_image, [
                       img["url"] for img in candidates]))
    working = [img for img, ok in zip(candidates, results) if ok]
    return working[:MAX_IMAGES]


def researcher_node(state: ResearchState, config: RunnableConfig) -> dict:
    print("🔎 Researcher: searching...")

    # Who is running this? It comes from the login session, via the graph's config
    user_id = config.get("configurable", {}).get("user_id")
    doc_ids = visible_doc_ids(user_id)
    has_docs = bool(doc_ids)

    agent = create_agent(
        model=get_llm(),
        tools=[make_search_documents_tool(doc_ids), web_search] if has_docs else [
            web_search],
        system_prompt=RESEARCHER_PROMPT + (DOCS_RULES if has_docs else ""),
    )

    task = f"Research this topic: {state['topic']}"
    if state.get("critique"):
        task += (
            "\n\nA reviewer said the previous research was lacking. "
            f"Focus on filling these gaps:\n{state['critique']}"
        )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        # slightly higher: document searches add steps
        config={"recursion_limit": 16},
    )

    messages = result["messages"]
    notes = messages[-1].content
    sources = _extract_urls(messages)
    doc_sources = _extract_doc_sources(messages)
    images = _extract_images(messages)

    print(f"   ✓ Done. {len(sources)} web sources, {len(doc_sources)} document passages, "
          f"{len(images)} images.")
    return {
        "research_notes": [notes],
        "sources": sources,
        "doc_sources": doc_sources,
        "images": images,
        "last_agent": "researcher",
    }
