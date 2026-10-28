from langchain_core.tools import tool

from app.rag.retrieval import format_source, retrieve


def make_search_documents_tool(doc_ids: list[str]):
    """Create a document-search tool that can ONLY see the given documents."""

    @tool(response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[dict]]:
        """Search the user's own uploaded documents (books, papers, notes).
        Returns the most relevant passages, each labelled with its document name and page number."""
        chunks = retrieve(query, doc_ids=doc_ids)
        if not chunks:
            return ("No relevant passages found in the user's documents for this query. "
                    "Use web search instead."), []
        text_for_llm = "\n\n".join(f"[DOC: {format_source(c)}]\n{c['text']}" for c in chunks)
        return text_for_llm, chunks

    return search_documents