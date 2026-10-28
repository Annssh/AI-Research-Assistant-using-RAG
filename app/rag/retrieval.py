import re
from functools import lru_cache

from flashrank import Ranker, RerankRequest
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from streamlit import context

from app.config import (MIN_RELEVANCE, RAG_CANDIDATES, RAG_TOP_K, RERANK_MODEL,
                        RERANK_POOL, get_llm, ORIGINAL_QUERY_WEIGHT, QUERY_EXPANSION, RERANK_MODE)

from app.rag.store import get_all_chunks, list_documents, vector_search_ids


# ---------- 1. Query rewriting ----------

class SearchQueries(BaseModel):
    corrected_question: str = Field(
        description="The user's question with spelling and grammar fixed, meaning unchanged")
    alternative_queries: list[str] = Field(
        description="2-3 different search queries for the same need: rephrasings with "
                    "synonyms, and one short keyword-only query")


REWRITE_PROMPT = """You turn a user's question into search queries for a document search engine.
1. Fix spelling mistakes without changing the meaning.
2. If conversation context is given, resolve references like "it", "that" or "this method"
   so that every query makes sense on its own.
3. Write 2-3 alternative queries that could find the answer using different wording:
   use synonyms, expand abbreviations, and include one short keyword-only query."""


def rewrite_query(question: str, context: str = "") -> list[str]:
    """Return [corrected question, alternative queries...], without duplicates."""
    user_input = (f"Conversation so far:\n{context}\n\nNew question: {question}"
                  if context else question)
    result: SearchQueries = get_llm().with_structured_output(SearchQueries).invoke(
        [("system", REWRITE_PROMPT), ("human", user_input)])  # type: ignore
    queries = [result.corrected_question, *result.alternative_queries]
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:4]

# ---------- 2. Keyword search (BM25) ----------


STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for",
             "and", "or", "what", "how", "why", "which", "does", "do", "it", "this",
             "that", "with", "as", "by", "be", "can", "from", "at"}
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


_cache = {"key": None, "bm25": None, "ids": [], "by_id": {}}


def _get_index() -> dict:
    """Build the BM25 index from all chunks in Chroma. Rebuilt only when the library changes."""
    key = tuple(sorted(d["doc_id"] for d in list_documents()))
    if _cache["key"] != key:
        ids, texts, metas = get_all_chunks()
        _cache.update(
            key=key,
            ids=ids,
            by_id={i: (t, m) for i, t, m in zip(ids, texts, metas)},
            bm25=BM25Okapi([_tokenize(t) for t in texts]) if texts else None,
        )
    return _cache


def keyword_search(query: str, k: int = RAG_CANDIDATES,
                   doc_ids: list[str] | None = None) -> list[str]:
    index = _get_index()
    if index["bm25"] is None:
        return []
    allowed = set(doc_ids) if doc_ids is not None else None
    scores = index["bm25"].get_scores(_tokenize(query))

    results = []
    for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
        if scores[i] <= 0:
            break
        chunk_id = index["ids"][i]
        if allowed is not None and index["by_id"][chunk_id][1]["doc_id"] not in allowed:
            continue  # a document this user can't see
        results.append(chunk_id)
        if len(results) == k:
            break
    return results


def vector_search(query: str, k: int = RAG_CANDIDATES,
                  doc_ids: list[str] | None = None) -> list[str]:
    return vector_search_ids(query, k, doc_ids)

# ---------- 4. Reciprocal Rank Fusion ----------

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[str]:
    """Merge ranked lists: each appearance adds weight / (k + rank)."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)  # type: ignore


def hybrid_search(queries: list[str], doc_ids: list[str] | None = None) -> list[str]:
    """Vector + keyword search for every query, fused with weighted RRF.
    queries[0] is the user's own (spell-corrected) question, so it gets extra weight."""
    lists, weights = [], []
    for i, q in enumerate(queries):
        weight = ORIGINAL_QUERY_WEIGHT if i == 0 else 1.0
        lists += [vector_search(q, doc_ids=doc_ids), keyword_search(q, doc_ids=doc_ids)]
        weights += [weight, weight]
    return reciprocal_rank_fusion(lists, weights=weights)


# ---------- 5. Reranking ----------


@lru_cache(maxsize=1)
def _get_ranker() -> Ranker:
    # Downloads the model once (~34 MB) into data/models
    return Ranker(model_name=RERANK_MODEL, cache_dir="data/models", max_length=512)


def rerank(question: str, chunk_ids: list[str], top_k: int) -> list[tuple[str, float]]:
    by_id = _get_index()["by_id"]
    passages = [{"id": cid, "text": by_id[cid][0]} for cid in chunk_ids]
    results = _get_ranker().rerank(RerankRequest(query=question, passages=passages))
    return [(r["id"], float(r["score"])) for r in results[:top_k]]


# ---------- The full pipeline ----------

def retrieve(question: str, top_k: int = RAG_TOP_K, queries: list[str] | None = None,
             min_relevance: float = MIN_RELEVANCE, context: str = "",
             rerank_mode: str = RERANK_MODE, doc_ids: list[str] | None = None) -> list[dict]:
    """Rewrite -> hybrid search -> reranker (filter or rank). Returns chunks with citations."""
    if doc_ids is not None and not doc_ids:
        return []  # the user has no documents
    index = _get_index()
    if not index["by_id"]:
        return []  # the library is empty

    if queries is None:
        queries = rewrite_query(question, context)
        if not QUERY_EXPANSION:
            # keep only the corrected / context-resolved question
            queries = queries[:1]

    pool = [c for c in hybrid_search(queries, doc_ids) if c in index["by_id"]][:RERANK_POOL]
    # score every candidate
    scores = dict(rerank(queries[0], pool, top_k=len(pool)))

    # "rank": order by reranker score. "filter": keep hybrid's order, only drop low scores.
    ordered = sorted(pool, key=scores.get, reverse=True) if rerank_mode == "rank" else pool #type: ignore

    results = []
    for chunk_id in ordered:
        if scores[chunk_id] < min_relevance:
            continue
        text, meta = index["by_id"][chunk_id]
        results.append({
            "id": chunk_id,
            "text": text,
            "filename": meta["filename"],
            "page": meta["page"],
            "score": round(scores[chunk_id], 3),
        })
        if len(results) == top_k:
            break
    return results


def format_source(chunk: dict) -> str:
    """Human-readable citation, e.g. 'book.pdf, p. 21'."""
    return f"{chunk['filename']}, p. {chunk['page']}" if chunk["page"] else chunk["filename"]
