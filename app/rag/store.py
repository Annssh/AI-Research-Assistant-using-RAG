from datetime import datetime
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL, USE_POSTGRES
from app.db import add_column_if_missing, execute, execute_many, fetch_all, fetch_one

CHROMA_DIR = Path("data") / "chroma"


# =============== Vector storage: Chroma locally, Postgres + pgvector online ===============

@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    """Local mode only. Imported lazily, so the online app never loads Chroma at all."""
    from langchain_chroma import Chroma
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name="documents", embedding_function=_get_embeddings(),
                  persist_directory=str(CHROMA_DIR))


def _to_pgvector(vector: list[float]) -> str:
    """pgvector reads vectors written as text: '[0.1,0.2,...]'."""
    return "[" + ",".join(map(str, vector)) + "]"


@lru_cache(maxsize=1)
def _ensure_chunks_table() -> None:
    execute("CREATE EXTENSION IF NOT EXISTS vector")
    execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            filename    TEXT NOT NULL,
            page        INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({EMBEDDING_DIM}) NOT NULL
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id)")


def add_chunks(chunks: list[Document], ids: list[str]) -> None:
    """Embed and store chunks."""
    if not USE_POSTGRES:
        get_vectorstore().add_documents(chunks, ids=ids)
        return

    _ensure_chunks_table()
    vectors = _get_embeddings().embed_documents(
        [c.page_content for c in chunks])
    rows = [
        (chunk_id, c.metadata["doc_id"], c.metadata["filename"], c.metadata["page"],
         c.metadata["chunk_index"], c.page_content, _to_pgvector(vec))
        for chunk_id, c, vec in zip(ids, chunks, vectors)
    ]
    execute_many(
        "INSERT INTO chunks (id, doc_id, filename, page, chunk_index, content, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?::vector) ON CONFLICT (id) DO NOTHING",
        rows,
    )


def delete_chunks(doc_id: str, ids: list[str]) -> None:
    if not USE_POSTGRES:
        get_vectorstore().delete(ids=ids)
        return
    _ensure_chunks_table()
    execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def vector_search_ids(query: str, k: int, doc_ids: list[str] | None = None) -> list[str]:
    """IDs of the k closest chunks. doc_ids=None searches everything; a list restricts it."""
    if doc_ids is not None and not doc_ids:
        return []  # this user can't see any documents

    if not USE_POSTGRES:
        kwargs = {"filter": {"doc_id": {"$in": doc_ids}}
                  } if doc_ids is not None else {}
        docs = get_vectorstore().similarity_search(query, k=k, **kwargs) #type: ignore
        return [f"{d.metadata['doc_id']}-{d.metadata['chunk_index']}" for d in docs]

    _ensure_chunks_table()
    query_vector = _to_pgvector(_get_embeddings().embed_query(query))
    if doc_ids is None:
        rows = fetch_all("SELECT id FROM chunks ORDER BY embedding <=> ?::vector LIMIT ?",
                         (query_vector, k))
    else:
        rows = fetch_all("SELECT id FROM chunks WHERE doc_id = ANY(?) "
                         "ORDER BY embedding <=> ?::vector LIMIT ?",
                         (doc_ids, query_vector, k))
    return [r["id"] for r in rows]


def get_all_chunks() -> tuple[list[str], list[str], list[dict]]:
    """Every chunk's (id, text, metadata). BM25 needs them all to build its index."""
    if not USE_POSTGRES:
        data = get_vectorstore().get(include=["documents", "metadatas"])
        return data["ids"], data["documents"], data["metadatas"]

    _ensure_chunks_table()
    rows = fetch_all(
        "SELECT id, doc_id, filename, page, chunk_index, content FROM chunks")
    ids = [r["id"] for r in rows]
    texts = [r["content"] for r in rows]
    metas = [{"doc_id": r["doc_id"], "filename": r["filename"], "page": r["page"],
              "chunk_index": r["chunk_index"]} for r in rows]
    return ids, texts, metas


# =============== Document registry: a list of what's in the library ===============

@lru_cache(maxsize=1)
def _ensure_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            pages       INTEGER NOT NULL,
            chunks      INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    add_column_if_missing("documents", "owner_id", "TEXT")
    add_column_if_missing("documents", "is_shared",
                          "INTEGER NOT NULL DEFAULT 0")

def add_document_record(doc_id: str, filename: str, file_type: str, pages: int,
                        chunks: int, owner_id: str | None = None) -> None:
    _ensure_table()
    execute(
        "INSERT INTO documents (doc_id, filename, file_type, pages, chunks, uploaded_at, owner_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (doc_id, filename, file_type, pages, chunks,
         datetime.now().isoformat(timespec="seconds"), owner_id),
    )


def list_documents(user_id: str | None = None) -> list[dict]:
    """Documents a user may see: their own plus shared ones. None (local mode) = everything."""
    _ensure_table()
    if user_id is None:
        return fetch_all("SELECT * FROM documents ORDER BY uploaded_at DESC")
    return fetch_all("SELECT * FROM documents WHERE owner_id = ? OR is_shared = 1 "
                     "ORDER BY uploaded_at DESC", (user_id,))


def visible_doc_ids(user_id: str | None) -> list[str]:
    return [d["doc_id"] for d in list_documents(user_id)]


def set_shared(doc_id: str, shared: bool) -> None:
    _ensure_table()
    execute("UPDATE documents SET is_shared = ? WHERE doc_id = ?", (1 if shared else 0, doc_id))


def get_document(doc_id: str) -> dict | None:
    _ensure_table()
    return fetch_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))


def remove_document_record(doc_id: str) -> None:
    _ensure_table()
    execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
