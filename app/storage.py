import sqlite3
import uuid
from datetime import datetime
from functools import lru_cache

from app.config import USE_POSTGRES
from app.db import SQLITE_PATH, execute, fetch_all, get_pool, add_column_if_missing, fetch_one


def get_checkpointer():
    """LangGraph's checkpointer: Postgres online, a local SQLite file otherwise."""
    if USE_POSTGRES:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver(get_pool()) #type: ignore
        checkpointer.setup()  # creates LangGraph's own tables on first run; safe to repeat
        return checkpointer

    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_PATH.parent.mkdir(exist_ok=True)
    return SqliteSaver(sqlite3.connect(SQLITE_PATH, check_same_thread=False))


@lru_cache(maxsize=1)
def _ensure_table() -> None:
    """Create the threads table once per run (cached, so it's one query, not one per call)."""
    execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id  TEXT PRIMARY KEY,
            topic      TEXT NOT NULL,
            status     TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    add_column_if_missing("threads", "owner_id", "TEXT")


def new_thread_id() -> str:
    return uuid.uuid4().hex

def save_thread(thread_id: str, topic: str, status: str, owner_id: str | None = None) -> None:
    _ensure_table()
    execute(
        """
        INSERT INTO threads (thread_id, topic, status, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (thread_id) DO UPDATE SET status = excluded.status
        """,
        (thread_id, topic, status, datetime.now().isoformat(timespec="seconds"), owner_id),
    )


def list_threads(owner_id: str | None = None, limit: int = 50) -> list[dict]:
    """A user's threads. owner_id=None (local mode, no login) lists everything."""
    _ensure_table()
    if owner_id is None:
        return fetch_all("SELECT thread_id, topic, status, created_at FROM threads "
                         "ORDER BY created_at DESC LIMIT ?", (limit,))
    return fetch_all("SELECT thread_id, topic, status, created_at FROM threads "
                     "WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?", (owner_id, limit))


def get_thread_owner(thread_id: str) -> str | None:
    _ensure_table()
    row = fetch_one("SELECT owner_id FROM threads WHERE thread_id = ?", (thread_id,))
    return row["owner_id"] if row else None



def delete_thread(checkpointer, thread_id: str) -> None:
    """Delete a thread's saved graph states and its history entry."""
    checkpointer.delete_thread(thread_id)
    _ensure_table()
    execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))