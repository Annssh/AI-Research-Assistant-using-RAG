"""A tiny database layer for our own tables (threads, documents).
Uses Postgres when DATABASE_URL is set, otherwise a local SQLite file.
Write SQL with '?' placeholders; they are converted to '%s' for Postgres."""
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


import atexit

from app.config import DATABASE_URL, USE_POSTGRES

SQLITE_PATH = Path("data") / "research.db"


@lru_cache(maxsize=1)
def get_pool():
    """A shared pool of Postgres connections, created once and reused."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        max_idle=240,                            # close idle connections before Neon pauses (~5 min)
        check=ConnectionPool.check_connection,   # test each connection before handing it out
        kwargs={
            "autocommit": True,                  # required by LangGraph's PostgresSaver
            "row_factory": dict_row,             # rows come back as dicts, like {"topic": ...}
        },
        open=True,
    )

    atexit.register(pool.close)  # close cleanly when a script finishes
    return pool


@contextmanager
def _sqlite():
    SQLITE_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def execute_many(sql: str, rows: list[tuple]) -> None:
    """Run one statement for many rows. psycopg sends them as a single pipelined batch."""
    if USE_POSTGRES:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql.replace("?", "%s"), rows) #type: ignore
    else:
        with _sqlite() as conn:
            conn.executemany(sql, rows)

def execute(sql: str, params: tuple = ()) -> None:
    if USE_POSTGRES:
        with get_pool().connection() as conn:
            conn.execute(sql.replace("?", "%s"), params) #type: ignore
    else:
        with _sqlite() as conn:
            conn.execute(sql, params)


def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    if USE_POSTGRES:
        with get_pool().connection() as conn:
            return list(conn.execute(sql.replace("?", "%s"), params).fetchall()) #type: ignore
    with _sqlite() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None

def add_column_if_missing(table: str, column: str, definition: str) -> None:
    """Add a column to an existing table. Safe to call every time the app starts."""
    try:
        execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except Exception as e:
        message = str(e).lower()
        # SQLite says "duplicate column name"; Postgres says "... already exists"
        if "duplicate column" not in message and "already exists" not in message:
            raise