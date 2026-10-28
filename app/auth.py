import uuid
from datetime import datetime, timedelta
from functools import lru_cache

import bcrypt

from app.config import ADMIN_EMAIL, ALLOWED_EMAILS
from app.db import execute, fetch_one

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_BYTES = 72        # bcrypt only uses the first 72 bytes of a password
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


class AuthError(Exception):
    """A problem to show the user, such as a wrong password or an email that isn't invited."""


@lru_cache(maxsize=1)
def _ensure_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id         TEXT PRIMARY KEY,
            email           TEXT UNIQUE NOT NULL,
            name            TEXT NOT NULL,
            password_hash   TEXT NOT NULL,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until    TEXT,
            created_at      TEXT NOT NULL
        )
    """)


def _hash_password(password: str) -> str:
    # gensalt() creates a random salt; it's stored inside the hash itself
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _password_matches(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return _hash_password("not-a-real-password")


def _public(user: dict) -> dict:
    """The user details the app needs. The password hash never leaves this module."""
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"],
            "is_admin": user["email"] == ADMIN_EMAIL}


def sign_up(email: str, name: str, password: str, confirm: str) -> dict:
    _ensure_table()
    email = email.strip().lower()

    if email not in ALLOWED_EMAILS:
        raise AuthError("This email isn't on the invite list. Ask the app owner to add it.")
    if not name.strip():
        raise AuthError("Please enter your name.")
    if password != confirm:
        raise AuthError("The two passwords don't match.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError("That password is too long (maximum 72 characters).")
    if fetch_one("SELECT user_id FROM users WHERE email = ?", (email,)):
        raise AuthError("An account with this email already exists. Please log in instead.")

    user = {"user_id": uuid.uuid4().hex, "email": email, "name": name.strip()}
    execute(
        "INSERT INTO users (user_id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (user["user_id"], email, user["name"], _hash_password(password),
         datetime.now().isoformat(timespec="seconds")),
    )
    return _public(user)


def log_in(email: str, password: str) -> dict:
    _ensure_table()
    email = email.strip().lower()
    wrong = AuthError("Wrong email or password.")  # never reveal which one was wrong

    user = fetch_one("SELECT * FROM users WHERE email = ?", (email,))
    if not user or len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        # Check against a dummy hash anyway, so a missing account takes as long
        # to reject as a wrong password. Timing can't reveal which emails exist.
        _password_matches("x", _dummy_hash())
        raise wrong

    now = datetime.now()
    if user["locked_until"] and datetime.fromisoformat(user["locked_until"]) > now:
        raise AuthError(f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes.")

    if not _password_matches(password, user["password_hash"]):
        attempts = user["failed_attempts"] + 1
        if attempts >= MAX_FAILED_ATTEMPTS:
            locked_until = (now + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
            execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE user_id = ?",
                    (locked_until, user["user_id"]))
        else:
            execute("UPDATE users SET failed_attempts = ? WHERE user_id = ?",
                    (attempts, user["user_id"]))
        raise wrong

    # Removing an email from ALLOWED_EMAILS revokes access, even for existing accounts
    if email not in ALLOWED_EMAILS:
        raise AuthError("This account no longer has access. Contact the app owner.")

    execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE user_id = ?",
            (user["user_id"],))
    return _public(user)