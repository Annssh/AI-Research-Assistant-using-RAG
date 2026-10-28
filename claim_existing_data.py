
"""One-time migration: give all research that has no owner to the admin account."""
from app.rag.store import list_documents
from app.config import ADMIN_EMAIL
from app.db import execute, fetch_one
from app.storage import list_threads

admin = fetch_one("SELECT user_id FROM users WHERE email = ?", (ADMIN_EMAIL,))
if not admin:
    raise SystemExit("❌ Create your account in the app first (with your ADMIN_EMAIL), "
                     "then run this script again.")

list_threads()  # makes sure the owner_id column exists
execute("UPDATE threads SET owner_id = ? WHERE owner_id IS NULL",
        (admin["user_id"],))

count = fetch_one(
    "SELECT COUNT(*) AS n FROM threads WHERE owner_id = ?", (admin["user_id"],))
# print(f"✅ Your account now owns {count['n']} research thread(s).")


list_documents()  # makes sure the owner_id and is_shared columns exist
execute("UPDATE documents SET owner_id = ? WHERE owner_id IS NULL",
        (admin["user_id"],))
docs = fetch_one(
    "SELECT COUNT(*) AS n FROM documents WHERE owner_id = ?", (admin["user_id"],))
# print(f"✅ Your account now owns {docs['n']} document(s), private until you share them.")
