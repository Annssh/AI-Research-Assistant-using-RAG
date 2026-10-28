# import os
# import tempfile
# from pathlib import Path

# from app.rag.ingest import delete_document, ingest_file
# from app.rag.store import list_documents
# import time

# import streamlit as st
# from langgraph.errors import GraphRecursionError

# from app.config import APPROVAL_SCORE, MAX_REVISIONS, MODEL_NAME, USE_POSTGRES, LOGIN_REQUIRED
# from app.graph import build_graph
# from app.storage import (delete_thread, get_checkpointer, list_threads,
#                          new_thread_id, save_thread, get_thread_owner)
# from main import ask_followup, load_research, run_research, status_of

# from app.auth import AuthError, log_in, sign_up

# from app.rag.ingest import delete_document, ingest_file, share_document

# st.set_page_config(page_title="AI Research Assistant",
#                    page_icon="🔬", layout="wide")


# def auth_screen() -> None:
#     st.title("🔬 AI Research Assistant")
#     st.caption("Sign in to see your research.")
#     tab_login, tab_signup = st.tabs(["Log in", "Create account"])

#     with tab_login:
#         with st.form("login"):
#             email = st.text_input("Email")
#             password = st.text_input("Password", type="password")
#             if st.form_submit_button("Log in", type="primary"):
#                 try:
#                     st.session_state.user = log_in(email, password)
#                     st.rerun()
#                 except AuthError as e:
#                     st.error(str(e))

#     with tab_signup:
#         with st.form("signup"):
#             name = st.text_input("Your name")
#             email = st.text_input("Email (must be on the invite list)")
#             password = st.text_input(
#                 "Password (at least 10 characters)", type="password")
#             confirm = st.text_input("Repeat password", type="password")
#             if st.form_submit_button("Create account", type="primary"):
#                 try:
#                     st.session_state.user = sign_up(
#                         email, name, password, confirm)
#                     st.rerun()
#                 except AuthError as e:
#                     st.error(str(e))


# if LOGIN_REQUIRED and "user" not in st.session_state:
#     auth_screen()
#     st.stop()  # nothing below runs until the user is logged in

# USER = st.session_state.get("user")          # None in local mode
# USER_ID = USER["user_id"] if USER else None

# STATUS_ICONS = {"approved": "✅", "best effort": "🟡",
#                 "running": "⏳", "failed": "❌"}


# @st.cache_resource
# def get_engine():
#     """Build the graph and database connection ONCE, and reuse them on every rerun."""
#     checkpointer = get_checkpointer()
#     return build_graph(checkpointer), checkpointer


# graph, checkpointer = get_engine()

# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = None  # None means the "new research" screen


# def open_thread(thread_id):
#     st.session_state.thread_id = thread_id
#     st.rerun()


# def make_live_writer(placeholder):
#     """Return (on_token, reset) callbacks that stream text into a placeholder."""
#     live = {"text": "", "last_render": 0.0}

#     def on_token(token):
#         live["text"] += token
#         now = time.time()
#         if now - live["last_render"] > 0.15:  # re-render at most ~7 times per second
#             placeholder.markdown(live["text"] + " ▌")
#             live["last_render"] = now

#     def reset():
#         live["text"] = ""

#     return on_token, reset


# def document_library() -> None:
#     """Sidebar panel: the user's own documents plus shared ones."""
#     st.session_state.setdefault("uploader_key", 0)
#     docs = list_documents(USER_ID)

#     with st.expander(f"📚 Document library ({len(docs)})"):
#         for kind, text in st.session_state.pop("upload_messages", []):
#             if kind == "ok":
#                 st.success(text)
#             else:
#                 st.error(text)

#         files = st.file_uploader(
#             "Add PDF, Word, TXT or Markdown files (private to you)",
#             type=["pdf", "docx", "txt", "md"],
#             accept_multiple_files=True,
#             key=f"uploader_{st.session_state.uploader_key}",
#         )
#         if files and st.button("📥 Add to library", width="stretch"):
#             messages = []
#             for f in files:
#                 with st.spinner(f"Indexing {f.name}... large files can take a minute"):
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
#                         tmp.write(f.getvalue())
#                         tmp_path = tmp.name
#                     try:
#                         result = ingest_file(tmp_path, filename=f.name, owner_id=USER_ID)
#                         messages.append(("ok", f"{f.name}: {result['status']} "
#                                                f"({result['chunks']} chunks)"))
#                     except Exception as e:
#                         messages.append(("error", f"{f.name}: {e}"))
#                     finally:
#                         os.remove(tmp_path)
#             st.session_state.upload_messages = messages
#             st.session_state.uploader_key += 1
#             st.rerun()

#         if not docs:
#             st.caption("No documents yet. Your uploads are private to you.")

#         for d in docs:
#             is_owner = USER_ID is None or d["owner_id"] == USER_ID
#             badge = ("🌐 Shared · " if d["is_shared"] else "🔒 Private · ") if USER else ""
#             pages = f"{d['pages']} pages · " if d["pages"] else ""

#             col1, col2 = st.columns([5, 1])
#             col1.caption(f"📄 **{d['filename']}**  \n{badge}{pages}{d['chunks']} chunks")
#             if is_owner and col2.button("🗑️", key=f"del_{d['doc_id']}", help="Remove from library"):
#                 delete_document(d["doc_id"], user_id=USER_ID)
#                 st.rerun()

#             # Only the admin can share, and only their own documents
#             if USER and USER["is_admin"] and is_owner:
#                 shared = st.toggle("Share with all users", value=bool(d["is_shared"]),
#                                    key=f"share_{d['doc_id']}")
#                 if shared != bool(d["is_shared"]):
#                     share_document(d["doc_id"], shared, USER)
#                     st.rerun()


# # ---------- Sidebar: history ----------
# with st.sidebar:

#     if USER:
#         st.caption(f"👤 Signed in as **{USER['name']}**")
#         if st.button("Log out", width="stretch"):
#             st.session_state.clear()  # forget the user and everything else in this session
#             st.rerun()

#     if st.button("➕ New research", type="primary", width="stretch"):
#         open_thread(None)

#     document_library()

#     st.subheader("🕘 History")
#     threads = list_threads(USER_ID)
#     if not threads:
#         st.caption("No past research yet.")
#     for t in threads:
#         icon = STATUS_ICONS.get(t["status"], "•")
#         topic_label = t["topic"] if len(
#             t["topic"]) <= 38 else t["topic"][:38] + "…"
#         if st.button(f"{icon} {topic_label}", key=t["thread_id"], width="stretch",
#                      help=f"{t['created_at']} · {t['status']}"):
#             open_thread(t["thread_id"])

#     st.divider()
#     with st.expander("⚙️ Settings"):
#         st.markdown(
#             f"**Storage:** {'Neon Postgres' if USE_POSTGRES else 'Local files'}")
#         st.markdown(f"**Model:** `{MODEL_NAME}`")
#         st.markdown(f"**Max drafts:** {MAX_REVISIONS + 1}")
#         st.markdown(f"**Approval score:** {APPROVAL_SCORE}/10")


# def describe_step(node: str, update: dict) -> str:
#     if node == "supervisor":
#         return f"🧭 **Supervisor** → `{update.get('next_agent')}`"
#     if node == "researcher":
#         return (f"🔎 **Researcher** used {len(update.get('doc_sources', []))} document passages, "
#                 f"{len(update.get('sources', []))} web sources and "
#                 f"{len(update.get('images', []))} images")
#     if node == "writer":
#         return f"✍️ **Writer** finished draft #{update.get('revision_count')}"
#     if node == "critic":
#         return "🧐 **Critic** " + ("approved the draft ✅" if update.get("approved")
#                                   else "requested changes ❌")
#     return node


# def show_result(state: dict, thread_id: str) -> None:
#     st.title(f"🔬 {state['topic']}")
#     st.caption(f"🧵 Thread ID: `{thread_id}`")

#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Drafts written", state.get("revision_count", 0))
#     c2.metric("Approved", "Yes ✅" if state.get("approved") else "Best effort")
#     c3.metric("Sources", len(state.get("sources", [])) +
#               len(state.get("doc_sources", [])))
#     c4.metric("Images found", len(state.get("images", [])))

#     b1, b2 = st.columns(2)
#     with b1:
#         st.download_button("⬇️ Download report (.md)", data=state["draft"],
#                            file_name=f"report-{thread_id[:8]}.md", mime="text/markdown")
#     with b2:
#         if st.button("🗑️ Delete this research"):
#             delete_thread(checkpointer, thread_id)
#             open_thread(None)

#     # Open the report by default; collapse it once a conversation has started
#     with st.expander("📄 Report", expanded=not state.get("messages")):
#         st.markdown(state["draft"])

#     with st.expander("📚 Sources and research notes"):
#         for src in state.get("doc_sources", []):
#             st.markdown(f"- 📄 {src}")
#         for url in state.get("sources", []):
#             st.markdown(f"- {url}")
#         for i, note in enumerate(state.get("research_notes", []), start=1):
#             st.markdown(f"**Research round {i}**")
#             st.markdown(note)


# def show_chat(state: dict, thread_id: str) -> None:
#     st.subheader("💬 Ask follow-up questions")

#     # Past conversation, loaded from the database
#     for msg in state.get("messages", []):
#         with st.chat_message("user" if msg.type == "human" else "assistant"):
#             st.markdown(msg.content)

#     question = st.chat_input("Ask anything about this report...")
#     if question:
#         with st.chat_message("user"):
#             st.markdown(question)
#         with st.chat_message("assistant"):
#             box = st.empty()
#             on_token, _ = make_live_writer(box)
#             try:
#                 answer = ask_followup(graph, thread_id, question, on_token=on_token, user_id=USER_ID)
#                 box.markdown(answer)
#             except Exception as e:
#                 box.error(f"{type(e).__name__}: {e}")


# # ---------- Main area ----------
# if st.session_state.thread_id:
#     thread_id = st.session_state.thread_id

#     if USER_ID and get_thread_owner(thread_id) != USER_ID:
#         st.error("You don't have access to this research.")
#         st.session_state.thread_id = None
#         st.stop()
        
#     state = load_research(graph, thread_id)

#     if state.get("draft"):
#         show_result(state, thread_id)
#         show_chat(state, thread_id)
#     else:
#         st.warning("This research didn't finish, so there's no report to show.")
#         if st.button("🗑️ Delete this research"):
#             delete_thread(checkpointer, thread_id)
#             open_thread(None)

# else:
#     st.title("🔬 AI Research Assistant")
#     st.caption(
#         "A multi-agent LangGraph system: Supervisor → Researcher → Writer → Critic")

#     topic = st.text_input("What should I research?",
#                           placeholder="e.g. How does the Transformer architecture work?")

#     if st.button("🚀 Start research", type="primary", disabled=not topic.strip()):
#         topic = topic.strip()
#         thread_id = new_thread_id()
#         save_thread(thread_id, topic, "running", owner_id=USER_ID)

#         status = st.status("Agents are working...", expanded=True)
#         live_label = st.empty()
#         live_report = st.empty()
#         on_token, reset_live = make_live_writer(live_report)

#         def on_step(node, update, elapsed):
#             status.write(f"`{elapsed:4.0f}s` {describe_step(node, update)}")
#             if node == "supervisor" and update.get("next_agent") == "writer":
#                 reset_live()  # a new draft is starting, so clear the old one
#                 live_label.caption("✍️ The writer is drafting live...")

#         try:
#             final_state = run_research(graph, topic, thread_id, on_step, on_token, user_id=USER_ID)
#         except GraphRecursionError:
#             save_thread(thread_id, topic, "running", owner_id=USER_ID)
#             status.update(label="Stopped: too many steps", state="error")
#             st.error(
#                 "The agents took too many steps without finishing. Try a narrower topic.")
#             st.stop()
#         except Exception as e:
#             save_thread(thread_id, topic, "running", owner_id=USER_ID)
#             status.update(label="Something went wrong", state="error")
#             st.error(f"{type(e).__name__}: {e}")
#             st.stop()

#         save_thread(thread_id, topic, status_of(final_state), owner_id=USER_ID)
#         status.update(label="Research complete!",
#                       state="complete", expanded=False)
#         # reload the page, now showing the saved result + chat
#         open_thread(thread_id)


"""Entry point: login gate, page navigation, and the sidebar shown on every page."""
import streamlit as st

from app.config import LOGIN_REQUIRED
from app.storage import list_threads
from ui import account_page, auth_page, library_page, research_page
from ui.common import current_user, current_user_id

st.set_page_config(page_title="AI Research Assistant", page_icon=":material/science:",
                   layout="wide")
st.session_state.setdefault("thread_id", None)

# ---------- Login gate: nothing below runs until the user is signed in ----------
if LOGIN_REQUIRED and current_user() is None:
    auth_page.render()
    st.stop()

st.logo("assets/logo.svg", icon_image="assets/icon.svg")

# ---------- Pages ----------
research = st.Page(research_page.render, title="Research", icon=":material/science:",
                   url_path="research", default=True)
library = st.Page(library_page.render, title="Library", icon=":material/library_books:",
                  url_path="library")
account = st.Page(account_page.render, title="Account", icon=":material/person:",
                  url_path="account")
navigation = st.navigation([research, library, account])

# ---------- Sidebar (shown on every page) ----------
STATUS_ICONS = {"approved": "✅", "best effort": "🟡", "running": "⏳", "failed": "❌"}

with st.sidebar:
    if st.button("New research", icon=":material/add:", type="primary", width="stretch"):
        st.session_state.thread_id = None
        st.switch_page(research)

    st.caption("HISTORY")
    threads = list_threads(current_user_id())
    if not threads:
        st.caption("Your research will appear here.")
    for t in threads:
        icon = STATUS_ICONS.get(t["status"], "•")
        topic = t["topic"] if len(t["topic"]) <= 34 else t["topic"][:34] + "…"
        is_open = t["thread_id"] == st.session_state.thread_id
        if st.button(f"{icon} {topic}", key=t["thread_id"], width="stretch",
                     type="secondary" if is_open else "tertiary",  # highlight the open one
                     help=f"{t['created_at']} · {t['status']}"):
            st.session_state.thread_id = t["thread_id"]
            st.switch_page(research)

    user = current_user()
    if user:
        st.divider()
        st.caption(f"Signed in as **{user['name']}**")

navigation.run()  # render the selected page