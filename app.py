"""
app.py — AMS AI Chat Assistant
Run with: streamlit run app.py
"""

import streamlit as st

import config
import persistence
import sidebar
# from maf import ChatAssistantMAF
# from rag_engine import get_answer, get_retrieved_sources, verify_references, spell_check
from agents.agent_manager import get_chat_agent
# from rag_engine import get_retrieved_sources, verify_references
# from agents.chat_assistant_agent import ChatAssistantAgent

from rag_engine import (
    get_retrieved_sources,
    verify_references,
    
)
from persistence import log_interaction, save_threads, new_thread_title
# from agents.chat_assistant_agent import ChatAssistantAgent
from auth import (
    login,
    logout,
    is_logged_in,
    has_role,
)

# ── Setup ─────────────────────────────────────────────────────────────────
config.configure_page()
persistence.init_session_state()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None
    

config.render_header()
# ======================================================
# Authentication
# ======================================================

if not is_logged_in():

    # st.title("AMS AI Chat Assistant")
    st.subheader("🔐 User Login")

    with st.form("login_form"):

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button("Login")

        if submitted:

            if login(username, password):
                st.success("Login successful.")
                st.rerun()

            else:
                st.error("Invalid username or password.")

    # Stop the application until the user logs in
    st.stop()
# ======================================================
# Role-Based Access
# ======================================================

if not has_role(
    "admin",
    "marketing",
    "sales",
):
    st.error("❌ You are not authorized to use this application.")
    st.stop()

    

# ── Sidebar ───────────────────────────────────────────────────────────────
sidebar.render_sidebar()
# Show logged-in user
st.sidebar.divider()
st.sidebar.write(f"👤 **User:** {st.session_state.username}")
st.sidebar.write(f"🔑 **Role:** {st.session_state.role}")

if st.sidebar.button("🚪 Logout"):
    logout()
    st.rerun()

# ── Ensure active thread ──────────────────────────────────────────────────
if not st.session_state.threads:
    thread = persistence.make_thread()
    st.session_state.threads.insert(0, thread)
    st.session_state.current_thread_id = thread["id"]
    save_threads(st.session_state.threads)

if st.session_state.current_thread_id is None and st.session_state.threads:
    st.session_state.current_thread_id = st.session_state.threads[0]["id"]

active_thread = next(
    (t for t in st.session_state.threads
     if t["id"] == st.session_state.current_thread_id),
    None,
)

if active_thread is None:
    st.info("Click ➕ New Chat in the sidebar to start.")
    st.stop()

# ── Render existing messages ──────────────────────────────────────────────
if not active_thread["messages"]:
    st.markdown(
        "<div style='text-align:center;color:grey;padding:3rem 0;font-size:0.95rem'>"
        "👋 Ask anything about your uploaded documents.<br/>"
        "<span style='font-size:0.8rem'>Answers are grounded in your "
        "knowledge base, with explanations from Phi-3.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in active_thread["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("corrected_from"):
                st.caption(f"✏️ Spell corrected from: *\"{msg['corrected_from']}\"*")

# ── Chat input ────────────────────────────────────────────────────────────
no_docs    = st.session_state.get("vectorstore") is None
user_input = st.chat_input(
    "Ask a question about your documents…",
    disabled=no_docs,
)

if no_docs:
    st.caption("⬆ Upload at least one document in the sidebar to start chatting.")

# ── Handle message ────────────────────────────────────────────────────────
if user_input:
    # user_id = st.session_state.get("user_id_input", "ams_user")
    user_id = st.session_state.username
    kernel = st.session_state.get("kernel")

    agent = get_chat_agent(kernel)

    corrected = user_input
    spell_changed = False

    # # Step 1 — Spell check
    # with st.spinner("✏️ Checking spelling…"):
    #     corrected = spell_check(kernel, user_input)

    # spell_changed = corrected.lower().strip() != user_input.lower().strip()

    # Show user message
    with st.chat_message("user"):
        st.markdown(corrected)
        if spell_changed:
            st.caption(f"✏️ Spell corrected from: *\"{user_input}\"*")

    # Save user message
    user_msg = {"role": "user", "content": corrected}
    if spell_changed:
        user_msg["corrected_from"] = user_input
    active_thread["messages"].append(user_msg)

    # Set thread title from first message
    if active_thread["title"] == "New Chat":
        active_thread["title"] = new_thread_title(corrected)

    # Step 2 — Retrieve + Answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base…"):
            history = active_thread["messages"][:-1]
            answer = agent.run(corrected, chat_history=history)

            st.write("Agent Type:", type(agent))
            st.write("Answer:", repr(answer))

        st.markdown(answer)

        # Reference verification (POC requirement)
        known_files       = {f["name"] for f in st.session_state.kb_files}
        retrieved_sources = get_retrieved_sources(st.session_state.vectorstore, corrected)
        ref_warning       = verify_references(answer, retrieved_sources, known_files)

        if ref_warning:
            st.warning(ref_warning)

        if retrieved_sources:
            with st.expander("📎 Sources used"):
                for src in sorted(retrieved_sources):
                    st.caption(f"• {src}")
        else:
            st.caption("ℹ️ No matching content found in the knowledge base.")

    # Save assistant message
    active_thread["messages"].append({"role": "assistant", "content": answer})

    # Coverage KPI
    answered = (
        "could not find" not in answer.lower() and
        "not been uploaded" not in answer.lower()
    )
    if answered:
        st.session_state.answered_count += 1
    else:
        st.session_state.unanswered_count += 1

    # Audit log
    log_interaction(user_id, corrected, answer, answered)

    # Persist
    save_threads(st.session_state.threads)
    st.rerun()

config.render_footer()
