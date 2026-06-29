"""
sidebar.py
─────────────────────────────────────────────────────────────────────────
Renders the Streamlit sidebar.
"""

from __future__ import annotations

import streamlit as st
import traceback

from persistence import save_threads, make_thread
from rag_engine import ingest_files, generate_doc_brief
from config import SUPPORTED_EXTENSIONS


def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='margin-bottom:4px'>🤖 AMS Assistant</h2>"
            "<p style='color:grey;font-size:0.78rem;margin-top:0'>"
            "Enterprise Knowledge Platform</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_user_id()
        st.divider()
        _render_chat_threads()
        st.divider()
        _render_knowledge_base()
        st.divider()
        _render_coverage_kpi()


def _render_user_id():
    st.subheader("👤 User")
    st.text_input(
        "User ID",
        key="user_id_input",
        value=st.session_state.get("user_id_input", "ams_user"),
        help="Used for audit logging.",
    )
    st.caption("All queries are logged for governance.")


def _render_chat_threads():
    st.subheader("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        thread = make_thread()
        st.session_state.threads.insert(0, thread)
        st.session_state.current_thread_id = thread["id"]
        save_threads(st.session_state.threads)
        st.rerun()

    if not st.session_state.threads:
        st.info("No conversations yet.")
        return

    for t in st.session_state.threads:
        active = t["id"] == st.session_state.current_thread_id
        label  = f"🟢 **{t['title']}**" if active else f"💬 {t['title']}"
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"t_{t['id']}", use_container_width=True):
                st.session_state.current_thread_id = t["id"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"d_{t['id']}"):
                st.session_state.threads = [
                    x for x in st.session_state.threads if x["id"] != t["id"]
                ]
                if st.session_state.current_thread_id == t["id"]:
                    st.session_state.current_thread_id = None
                save_threads(st.session_state.threads)
                st.rerun()


def _render_knowledge_base():
    st.subheader("📚 Knowledge Base")
    st.caption("Upload organisational documents.")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )

    if uploaded_files:
        existing_names = [f["name"] for f in st.session_state.kb_files]
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            with st.status("Building Knowledge Base...", expanded=True) as status:
                try:
                    st.write("📄 Reading documents...")

                    vs, kernel, meta, per_file_text = ingest_files(
                        new_files,
                        existing_vectorstore=st.session_state.vectorstore,
)

                    st.write("🧠 Creating embeddings...")
                    st.write("✅ Updating FAISS index...")
                    st.write("🤖 Initializing AI Assistant...")

                    st.session_state.vectorstore = vs
                    st.session_state.kernel = kernel
                    st.session_state.kb_files.extend(meta)

                    status.update(
                        label="Knowledge Base Ready ✅",
                        state="complete",
                    )

                except Exception as e:
                    traceback.print_exc()
                    st.exception(e)
                    status.update(
                        label=f"Failed: {e}",
                        state="error",
                    )
                    per_file_text = {}

            for fname, text in per_file_text.items():
                if fname not in st.session_state.doc_briefs:
                    with st.spinner(f"📝 Generating brief for {fname}..."):
                        st.session_state.doc_briefs[fname] = generate_doc_brief(text)

    st.markdown("### 📂 Uploaded Documents")

    if st.session_state.kb_files:
        st.caption(f"{len(st.session_state.kb_files)} document(s) indexed")

        for file in st.session_state.kb_files:
            with st.expander(f"✅ {file['name']}"):
                brief = st.session_state.doc_briefs.get(file["name"])

                if brief:
                    st.markdown(brief)
                else:
                    st.caption("Brief not yet generated.")
    else:
        st.info("No documents uploaded yet.")

def _render_coverage_kpi():
    st.subheader("📊 Coverage")
    total = st.session_state.answered_count + st.session_state.unanswered_count
    if total == 0:
        st.info("No questions asked yet.")
        return
    pct = st.session_state.answered_count / total
    st.progress(pct)
    st.metric("Knowledge Coverage", f"{pct * 100:.0f}%")
    st.caption(
        f"✅ {st.session_state.answered_count} Answered\n\n"
        f"❌ {st.session_state.unanswered_count} Unanswered"
    ) 