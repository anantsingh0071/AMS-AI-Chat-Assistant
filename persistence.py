"""
persistence.py
─────────────────────────────────────────────────────────────────────────
Session state bootstrap, chat thread persistence, and audit logging.
"""

from __future__ import annotations

import os
import json
import uuid
import datetime

import streamlit as st

from config import LOG_FILE, HISTORY_FILE


def init_session_state():
    defaults = {
        "threads":           load_threads(),
        "current_thread_id": None,
        "vectorstore":       None,
        "rag_chain":         None,
        "kb_files":          [],
        "doc_briefs":        {},
        "answered_count":    0,
        "unanswered_count":  0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Threads ───────────────────────────────────────────────────────────────
def load_threads() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_threads(threads: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(threads, f, ensure_ascii=False, indent=2)


def make_thread() -> dict:
    return {
        "id":         str(uuid.uuid4()),
        "title":      "New Chat",
        "messages":   [],
        "created_at": datetime.datetime.now().isoformat(),
    }


def new_thread_title(first_message: str) -> str:
    title = first_message.strip().replace("\n", " ")
    return title[:48] + ("…" if len(title) > 48 else "")


# ── Audit log ─────────────────────────────────────────────────────────────
def log_interaction(user_id: str, query: str, response: str, answered: bool):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id":   user_id,
        "query":     query,
        "response":  response,
        "answered":  answered,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass