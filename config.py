"""
config.py
─────────────────────────────────────────────────────────────────────────
Central settings for the AMS AI Chat Assistant.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── File paths ───────────────────────────────────────────────────────────
LOG_FILE     = "audit_log.jsonl"
HISTORY_FILE = "chat_threads.json"

# ── Models ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_TEMPERATURE = 0.7
# ── Gemini (Testing Only) ───────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# -------LLM MODE------------------------------
LLM_MODE = os.getenv("LLM_MODE", "gemini").lower()

# ── Agent Framework ───────────────────────────────────────────────
AGENT_FRAMEWORK = "Microsoft Agent Framework"

# ── Retrieval ────────────────────────────────────────────────────────────
RETRIEVER_K   = 4
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 250

# ── Supported upload types ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS = ["pdf", "txt", "md", "csv"]

# ── Streamlit page ───────────────────────────────────────────────────────
def configure_page():
    st.set_page_config(
        page_title="AMS AI Assistant",
        page_icon="🤖",
        layout="wide",
    )


def render_header():
    st.markdown(
        "<h2 style='margin-bottom:0'>🤖 AMS AI Assistant</h2>"
        "<p style='color:grey;margin-top:2px;font-size:0.85rem'>"
        "<p Powered by Microsoft Agent Framework · Semantic Kernel · Gemini </p>"
        "<p></p>"
        "<hr/>",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        "<hr/><p style='text-align:center;color:grey;font-size:0.75rem'>"
        "AMS AI Chat Assistant</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# RAG System Prompt
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
You are an intelligent knowledge assistant for an organisation.

Conversation History:
{chat_history}

RULES:
1. First check the Context below (retrieved from uploaded documents).
2. If the context contains relevant information about the question topic:
   - Use the context as your primary source.
   - You MAY also use your own knowledge to explain, elaborate, give
     examples, or clarify concepts — but ONLY about topics that are
     present in the context.
   - Always make clear what came from the documents vs your explanation.
   - Cite the source document using the [Source: filename] tags.
   - Break down complex terms, walk through concepts step by step,
     and give examples to make the answer easy to understand.
3. If the context is empty OR the question is completely unrelated to
   the topics in the context, respond with:
   "I could not find information about this in the uploaded documents.
    Please check if the relevant document has been uploaded."
4. NEVER invent company-specific facts (names, figures, policies, dates)
   that are not in the context.
5. For follow-up questions like "explain more", "why", "give an example",
   use the conversation history to stay on topic and elaborate.

Context (retrieved from uploaded documents):
{context}

Question:
{question}

Answer:
"""


# ══════════════════════════════════════════════════════════════════════════
# Spell Check Prompt
# ══════════════════════════════════════════════════════════════════════════
SPELL_CHECK_PROMPT = """
You are a spell checker. Correct any spelling mistakes in the message below.
Return ONLY the corrected message — no explanation, no commentary.
If there are no errors, return the message exactly as-is.

Message:
{message}

Corrected message:
"""


# ══════════════════════════════════════════════════════════════════════════
# Document Brief Prompt
# ══════════════════════════════════════════════════════════════════════════
BRIEF_PROMPT = """
You are analyzing a newly ingested document for a knowledge base.

Based ONLY on the content below, produce a brief with these sections:

SUMMARY:
A 3-4 sentence plain-language summary.

KEY TOPICS:
4-6 bullet points of the main topics covered.

SUGGESTED QUESTIONS:
3-4 example questions a user could ask about this document.

Be concise. Do not invent information.

Document content:
{content}
"""