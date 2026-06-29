"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────
RAG pipeline coordinator. Used by app.py and sidebar.py.
"""

from __future__ import annotations

import asyncio

from vector_store import VectorStore, chunk_text, load_uploaded_file
# from llm import generate
from kernel import build_kernel
from semantic_kernel.functions import KernelArguments
from config import (
    RETRIEVER_K,
    SYSTEM_PROMPT,
    SPELL_CHECK_PROMPT,
    BRIEF_PROMPT,
)


# ── Async → sync bridge ───────────────────────────────────────────────────
def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()

        return loop.run_until_complete(coro)

    except RuntimeError:
        return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════
# Ingestion
# ══════════════════════════════════════════════════════════════════════════
def ingest_files(uploaded_files, existing_vectorstore=None):
    """
    Load → Chunk → Embed → Index

    Returns:
        vectorstore,
        rag_chain (VectorStore during Gemini testing),
        file_meta,
        per_file_text
    """

    vs = existing_vectorstore or VectorStore()
    file_meta = []
    per_file_text = {}

    for uf in uploaded_files:
        text = load_uploaded_file(uf)

        if not text.strip():
            continue

        chunks = chunk_text(text)
        vs.add(chunks, source=uf.name)

        file_meta.append({"name": uf.name})
        per_file_text[uf.name] = text

    if not file_meta:
        raise ValueError("No readable content found in the uploaded files.")

    kernel = build_kernel(vs)

    return vs, kernel, file_meta, per_file_text


# ══════════════════════════════════════════════════════════════════════════
# Spell Check
# ══════════════════════════════════════════════════════════════════════════
def spell_check(kernel, text: str) -> str:
    try:
        plugin = kernel.get_plugin("spellcheck")
        function = plugin["correct"]

        result = _run(
            kernel.invoke(
                function,
                arguments=KernelArguments(
                    message=text
                ),
            )
        )

        corrected = str(result).strip()
        return corrected if corrected else text

    except Exception:
        return text

# ══════════════════════════════════════════════════════════════════════════
# Question Answering
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# Question Answering
# ══════════════════════════════════════════════════════════════════════════
def get_answer(kernel, query: str, chat_history=None) -> str:

    if kernel is None:
        return (
            "No documents uploaded yet. "
            "Please upload documents in the sidebar first."
        )

    history_str = ""

    if chat_history:
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in chat_history
        )

    # Step 1 - Retrieve Context
    retrieval_plugin = kernel.get_plugin("retrieval")
    retrieval_function = retrieval_plugin["retrieve"]

    context = _run(
        kernel.invoke(
            retrieval_function,
            arguments=KernelArguments(
                query=query,
            ),
        )
    )

    # Step 2 - Generate Answer
    answer_plugin = kernel.get_plugin("answer")
    answer_function = answer_plugin["answer"]

    result = _run(
        kernel.invoke(
            answer_function,
            arguments=KernelArguments(
                question=query,
                context=str(context),
                chat_history=history_str,
            ),
        )
    )

    return str(result)
# ══════════════════════════════════════════════════════════════════════════
# Source Verification
# ══════════════════════════════════════════════════════════════════════════
def get_retrieved_sources(vectorstore, query: str) -> set[str]:
    if vectorstore is None:
        return set()

    return {
        r.source
        for r in vectorstore.search(query, k=RETRIEVER_K)
    }


def verify_references(
    answer: str,
    retrieved_sources: set[str],
    known_files: set[str],
) -> str | None:

    mentioned = {
        f
        for f in known_files
        if f in answer
    }

    unverified = mentioned - retrieved_sources

    if unverified:
        return (
            "⚠️ Reference check: the answer mentions "
            + ", ".join(f"'{f}'" for f in unverified)
            + ", but that file was not part of the retrieved context."
        )

    return None


# ══════════════════════════════════════════════════════════════════════════
# Document Brief
# ══════════════════════════════════════════════════════════════════════════
import traceback

def generate_doc_brief(full_text: str) -> str:
    """
    Generate a brief for uploaded documents.
    """

    kernel = build_kernel()

    plugin = kernel.get_plugin("brief")
    function = plugin["generate_brief"]

    try:
        result = _run(
            kernel.invoke(
                function,
                arguments=KernelArguments(
                    content=full_text[:12000]
                ),
            )
        )

    except Exception as e:
        print("\n========== INVOKE ERROR ==========")
        traceback.print_exc()

        if hasattr(e, "__cause__") and e.__cause__:
            print("\n========== ROOT CAUSE ==========")
            traceback.print_exception(
                type(e.__cause__),
                e.__cause__,
                e.__cause__.__traceback__,
            )
            print("================================")

        raise

    if hasattr(result, "value") and result.value:
        value = result.value[0]

        if hasattr(value, "text"):
            return value.text

        return str(value)

    return str(result)