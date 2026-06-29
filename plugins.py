"""
plugins.py
─────────────────────────────────────────────────────────────────────────
Semantic Kernel plugins for the AMS AI Chat Assistant.
"""

from __future__ import annotations

from semantic_kernel.functions import kernel_function

from config import (
    SYSTEM_PROMPT,
    BRIEF_PROMPT,
    SPELL_CHECK_PROMPT,
    RETRIEVER_K,
)

from llm import generate


# ══════════════════════════════════════════════════════════════════════════
# Spell Check Plugin
# ══════════════════════════════════════════════════════════════════════════
class SpellCheckPlugin:
    @kernel_function(
        name="correct",
        description="Correct spelling mistakes in a user message.",
    )
    async def correct(self, message: str) -> str:
        prompt = SPELL_CHECK_PROMPT.format(message=message)
        result = await generate(prompt)
        return result.strip() or message


# ══════════════════════════════════════════════════════════════════════════
# Retrieval Plugin
# ══════════════════════════════════════════════════════════════════════════
class RetrievalPlugin:
    def __init__(self, vectorstore):
        self._vs = vectorstore

    @kernel_function(
        name="retrieve",
        description="Retrieve relevant document chunks from the knowledge base.",
    )
    def retrieve(self, query: str, k: int = RETRIEVER_K) -> str:
        if self._vs is None:
            return ""

        results = self._vs.search(query, k=k)

        if not results:
            return ""

        return "\n\n---\n\n".join(
            f"[Source: {r.source}]\n{r.chunk}"
            for r in results
        )

    def get_sources(self, query: str, k: int = RETRIEVER_K) -> set[str]:
        if self._vs is None:
            return set()

        return {
            r.source
            for r in self._vs.search(query, k=k)
        }


# ══════════════════════════════════════════════════════════════════════════
# Answer Plugin
# ══════════════════════════════════════════════════════════════════════════
class AnswerPlugin:

    @kernel_function(
        name="answer",
        description="Generate the final answer using retrieved context.",
    )
    async def answer(
        self,
        question: str,
        context: str,
        chat_history: str = "",
    ) -> str:

        prompt = SYSTEM_PROMPT.format(
            chat_history=chat_history,
            context=context,
            question=question,
        )

        return await generate(prompt)


# ══════════════════════════════════════════════════════════════════════════
# Brief Plugin
# ══════════════════════════════════════════════════════════════════════════
class BriefPlugin:

    @kernel_function(
        name="generate_brief",
        description="Generate a knowledge brief for an ingested document.",
    )
    async def generate_brief(self, content: str) -> str:

        prompt = BRIEF_PROMPT.format(
            content=content[:12000]
        )

        return await generate(prompt)