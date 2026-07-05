"""
azure_chat_agent.py

Azure Chat Assistant Agent

Current:
    Uses the existing Semantic Kernel + RAG pipeline.

Future:
    Will use Microsoft Agent Framework with Azure Phi-3.
"""

from __future__ import annotations

from agents.base_agent import ChatAssistantAgent

from rag_engine import (
    get_answer,
    spell_check,
)


class AzureChatAgent(ChatAssistantAgent):
    """
    Azure Chat Assistant Agent.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    def run(
        self,
        query: str,
        chat_history=None,
    ) -> str:
        """
        Execute the Azure Chat Assistant.
        """

        # Step 1 - Spell Check
        corrected_query = spell_check(
            self.kernel,
            query,
        )

        # Step 2 - Generate Answer
        answer = get_answer(
            self.kernel,
            corrected_query,
            chat_history,
        )

        return answer