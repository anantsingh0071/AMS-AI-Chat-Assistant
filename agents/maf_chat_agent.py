"""
maf_chat_agent.py

Microsoft Agent Framework Chat Agent.

This is the only agent that app.py will use.

Testing:
    Gemini

Production:
    Azure OpenAI

The rest of the application never changes.
"""

from __future__ import annotations

from agent_framework._agents import Agent

from agents.base_agent import ChatAssistantAgent
from agents.gemini_adapter import GeminiAdapter

from rag_engine import (
    spell_check,
    get_answer,
)


class MAFChatAgent(ChatAssistantAgent):
    """
    Microsoft Agent Framework Chat Agent.
    """

    def __init__(self, kernel):
        super().__init__()

        self.kernel = kernel

        # Gemini adapter (used during development)
        self.client = GeminiAdapter()

        # Create the Microsoft Agent Framework Agent
        self.agent = Agent(
            client=self.client,
            name="AMS Chat Assistant",
            instructions=(
                "You are an enterprise AI assistant. "
                "Always answer using the company knowledge base."
            ),
        )

    def run(
        self,
        query: str,
        chat_history=None,
    ) -> str:
        """
        Execute the assistant.

        Currently:
            Uses the existing Semantic Kernel + RAG pipeline.

        Future:
            This method will invoke the Microsoft Agent Framework
            Agent directly when Azure/OpenAI is integrated.
        """

        # Step 1 — Spell Check
        corrected_query = spell_check(
            self.kernel,
            query,
        )

        # Step 2 — Retrieve + Generate Answer
        answer = get_answer(
            self.kernel,
            corrected_query,
            chat_history,
        )

        return answer