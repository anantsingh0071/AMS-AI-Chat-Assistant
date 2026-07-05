"""
base_agent.py

Base class for all Chat Assistant Agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ChatAssistantAgent(ABC):
    """
    Base Chat Assistant Agent.

    Every agent must implement the run() method.
    """

    @abstractmethod
    def run(
        self,
        query: str,
        chat_history=None,
    ) -> str:
        """
        Execute the agent.

        Parameters
        ----------
        query : str
            User question.

        chat_history : list | None
            Previous conversation history.

        Returns
        -------
        str
            Final response.
        """
        pass