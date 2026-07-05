"""
gemini_adapter.py

Gemini Adapter for Microsoft Agent Framework.

This adapter acts as the bridge between
Microsoft Agent Framework and our existing
Gemini LLM connector.
"""

from __future__ import annotations

from llm import generate


class GeminiAdapter:
    """
    Gemini Adapter.

    Responsible for sending prompts
    to Gemini and returning responses.
    """

    async def chat(
        self,
        message: str,
    ) -> str:
        """
        Send a message to Gemini.
        """

        response = await generate(message)

        return response