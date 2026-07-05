"""
agent_manager.py

Returns the active chat agent.

Current:
    Microsoft Agent Framework + Gemini

Future:
    Microsoft Agent Framework + Azure OpenAI

No other file in the project needs to know
which LLM is being used.
"""

from agents.maf_chat_agent import MAFChatAgent


def get_chat_agent(kernel):
    """
    Return the active chat agent.
    """
    return MAFChatAgent(kernel)