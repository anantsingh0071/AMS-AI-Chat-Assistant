"""
llm.py
─────────────────────────────────────────────────────────────────────────
Single LLM connector. Mode is controlled by LLM_MODE in .env:

  LLM_MODE=gemini   → Google Gemini  (testing)
  LLM_MODE=azure    → Azure / Phi-3  (production)

No swapping files, no import tricks — just set LLM_MODE in .env.
"""

from __future__ import annotations


import asyncio
import os
from dotenv import load_dotenv
from config import (
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MODE,
)

load_dotenv()

# LLM_MODE    = os.getenv("LLM_MODE", "gemini").lower()   # default: gemini for easy testing
# TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))


# ══════════════════════════════════════════════════════════════════════════
# Gemini
# ══════════════════════════════════════════════════════════════════════════

def _gemini_generate(prompt: str) -> str:
    import sys
    import traceback
    import google.generativeai as genai

    print("Python Executable:", sys.executable)
    print("Python Path:", sys.path)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=api_key)

    model_name = GEMINI_MODEL

    cfg = genai.GenerationConfig(
        temperature=LLM_TEMPERATURE,
        max_output_tokens=2048,
    )

    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt,
            generation_config=cfg,
        )

        if hasattr(response, "text"):
            return response.text

        raise RuntimeError("Gemini returned an empty response.")

    except Exception as e:
        print("\n========== GEMINI ERROR ==========")
        traceback.print_exc()
        print("Exception:", e)
        print("==================================")
        raise
    
# ══════════════════════════════════════════════════════════════════════════
# Azure / Phi-3
# ══════════════════════════════════════════════════════════════════════════
def _azure_generate(prompt: str) -> str:
    from openai import AzureOpenAI

    api_key   = os.getenv("AZURE_OPENAI_API_KEY", "") or os.getenv("AZURE_AI_API_KEY", "")
    endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT", "") or os.getenv("AZURE_AI_ENDPOINT", "")
    api_ver   = os.getenv("AZURE_API_VERSION", "2024-02-01")
    model     = os.getenv("AZURE_MODEL", "Phi-3-medium-128k-instruct")

    if not api_key or not endpoint:
        raise EnvironmentError(
            "Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT in .env for Azure mode."
        )

    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_ver)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # temperature=TEMPERATURE,
        temperature=LLM_TEMPERATURE,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


# ══════════════════════════════════════════════════════════════════════════
# Public async interface — used by all SK plugins
# ══════════════════════════════════════════════════════════════════════════
async def generate(prompt: str) -> str:
    """
    Async generate — dispatches to Gemini or Azure based on LLM_MODE.
    Runs the synchronous SDK call in a thread pool executor.
    """
    loop = asyncio.get_event_loop()

    if LLM_MODE == "azure":
        return await loop.run_in_executor(None, _azure_generate, prompt)
    else:
        return await loop.run_in_executor(None, _gemini_generate, prompt)