"""
kernel.py
─────────────────────────────────────────────────────────────────────────
Builds the Semantic Kernel and registers all plugins.
"""

from __future__ import annotations

import semantic_kernel as sk

from plugins import (
    RetrievalPlugin,
    AnswerPlugin,
    BriefPlugin,
    SpellCheckPlugin,
)


def build_kernel(vectorstore=None) -> sk.Kernel:
    """Build Semantic Kernel with all plugins."""

    kernel = sk.Kernel()

    kernel.add_plugin(
        RetrievalPlugin(vectorstore),
        plugin_name="retrieval",
    )

    kernel.add_plugin(
        AnswerPlugin(),
        plugin_name="answer",
    )

    kernel.add_plugin(
        SpellCheckPlugin(),
        plugin_name="spellcheck",
    )

    kernel.add_plugin(
        BriefPlugin(),
        plugin_name="brief",
    )

    return kernel