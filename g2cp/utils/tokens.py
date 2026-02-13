"""Token counting utilities for G²CP evaluation.

Uses tiktoken cl100k_base tokenizer (consistent with GPT-4) to count
inter-agent message tokens. Implements the methodology described in
Section 6.2.1 of the paper.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TokenCounter:
    """Counts tokens using tiktoken cl100k_base (GPT-4 tokenizer).

    We count ALL inter-agent messages (the content exchanged between agents),
    excluding: (1) internal LLM reasoning, (2) original user query, (3) final response.
    This isolates the communication overhead that G²CP specifically targets.
    """

    def __init__(self) -> None:
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed; using word-based approximation")

    def count(self, text: str) -> int:
        """Count tokens in a text string."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: rough approximation (1 token ≈ 0.75 words)
        return int(len(text.split()) / 0.75)

    def count_message(self, message: Any) -> int:
        """Count tokens in a G²CP or baseline message."""
        from g2cp.protocol.messages import G2CPMessage

        if isinstance(message, G2CPMessage):
            return self.count(message.serialize())
        elif isinstance(message, str):
            return self.count(message)
        elif isinstance(message, dict):
            import json
            return self.count(json.dumps(message))
        return 0

    def count_conversation(self, messages: list[Any]) -> dict[str, int]:
        """Count tokens across a conversation of messages.

        Returns:
            Dict with total tokens, per-message breakdown, and stats.
        """
        counts = [self.count_message(m) for m in messages]
        return {
            "total": sum(counts),
            "per_message": counts,
            "mean": sum(counts) / max(len(counts), 1),
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "n_messages": len(counts),
        }
