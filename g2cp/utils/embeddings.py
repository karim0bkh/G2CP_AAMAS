"""Embedding-based entity linking for G²CP.

Implements Algorithm 2 Step 2: Entity Linking to Graph Nodes.
Uses sentence embeddings to fuzzy-match extracted entity mentions
to knowledge graph node UIDs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EntityLinker:
    """Links natural language entity mentions to knowledge graph node UIDs.

    Uses cosine similarity between sentence embeddings of mentions
    and node descriptions. Falls back to string matching when the
    sentence-transformers model is unavailable.
    """

    def __init__(self, threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.threshold = threshold
        self._model = None
        self._node_index: dict[str, np.ndarray] = {}  # uid -> embedding
        self._node_texts: dict[str, str] = {}  # uid -> text description

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; using string-based matching"
            )

    def build_index(self, nodes: list[dict[str, Any]]) -> int:
        """Build the entity index from knowledge graph nodes.

        Args:
            nodes: List of node dicts with 'uid' and optionally 'name', 'description'.

        Returns:
            Number of nodes indexed.
        """
        self._node_texts.clear()
        self._node_index.clear()

        for node in nodes:
            uid = node.get("uid", "")
            text = node.get("name", uid)
            if "description" in node:
                text += " " + node["description"]
            self._node_texts[uid] = text.lower()

        if self._model is not None:
            texts = list(self._node_texts.values())
            uids = list(self._node_texts.keys())
            if texts:
                embeddings = self._model.encode(texts, show_progress_bar=False)
                for uid, emb in zip(uids, embeddings):
                    self._node_index[uid] = emb

        return len(self._node_texts)

    def link(self, mention: str) -> Optional[str]:
        """Link a text mention to a knowledge graph node UID.

        Args:
            mention: The entity mention text (e.g., "bearing B-4521").

        Returns:
            The matched node UID, or None if no match above threshold.
        """
        mention_lower = mention.lower().strip()

        # Try exact match first
        for uid, text in self._node_texts.items():
            if mention_lower == text or mention_lower in uid.lower():
                return uid

        # Try embedding-based matching
        if self._model is not None and self._node_index:
            return self._embedding_match(mention)

        # Fallback: substring matching
        return self._string_match(mention_lower)

    def link_batch(self, mentions: list[str]) -> list[Optional[str]]:
        """Link multiple mentions to graph nodes."""
        return [self.link(m) for m in mentions]

    def _embedding_match(self, mention: str) -> Optional[str]:
        """Match using cosine similarity of sentence embeddings."""
        mention_emb = self._model.encode([mention], show_progress_bar=False)[0]

        best_uid = None
        best_score = -1.0

        for uid, node_emb in self._node_index.items():
            score = float(np.dot(mention_emb, node_emb) / (
                np.linalg.norm(mention_emb) * np.linalg.norm(node_emb) + 1e-8
            ))
            if score > best_score:
                best_score = score
                best_uid = uid

        if best_score >= self.threshold:
            return best_uid
        return None

    def _string_match(self, mention: str) -> Optional[str]:
        """Fallback string-based matching using normalized overlap."""
        best_uid = None
        best_score = 0.0

        mention_tokens = set(mention.replace("_", " ").replace("-", " ").split())

        for uid, text in self._node_texts.items():
            text_tokens = set(text.replace("_", " ").replace("-", " ").split())
            if not mention_tokens or not text_tokens:
                continue
            overlap = len(mention_tokens & text_tokens) / max(
                len(mention_tokens), len(text_tokens)
            )

            # Boost exact substring matches
            if mention in text or text in mention:
                overlap = max(overlap, 0.9)

            # Boost UID matches
            if mention.replace(" ", "_") in uid.lower() or mention.replace(" ", "") in uid.lower():
                overlap = max(overlap, 0.95)

            if overlap > best_score:
                best_score = overlap
                best_uid = uid

        if best_score >= self.threshold:
            return best_uid
        return None
