"""Node resolution pipeline for G²CP.

Implements Algorithm 2 Steps 1-2: entity extraction from natural language
queries and linking to knowledge graph node UIDs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from g2cp.protocol.messages import NodeSelector
from g2cp.protocol.operations import INTENT_EDGE_MAP
from g2cp.utils.embeddings import EntityLinker
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class NodeResolver:
    """Resolves natural language mentions to graph node selectors.

    Implements the full query-to-operation translation pipeline
    (Algorithm 3 in the paper, Section 5.5.1).
    """

    def __init__(self, llm: LLMClient, linker: EntityLinker) -> None:
        self.llm = llm
        self.linker = linker

    def resolve_query(self, query: str) -> dict[str, Any]:
        """Full pipeline: query → entities → linked nodes → intent → operation params.

        Returns:
            Dict with keys: source_uids, intent, edge_types, depth, raw_entities
        """
        # Step 1: Entity extraction via LLM
        raw_entities = self.llm.extract_entities(query)
        logger.info(f"Extracted entities: {raw_entities}")

        # Step 2: Entity linking to graph nodes
        source_uids = []
        for entity in raw_entities:
            uid = self.linker.link(entity)
            if uid:
                source_uids.append(uid)
                logger.info(f"  Linked '{entity}' → {uid}")
            else:
                logger.warning(f"  Failed to link '{entity}'")

        # Step 3: Intent classification
        intent = self.llm.classify_intent(query)
        logger.info(f"Intent: {intent}")

        # Step 4: Edge type selection
        edge_types = INTENT_EDGE_MAP.get(intent, ["causes", "indicates"])

        # Step 5: Depth estimation
        depth = self.llm.estimate_depth(query, len(source_uids), len(edge_types))
        logger.info(f"Estimated depth: {depth}")

        return {
            "source_uids": source_uids,
            "intent": intent,
            "edge_types": edge_types,
            "depth": depth,
            "raw_entities": raw_entities,
        }

    def resolve_selector(
        self, selector: NodeSelector, context: dict[str, Any] | None = None
    ) -> list[str]:
        """Resolve a NodeSelector to concrete UIDs.

        Priority system:
        1. Explicit IDs → direct return
        2. Type filter → query graph
        3. Property filter → query graph
        4. Context ref → resolve from conversation state
        """
        if selector.explicit_ids:
            return selector.explicit_ids

        if selector.type_filter:
            # Would query the graph here; return type as placeholder
            return [f"type:{t}" for t in selector.type_filter]

        if selector.context_ref and context:
            focus = context.get("focus_nodes", [])
            return focus

        return []
