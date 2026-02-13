"""Dispatcher Agent — routes user queries to specialist agents via G²CP.

Implements the query decomposition pipeline (Algorithm 2/3 in the paper):
1. Entity extraction via LLM
2. Entity linking to graph nodes
3. Intent classification
4. Edge type selection
5. Depth estimation
6. G²CP message construction
"""

from __future__ import annotations

import logging
from typing import Any

from g2cp.agents.base import BaseG2CPAgent
from g2cp.engine.resolver import NodeResolver
from g2cp.protocol.messages import (
    G2CPMessage,
    MessageContext,
    Performative,
    SubgraphResult,
)
from g2cp.protocol.operations import INTENT_EDGE_MAP, TraversalBuilder
from g2cp.utils.embeddings import EntityLinker
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Intent → primary agent mapping
INTENT_AGENT_MAP = {
    "diagnostic": "diagnostic",
    "procedural": "procedural",
    "predictive": "synthesis",
    "factoid": "diagnostic",  # Factoid queries default to diagnostic
}


class DispatcherAgent(BaseG2CPAgent):
    """Dispatcher agent: decomposes user queries into G²CP operations.

    The Dispatcher is the ONLY agent that interfaces with natural language.
    It converts user queries into structured G²CP messages and routes
    them to specialist agents. It also aggregates results and generates
    the final response.
    """

    def __init__(self, db: GraphDB, llm: LLMClient, linker: EntityLinker) -> None:
        super().__init__(
            agent_id="dispatcher",
            role="dispatcher",
            db=db,
            llm=llm,
        )
        self.resolver = NodeResolver(llm, linker)
        self.linker = linker

    def handle_query(self, query: str, context: MessageContext | None = None) -> dict[str, Any]:
        """Process a user query and generate G²CP messages.

        Implements Algorithm 3 (Query-to-G²CP Translation).

        Args:
            query: Natural language user query.
            context: Optional conversation context.

        Returns:
            Dict with 'messages' (list of G²CP messages to dispatch).
        """
        ctx = context or MessageContext()

        # Full resolution pipeline
        resolution = self.resolver.resolve_query(query)

        source_uids = resolution["source_uids"]
        intent = resolution["intent"]
        edge_types = resolution["edge_types"]
        depth = resolution["depth"]

        # Select target agent
        target_agent = INTENT_AGENT_MAP.get(intent, "diagnostic")

        # Build primary traversal operation
        builder = TraversalBuilder()
        if source_uids:
            builder.from_nodes(source_uids)
        else:
            # Fallback: search by type
            builder.from_type("Symptom", "Fault", "Component")
        builder.via(edge_types).depth(depth).return_subgraph()
        primary_op = builder.build()

        # Create primary REQUEST message
        primary_msg = self.create_request(
            receiver=target_agent,
            operation=primary_op,
            context=ctx,
        )

        messages = [primary_msg]

        # For diagnostic queries, also request procedural follow-up
        if intent == "diagnostic":
            proc_op = (
                TraversalBuilder()
                .from_context("CURRENT_FOCUS")
                .via(["addressed_by", "requires_part", "has_safety_protocol"])
                .depth(1)
                .return_subgraph()
                .build()
            )
            proc_msg = self.create_request(
                receiver="procedural",
                operation=proc_op,
                context=ctx,
            )
            messages.append(proc_msg)

        return {"messages": messages, "resolution": resolution}

    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle incoming INFORM messages from specialist agents."""
        # Collect results for response generation
        return {"messages": [], "result": prior_result}

    def generate_response(
        self, query: str, nodes: list[dict], edges: list[dict]
    ) -> str:
        """Generate final NL response from collected graph data."""
        # Format subgraph data for LLM
        data_lines = []
        for node in nodes[:20]:
            uid = node.get("uid", "")
            name = node.get("name", uid)
            ntype = node.get("type", "Entity")
            data_lines.append(f"[{ntype}] {name} (id: {uid})")

        for edge in edges[:20]:
            from_uid = edge.get("from", "")
            to_uid = edge.get("to", "")
            etype = edge.get("type", "RELATED")
            data_lines.append(f"{from_uid} --[{etype}]--> {to_uid}")

        subgraph_text = "\n".join(data_lines) if data_lines else "No data found."

        return self.llm.generate_response(query, subgraph_text)
