"""Procedural Agent — retrieves maintenance procedures for diagnosed faults.

Specializes in edge types: {addressed_by, requires, precedes, has_safety_protocol}.
Always includes safety protocols in returned subgraphs.
"""

from __future__ import annotations

import logging
from typing import Any

from g2cp.agents.base import BaseG2CPAgent
from g2cp.protocol.messages import (
    G2CPMessage,
    Performative,
    SubgraphResult,
    TraversalOperation,
)
from g2cp.protocol.operations import PROCEDURAL_EDGES, TraversalBuilder
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class ProceduralAgent(BaseG2CPAgent):
    """Procedural agent for fault → action retrieval."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        super().__init__(
            agent_id="procedural",
            role="procedural",
            db=db,
            llm=llm,
            specialized_edge_types=PROCEDURAL_EDGES,
        )

    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle REQUEST messages for procedure retrieval."""
        if msg.performative != Performative.REQUEST:
            return {"messages": []}

        result = prior_result
        if result is None and isinstance(msg.operation, TraversalOperation):
            result = self.execute_operation(msg.operation)

        if result is None or result.is_empty:
            return {"messages": [self.create_message(
                receiver=msg.sender,
                performative=Performative.INFORM,
                result=SubgraphResult(metadata={"warning": "No procedures found"}),
                context=msg.context,
            )]}

        # Ensure safety protocols are included
        result = self._enrich_with_safety(result)

        # Optionally query Synthesis agent for historical frequency
        synthesis_query = self._build_synthesis_query(result, msg)

        messages = [
            self.create_inform(receiver=msg.sender, result=result, context=msg.context)
        ]
        if synthesis_query:
            messages.append(synthesis_query)

        return {"messages": messages, "result": result}

    def _enrich_with_safety(self, result: SubgraphResult) -> SubgraphResult:
        """Ensure safety protocol nodes are included for all procedures."""
        procedure_uids = [
            n.get("uid", "")
            for n in result.nodes
            if n.get("type") in ("Procedure", "procedure")
        ]

        if not procedure_uids:
            return result

        # Query for safety protocols linked to found procedures
        for proc_uid in procedure_uids:
            safety_op = (
                TraversalBuilder()
                .from_nodes([proc_uid])
                .via(["has_safety_protocol"])
                .depth(1)
                .return_subgraph()
                .build()
            )
            try:
                safety_result = self.execute_operation(safety_op)
                # Merge safety nodes/edges
                existing_uids = {n.get("uid") for n in result.nodes}
                for node in safety_result.nodes:
                    if node.get("uid") not in existing_uids:
                        result.nodes.append(node)
                        existing_uids.add(node.get("uid"))
                result.edges.extend(safety_result.edges)
            except Exception as e:
                logger.warning(f"Safety protocol lookup failed for {proc_uid}: {e}")

        return result

    def _build_synthesis_query(
        self, result: SubgraphResult, original_msg: G2CPMessage
    ) -> G2CPMessage | None:
        """Optionally build a QUERY to Synthesis agent for historical data."""
        fault_uids = [
            n.get("uid", "")
            for n in result.nodes
            if n.get("type") in ("Fault", "fault")
        ]
        if not fault_uids:
            return None

        op = (
            TraversalBuilder()
            .from_nodes(fault_uids)
            .via(["occurred_in"])
            .depth(2)
            .return_leaves()
            .build()
        )
        return self.create_message(
            receiver="synthesis",
            performative=Performative.QUERY,
            operation=op,
            context=original_msg.context,
        )
