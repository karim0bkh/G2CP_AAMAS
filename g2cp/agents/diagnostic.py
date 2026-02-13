"""Diagnostic Agent — performs root cause analysis via symptom-to-fault traversals.

Specializes in edge types: {causes, indicates, correlates_with}.
Ranks faults by path convergence (number of symptoms pointing to each fault).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from g2cp.agents.base import BaseG2CPAgent
from g2cp.protocol.messages import (
    G2CPMessage,
    Performative,
    SubgraphResult,
    TraversalOperation,
)
from g2cp.protocol.operations import DIAGNOSTIC_EDGES, TraversalBuilder
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class DiagnosticAgent(BaseG2CPAgent):
    """Diagnostic agent for symptom → fault root cause analysis."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        super().__init__(
            agent_id="diagnostic",
            role="diagnostic",
            db=db,
            llm=llm,
            specialized_edge_types=DIAGNOSTIC_EDGES,
        )

    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle REQUEST messages for diagnostic analysis.

        Executes the traversal, ranks faults by convergence, and returns
        INFORM message with ranked results.
        """
        if msg.performative != Performative.REQUEST:
            return {"messages": []}

        # Execute the traversal if not already done
        result = prior_result
        if result is None and isinstance(msg.operation, TraversalOperation):
            result = self.execute_operation(msg.operation)

        if result is None or result.is_empty:
            error_msg = self.create_message(
                receiver=msg.sender,
                performative=Performative.INFORM,
                result=SubgraphResult(metadata={"warning": "No faults found"}),
                context=msg.context,
            )
            return {"messages": [error_msg]}

        # Rank faults by path convergence
        ranked_result = self._rank_faults(result)

        # Create INFORM response
        inform_msg = self.create_inform(
            receiver=msg.sender,
            result=ranked_result,
            context=msg.context,
        )

        return {"messages": [inform_msg], "result": ranked_result}

    def _rank_faults(self, result: SubgraphResult) -> SubgraphResult:
        """Rank faults by convergence — how many symptoms point to each fault."""
        fault_counts: Counter = Counter()
        fault_edges: dict[str, list[dict]] = {}

        for edge in result.edges:
            target = edge.get("to", "")
            # Count edges pointing to fault nodes
            for node in result.nodes:
                if node.get("uid") == target and node.get("type") in ("Fault", "fault"):
                    fault_counts[target] += 1
                    fault_edges.setdefault(target, []).append(edge)

        # Sort nodes: faults first (by convergence), then other nodes
        fault_uids = set(fault_counts.keys())
        fault_nodes = sorted(
            [n for n in result.nodes if n.get("uid") in fault_uids],
            key=lambda n: fault_counts.get(n.get("uid", ""), 0),
            reverse=True,
        )
        other_nodes = [n for n in result.nodes if n.get("uid") not in fault_uids]

        # Add confidence scores based on convergence
        max_count = max(fault_counts.values()) if fault_counts else 1
        for node in fault_nodes:
            uid = node.get("uid", "")
            count = fault_counts.get(uid, 0)
            node["confidence"] = round(count / max_count, 2)
            node["supporting_paths"] = len(fault_edges.get(uid, []))

        return SubgraphResult(
            nodes=fault_nodes + other_nodes,
            edges=result.edges,
            paths=result.paths,
            metadata={
                **result.metadata,
                "ranked_faults": [
                    {"uid": uid, "convergence": count}
                    for uid, count in fault_counts.most_common()
                ],
            },
        )
