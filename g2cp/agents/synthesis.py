"""Synthesis Agent — discovers patterns through temporal graph traversal.

Specializes in: {occurred_in, replaced_in, failed_after}.
Discovers new relationships via co-occurrence frequency analysis over
historical work order data (Section 4.1).
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
from g2cp.protocol.operations import SYNTHESIS_EDGES, TraversalBuilder, UpdateBuilder
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)

CO_OCCURRENCE_THRESHOLD = 0.6  # Minimum co-occurrence ratio to propose new edge


class SynthesisAgent(BaseG2CPAgent):
    """Synthesis agent for historical pattern analysis and knowledge discovery."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        super().__init__(
            agent_id="synthesis",
            role="synthesis",
            db=db,
            llm=llm,
            specialized_edge_types=SYNTHESIS_EDGES,
        )

    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle QUERY and REQUEST messages for pattern analysis."""
        result = prior_result
        if result is None and isinstance(msg.operation, TraversalOperation):
            result = self.execute_operation(msg.operation)

        if msg.performative == Performative.QUERY:
            # Answer existence question
            exists = result is not None and not result.is_empty
            response = self.create_message(
                receiver=msg.sender,
                performative=Performative.CONFIRM if exists else Performative.REJECT,
                result=result,
                context=msg.context,
            )
            return {"messages": [response], "result": result}

        elif msg.performative == Performative.REQUEST:
            if result is None or result.is_empty:
                return {"messages": [self.create_inform(
                    receiver=msg.sender,
                    result=SubgraphResult(metadata={"warning": "No historical data found"}),
                    context=msg.context,
                )]}

            # Analyze co-occurrences
            patterns = self._analyze_cooccurrences(result)
            result.metadata["patterns"] = patterns

            # Propose new edges for significant patterns
            update_messages = self._propose_new_edges(patterns, msg)

            messages = [
                self.create_inform(receiver=msg.sender, result=result, context=msg.context)
            ]
            messages.extend(update_messages)

            return {"messages": messages, "result": result}

        return {"messages": []}

    def _analyze_cooccurrences(self, result: SubgraphResult) -> list[dict[str, Any]]:
        """Analyze co-occurrence patterns in historical work order data.

        For each pair (fault, condition), counts how often they co-occur
        in work orders. Returns significant patterns.
        """
        # Group nodes by type
        work_orders = [n for n in result.nodes if n.get("type") in ("WorkOrder", "work_order")]
        faults = [n for n in result.nodes if n.get("type") in ("Fault", "fault")]
        conditions = [n for n in result.nodes if n.get("type") in ("Sensor", "sensor", "Condition")]

        # Count co-occurrences (simplified: based on shared edges)
        fault_condition_pairs: Counter = Counter()
        fault_totals: Counter = Counter()

        for edge in result.edges:
            from_uid = edge.get("from", "")
            to_uid = edge.get("to", "")
            # Count when faults and conditions appear in same work order context
            if from_uid in {f.get("uid") for f in faults}:
                fault_totals[from_uid] += 1
                if to_uid in {c.get("uid") for c in conditions}:
                    fault_condition_pairs[(from_uid, to_uid)] += 1

        # Compute co-occurrence ratios
        patterns = []
        for (fault_uid, cond_uid), count in fault_condition_pairs.most_common():
            total = fault_totals.get(fault_uid, 1)
            ratio = count / total
            if ratio >= CO_OCCURRENCE_THRESHOLD:
                patterns.append({
                    "fault": fault_uid,
                    "condition": cond_uid,
                    "co_occurrences": count,
                    "total_occurrences": total,
                    "confidence": round(ratio, 3),
                })

        return patterns

    def _propose_new_edges(
        self, patterns: list[dict[str, Any]], original_msg: G2CPMessage
    ) -> list[G2CPMessage]:
        """Propose UPDATE operations for significant co-occurrence patterns."""
        messages = []

        for pattern in patterns:
            if pattern["confidence"] >= CO_OCCURRENCE_THRESHOLD:
                update_op = (
                    UpdateBuilder()
                    .add_edge(
                        from_uid=pattern["fault"],
                        to_uid=pattern["condition"],
                        edge_type="risk_indicator",
                        weight=pattern["confidence"],
                        confidence=pattern["confidence"],
                        source="synthesis_agent",
                        co_occurrences=pattern["co_occurrences"],
                    )
                    .build()
                )
                msg = self.create_message(
                    receiver="ingestion",
                    performative=Performative.UPDATE,
                    operation=update_op,
                    context=original_msg.context,
                )
                messages.append(msg)

        return messages
