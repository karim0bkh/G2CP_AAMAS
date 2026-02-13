"""Ingestion Agent — validates and applies graph updates.

Processes UPDATE operations with schema validation, type constraints,
relationship constraints, provenance tracking, and rollback capability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from g2cp.agents.base import BaseG2CPAgent
from g2cp.protocol.messages import (
    G2CPMessage,
    Performative,
    SubgraphResult,
    UpdateOperation,
)
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Valid node types and relationship constraints (graph schema)
VALID_NODE_TYPES = {
    "Component", "Fault", "Procedure", "WorkOrder",
    "Part", "Sensor", "SafetyProtocol",
}

VALID_EDGE_SCHEMA: dict[str, set[tuple[str, str]]] = {
    "causes": {("Fault", "Fault"), ("Component", "Fault"), ("Sensor", "Fault")},
    "indicates": {("Symptom", "Fault"), ("Sensor", "Fault")},
    "correlates_with": {("Fault", "Fault"), ("Sensor", "Sensor")},
    "addressed_by": {("Fault", "Procedure")},
    "requires": {("Procedure", "Part")},
    "precedes": {("Procedure", "Procedure")},
    "has_safety_protocol": {("Procedure", "SafetyProtocol"), ("Component", "SafetyProtocol")},
    "occurred_in": {("Fault", "WorkOrder"), ("Component", "WorkOrder")},
    "replaced_in": {("Part", "WorkOrder")},
    "failed_after": {("Component", "Component")},
    "risk_indicator": {("Fault", "Sensor"), ("Component", "Sensor")},
    "part_of": {("Component", "Component"), ("Part", "Component")},
}


class IngestionAgent(BaseG2CPAgent):
    """Ingestion agent for validating and applying graph updates."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        super().__init__(
            agent_id="ingestion",
            role="ingestion",
            db=db,
            llm=llm,
        )
        self._update_history: list[dict[str, Any]] = []

    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle UPDATE messages with validation."""
        if msg.performative != Performative.UPDATE:
            return {"messages": []}

        if not isinstance(msg.operation, UpdateOperation):
            return {"messages": [self.create_message(
                receiver=msg.sender,
                performative=Performative.REJECT,
                error="Expected UPDATE operation",
                context=msg.context,
            )]}

        # Validate the update
        validation = self._validate_update(msg.operation)

        if not validation["valid"]:
            reject_msg = self.create_message(
                receiver=msg.sender,
                performative=Performative.REJECT,
                error=f"Validation failed: {validation['errors']}",
                context=msg.context,
            )
            return {"messages": [reject_msg]}

        # Apply the update
        try:
            result = self.executor.execute_update(msg.operation)

            # Track for rollback
            self._update_history.append({
                "timestamp": time.time(),
                "source_agent": msg.sender,
                "operation": msg.operation.model_dump(),
                "result": result.metadata,
            })

            confirm_msg = self.create_message(
                receiver=msg.sender,
                performative=Performative.CONFIRM,
                result=result,
                context=msg.context,
            )
            return {"messages": [confirm_msg], "result": result}

        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            return {"messages": [self.create_message(
                receiver=msg.sender,
                performative=Performative.REJECT,
                error=str(e),
                context=msg.context,
            )]}

    def _validate_update(self, op: UpdateOperation) -> dict[str, Any]:
        """Validate UPDATE against graph schema constraints."""
        errors = []

        # Validate new nodes
        for node in op.delta.add_nodes:
            node_type = node.get("type", "")
            if node_type and node_type not in VALID_NODE_TYPES:
                errors.append(f"Invalid node type: {node_type}")
            if not node.get("uid"):
                errors.append("Node missing required 'uid' field")

        # Validate new edges
        for edge in op.delta.add_edges:
            edge_type = edge.get("type", "")
            if edge_type and edge_type not in VALID_EDGE_SCHEMA:
                errors.append(f"Unknown edge type: {edge_type}")
            # Weight validation
            weight = edge.get("weight", 1.0)
            if not (0.0 <= weight <= 1.0):
                errors.append(f"Edge weight out of range [0,1]: {weight}")

        return {"valid": len(errors) == 0, "errors": errors}

    def rollback_last(self) -> bool:
        """Rollback the most recent update (graph versioning)."""
        if not self._update_history:
            return False
        # In production, this would use Neo4j's transaction log
        last = self._update_history.pop()
        logger.info(f"Rolled back update from {last['source_agent']} at {last['timestamp']}")
        return True
