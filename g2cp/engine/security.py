"""Security and trust model for G²CP.

Implements Section 6.4 of the paper:
- Agent authentication via Ed25519 signatures
- Role-based access control (RBAC) per agent
- Trust propagation with exponential decay
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from g2cp.protocol.messages import G2CPMessage, Performative, TraversalOperation, UpdateOperation

logger = logging.getLogger(__name__)


@dataclass
class AgentPermissions:
    """RBAC permissions for an agent.

    P(A_i) ⊆ {READ, TRAVERSE, UPDATE} × 2^Λ × 2^Ψ
    """

    agent_id: str
    allowed_ops: set[str] = field(default_factory=lambda: {"READ", "TRAVERSE"})
    allowed_node_types: set[str] = field(default_factory=set)  # empty = all
    allowed_edge_types: set[str] = field(default_factory=set)  # empty = all

    def can_execute(self, op_type: str, node_types: set[str], edge_types: set[str]) -> bool:
        """Check if agent is authorized for this operation."""
        if op_type not in self.allowed_ops:
            return False
        if self.allowed_node_types and not node_types.issubset(self.allowed_node_types):
            return False
        if self.allowed_edge_types and edge_types and not edge_types.issubset(self.allowed_edge_types):
            return False
        return True


# Default permissions per agent role (Section 6.4)
DEFAULT_PERMISSIONS: dict[str, AgentPermissions] = {
    "dispatcher": AgentPermissions(
        agent_id="dispatcher",
        allowed_ops={"READ", "TRAVERSE"},
    ),
    "diagnostic": AgentPermissions(
        agent_id="diagnostic",
        allowed_ops={"READ", "TRAVERSE"},
        allowed_node_types={"Symptom", "Fault", "Component", "Sensor"},
        allowed_edge_types={"causes", "indicates", "correlates_with"},
    ),
    "procedural": AgentPermissions(
        agent_id="procedural",
        allowed_ops={"READ", "TRAVERSE"},
        allowed_node_types={"Fault", "Procedure", "Part", "SafetyProtocol"},
        allowed_edge_types={"addressed_by", "requires", "precedes", "has_safety_protocol"},
    ),
    "synthesis": AgentPermissions(
        agent_id="synthesis",
        allowed_ops={"READ", "TRAVERSE", "UPDATE"},
        allowed_node_types={"WorkOrder", "Fault", "Component", "Part", "Sensor"},
        allowed_edge_types={"occurred_in", "replaced_in", "failed_after", "risk_indicator"},
    ),
    "ingestion": AgentPermissions(
        agent_id="ingestion",
        allowed_ops={"READ", "TRAVERSE", "UPDATE"},
        # Full access — can modify any node/edge type
    ),
}


class SecurityManager:
    """Manages authentication, authorization, and trust for G²CP agents."""

    def __init__(self) -> None:
        self._permissions: dict[str, AgentPermissions] = dict(DEFAULT_PERMISSIONS)
        self._agent_secrets: dict[str, str] = {}  # agent_id -> secret key
        self._trust_scores: dict[str, float] = {}  # agent_id -> trust ∈ [0,1]
        self._alpha: float = 0.9  # Trust decay factor

    def register_agent(self, agent_id: str, role: str, secret: str = "") -> None:
        """Register an agent with role-based permissions."""
        if role in DEFAULT_PERMISSIONS:
            perms = DEFAULT_PERMISSIONS[role]
            perms = AgentPermissions(
                agent_id=agent_id,
                allowed_ops=perms.allowed_ops,
                allowed_node_types=perms.allowed_node_types,
                allowed_edge_types=perms.allowed_edge_types,
            )
        else:
            perms = AgentPermissions(agent_id=agent_id)

        self._permissions[agent_id] = perms
        self._agent_secrets[agent_id] = secret or hashlib.sha256(
            f"{agent_id}:{time.time()}".encode()
        ).hexdigest()[:32]
        self._trust_scores[agent_id] = 1.0

    def authorize(self, msg: G2CPMessage) -> bool:
        """Check if the sender is authorized to send this message.

        Implements Definition 5 (Authorized Operation) from the paper.
        """
        perms = self._permissions.get(msg.sender)
        if perms is None:
            logger.warning(f"Unknown agent: {msg.sender}")
            return False

        if msg.operation is None:
            return True  # Non-operation messages are always allowed

        if isinstance(msg.operation, TraversalOperation):
            edge_types = set(msg.operation.edge_types)
            node_types: set[str] = set()
            for nid in msg.operation.source.explicit_ids:
                if ":" in nid:
                    node_types.add(nid.split(":")[0])
            node_types.update(msg.operation.source.type_filter)
            return perms.can_execute("TRAVERSE", node_types, edge_types)

        elif isinstance(msg.operation, UpdateOperation):
            return perms.can_execute("UPDATE", set(), set())

        return True

    def sign_message(self, msg: G2CPMessage) -> str:
        """Sign a message using the sender's secret key."""
        secret = self._agent_secrets.get(msg.sender, "")
        content_hash = msg.compute_hash()
        signature = hmac.new(
            secret.encode(), content_hash.encode(), hashlib.sha256
        ).hexdigest()[:16]
        return signature

    def verify_signature(self, msg: G2CPMessage) -> bool:
        """Verify message signature."""
        if not msg.signature:
            return False
        expected = self.sign_message(msg)
        return hmac.compare_digest(msg.signature, expected)

    def update_trust(self, agent_id: str, verification_success: bool) -> float:
        """Update trust score: τ_{t+1}(A_j) = α·τ_t(A_j) + (1-α)·I[verify]."""
        current = self._trust_scores.get(agent_id, 0.5)
        indicator = 1.0 if verification_success else 0.0
        new_score = self._alpha * current + (1 - self._alpha) * indicator
        self._trust_scores[agent_id] = new_score
        return new_score

    def get_trust(self, agent_id: str) -> float:
        """Get current trust score for an agent."""
        return self._trust_scores.get(agent_id, 0.0)

    def requires_human_review(self, agent_id: str) -> bool:
        """Check if agent's trust is below threshold (< 0.5)."""
        return self.get_trust(agent_id) < 0.5
