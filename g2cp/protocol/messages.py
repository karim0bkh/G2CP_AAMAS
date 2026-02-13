"""G²CP Message types and performatives.

Defines the core data structures for G²CP communication:
- Performatives (REQUEST, INFORM, QUERY, PROPOSE, CONFIRM, REJECT, UPDATE)
- G2CPMessage tuple: <sender, receiver, performative, operation, context>
- MessageContext for conversation tracking
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Performative(str, Enum):
    """G²CP performatives defining illocutionary force of messages."""

    REQUEST = "REQUEST"    # Sender asks receiver to execute op and return results
    INFORM = "INFORM"      # Sender asserts that op has been executed with results
    QUERY = "QUERY"        # Sender asks whether op would return non-empty results
    PROPOSE = "PROPOSE"    # Sender suggests op as candidate for consideration
    CONFIRM = "CONFIRM"    # Sender validates receiver's previous operation result
    REJECT = "REJECT"      # Sender indicates receiver's operation violated constraints
    UPDATE = "UPDATE"      # Sender commands modification to G via op


class ReturnFormat(str, Enum):
    """Return format for traversal operations."""

    SUBGRAPH = "SUBGRAPH"
    PATHS = "PATHS"
    LEAVES = "LEAVES"


class NodeSelector(BaseModel):
    """Selects nodes from the knowledge graph.

    Supports four selection modes:
    - explicit_ids: Direct node ID references (e.g., {"Part:B-4521"})
    - type_filter: Match by node type (e.g., {"type:Fault"})
    - property_filter: Match by property (e.g., {"Symptom WHERE severity>0.8"})
    - context_ref: Reference conversation context (e.g., "CURRENT_FOCUS")
    """

    explicit_ids: list[str] = Field(default_factory=list)
    type_filter: list[str] = Field(default_factory=list)
    property_filter: Optional[str] = None
    context_ref: Optional[str] = None

    def to_cypher_clause(self) -> str:
        """Convert node selector to a Cypher WHERE clause."""
        clauses = []
        if self.explicit_ids:
            id_list = ", ".join(f'"{nid}"' for nid in self.explicit_ids)
            clauses.append(f"n.uid IN [{id_list}]")
        if self.type_filter:
            labels = " OR ".join(f"n:{t}" for t in self.type_filter)
            clauses.append(f"({labels})")
        if self.property_filter:
            clauses.append(self._parse_property_filter())
        return " AND ".join(clauses) if clauses else "TRUE"

    def _parse_property_filter(self) -> str:
        if not self.property_filter:
            return "TRUE"
        # Parse "NodeType WHERE property op value"
        parts = self.property_filter.split(" WHERE ")
        if len(parts) == 2:
            return f"n:{parts[0]} AND n.{parts[1]}"
        return f"n.{self.property_filter}"

    @property
    def is_empty(self) -> bool:
        return (
            not self.explicit_ids
            and not self.type_filter
            and not self.property_filter
            and not self.context_ref
        )


class TraversalOperation(BaseModel):
    """TRAVERSE operation: walk the graph from source nodes along filtered edges.

    TRAVERSE(V_s, Ψ_f, h, ret) where:
    - V_s: source node set (NodeSelector)
    - Ψ_f: edge type filter
    - h: hop depth
    - ret: return format
    """

    source: NodeSelector
    edge_types: list[str] = Field(default_factory=list)
    depth: int = 1
    return_format: ReturnFormat = ReturnFormat.SUBGRAPH
    constraints: Optional[str] = None

    @property
    def op_type(self) -> str:
        return "TRAVERSE"


class GraphDelta(BaseModel):
    """Delta for graph updates: nodes/edges to add or remove."""

    add_nodes: list[dict[str, Any]] = Field(default_factory=list)
    remove_nodes: list[str] = Field(default_factory=list)
    add_edges: list[dict[str, Any]] = Field(default_factory=list)
    remove_edges: list[str] = Field(default_factory=list)


class UpdateOperation(BaseModel):
    """UPDATE operation: modify the knowledge graph.

    UPDATE(ΔG) where ΔG = (ΔV+, ΔV−, ΔE+, ΔE−)
    """

    delta: GraphDelta

    @property
    def op_type(self) -> str:
        return "UPDATE"


# Union type for operations
GraphOperation = TraversalOperation | UpdateOperation


class SubgraphResult(BaseModel):
    """Result of a graph operation — a subgraph fragment."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.nodes and not self.edges and not self.paths


class MessageContext(BaseModel):
    """Conversation context attached to every G²CP message."""

    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    focus_subgraph: Optional[SubgraphResult] = None
    parent_message_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class G2CPMessage(BaseModel):
    """Core G²CP message: <sender, receiver, performative, operation, context>.

    This is the fundamental unit of inter-agent communication in G²CP.
    All fields are explicit, deterministic, and verifiable.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    performative: Performative
    operation: Optional[GraphOperation] = None
    result: Optional[SubgraphResult] = None
    context: MessageContext = Field(default_factory=MessageContext)
    error: Optional[str] = None
    signature: Optional[str] = None

    @property
    def is_traversal(self) -> bool:
        return isinstance(self.operation, TraversalOperation)

    @property
    def is_update(self) -> bool:
        return isinstance(self.operation, UpdateOperation)

    def compute_hash(self) -> str:
        """Compute deterministic hash for integrity verification."""
        content = (
            f"{self.sender}:{self.receiver}:{self.performative.value}"
            f":{self.operation.model_dump_json() if self.operation else 'none'}"
            f":{self.context.conversation_id}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def serialize(self) -> str:
        """Serialize to G²CP text format."""
        lines = [
            f"{self.sender} TO {self.receiver}",
            f"PERFORMATIVE: {self.performative.value}",
            f"CONVERSATION: {self.context.conversation_id}",
        ]
        if self.operation:
            if isinstance(self.operation, TraversalOperation):
                lines.append("OPERATION:")
                lines.append("  TRAVERSE")
                if self.operation.source.explicit_ids:
                    ids = ", ".join(self.operation.source.explicit_ids)
                    lines.append(f"    FROM: {{{ids}}}")
                elif self.operation.source.type_filter:
                    types = ", ".join(f"type:{t}" for t in self.operation.source.type_filter)
                    lines.append(f"    FROM: {{{types}}}")
                elif self.operation.source.property_filter:
                    lines.append(f"    FROM: {{{self.operation.source.property_filter}}}")
                if self.operation.edge_types:
                    edges = ", ".join(self.operation.edge_types)
                    lines.append(f"    VIA: {{{edges}}}")
                lines.append(f"    DEPTH: {self.operation.depth}")
                lines.append(f"    RETURN: {self.operation.return_format.value}")
                if self.operation.constraints:
                    lines.append(f"    CONSTRAINTS: {self.operation.constraints}")
            elif isinstance(self.operation, UpdateOperation):
                lines.append("OPERATION:")
                lines.append("  UPDATE APPLY")
                for edge in self.operation.delta.add_edges:
                    lines.append(f"    ADD_EDGE: {edge}")
                for node in self.operation.delta.add_nodes:
                    lines.append(f"    ADD_NODE: {node}")
                for eid in self.operation.delta.remove_edges:
                    lines.append(f"    REMOVE_EDGE: {eid}")

        if self.result:
            lines.append("RESULT:")
            if self.result.nodes:
                node_strs = [n.get("uid", str(n)) for n in self.result.nodes]
                lines.append(f"  NODES: {{{', '.join(node_strs)}}}")
            if self.result.edges:
                lines.append(f"  EDGES: [{len(self.result.edges)} edges]")
            if self.result.paths:
                for p in self.result.paths[:3]:
                    lines.append(f"  PATH: {' -> '.join(p)}")

        if self.error:
            lines.append(f"ERROR: {self.error}")

        return "\n".join(lines)


# ── Priority system for message queuing ──────────────────────────────────

MESSAGE_PRIORITY: dict[Performative, int] = {
    Performative.REQUEST: 1,   # Highest — diagnostic-critical
    Performative.INFORM: 2,
    Performative.QUERY: 3,
    Performative.CONFIRM: 3,
    Performative.REJECT: 2,
    Performative.PROPOSE: 4,
    Performative.UPDATE: 4,    # Lowest — background knowledge synthesis
}
