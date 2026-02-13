"""Graph operation builders — convenience API for constructing G²CP operations.

Provides a fluent builder interface for constructing TRAVERSE and UPDATE
operations, used by LLM-based operation selection (Algorithm 2 in paper).
"""

from __future__ import annotations

from g2cp.protocol.messages import (
    GraphDelta,
    NodeSelector,
    ReturnFormat,
    TraversalOperation,
    UpdateOperation,
)


class TraversalBuilder:
    """Fluent builder for TRAVERSE operations.

    Usage:
        op = (TraversalBuilder()
              .from_nodes(["Symptom:grinding_1200RPM", "Symptom:temp_85C"])
              .via(["causes", "indicates"])
              .depth(2)
              .return_subgraph()
              .build())
    """

    def __init__(self) -> None:
        self._source = NodeSelector()
        self._edge_types: list[str] = []
        self._depth: int = 1
        self._return_format: ReturnFormat = ReturnFormat.SUBGRAPH
        self._constraints: str | None = None

    def from_nodes(self, node_ids: list[str]) -> TraversalBuilder:
        """Set source nodes by explicit IDs."""
        self._source = NodeSelector(explicit_ids=node_ids)
        return self

    def from_type(self, *types: str) -> TraversalBuilder:
        """Set source nodes by type filter."""
        self._source = NodeSelector(type_filter=list(types))
        return self

    def from_property(self, filter_expr: str) -> TraversalBuilder:
        """Set source nodes by property filter (e.g., 'Symptom WHERE severity>0.8')."""
        self._source = NodeSelector(property_filter=filter_expr)
        return self

    def from_context(self, ref: str = "CURRENT_FOCUS") -> TraversalBuilder:
        """Set source nodes from conversation context."""
        self._source = NodeSelector(context_ref=ref)
        return self

    def via(self, edge_types: list[str]) -> TraversalBuilder:
        """Set edge type filter."""
        self._edge_types = edge_types
        return self

    def depth(self, h: int) -> TraversalBuilder:
        """Set hop depth."""
        self._depth = h
        return self

    def return_subgraph(self) -> TraversalBuilder:
        self._return_format = ReturnFormat.SUBGRAPH
        return self

    def return_paths(self) -> TraversalBuilder:
        self._return_format = ReturnFormat.PATHS
        return self

    def return_leaves(self) -> TraversalBuilder:
        self._return_format = ReturnFormat.LEAVES
        return self

    def with_constraints(self, constraints: str) -> TraversalBuilder:
        self._constraints = constraints
        return self

    def build(self) -> TraversalOperation:
        return TraversalOperation(
            source=self._source,
            edge_types=self._edge_types,
            depth=self._depth,
            return_format=self._return_format,
            constraints=self._constraints,
        )


class UpdateBuilder:
    """Fluent builder for UPDATE operations.

    Usage:
        op = (UpdateBuilder()
              .add_edge("Part:X", "Sensor:temp_anomaly", "risk_indicator", confidence=0.89)
              .build())
    """

    def __init__(self) -> None:
        self._delta = GraphDelta()

    def add_node(
        self, uid: str, node_type: str, attributes: dict | None = None, **kwargs: object,
    ) -> UpdateBuilder:
        node = {"uid": uid, "type": node_type, **(attributes or {}), **kwargs}
        self._delta.add_nodes.append(node)
        return self

    def remove_node(self, uid: str) -> UpdateBuilder:
        self._delta.remove_nodes.append(uid)
        return self

    def add_edge(
        self,
        from_uid: str,
        to_uid: str,
        edge_type: str,
        weight: float = 1.0,
        **properties: object,
    ) -> UpdateBuilder:
        edge = {
            "from": from_uid,
            "to": to_uid,
            "type": edge_type,
            "weight": weight,
            **properties,
        }
        self._delta.add_edges.append(edge)
        return self

    def remove_edge(self, edge_id: str) -> UpdateBuilder:
        self._delta.remove_edges.append(edge_id)
        return self

    def build(self) -> UpdateOperation:
        return UpdateOperation(delta=self._delta)


# ── Predefined edge type sets per agent role ──────────────────────────────

DIAGNOSTIC_EDGES = {"causes", "indicates", "correlates_with"}
PROCEDURAL_EDGES = {"addressed_by", "requires", "precedes", "has_safety_protocol"}
SYNTHESIS_EDGES = {"occurred_in", "replaced_in", "failed_after"}
ALL_EDGES = DIAGNOSTIC_EDGES | PROCEDURAL_EDGES | SYNTHESIS_EDGES

# Mapping from intent category to edge types (used in LLM-based selection)
INTENT_EDGE_MAP: dict[str, list[str]] = {
    "diagnostic": sorted(DIAGNOSTIC_EDGES),
    "procedural": sorted(PROCEDURAL_EDGES),
    "predictive": sorted(SYNTHESIS_EDGES),
    "factoid": sorted(DIAGNOSTIC_EDGES | PROCEDURAL_EDGES),
}
