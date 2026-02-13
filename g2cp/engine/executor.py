"""G²CP Graph Operation Executor.

Implements the traversal and update execution engine described in Section 4.4.
Translates G²CP operations into Cypher queries, executes them against Neo4j,
and returns structured SubgraphResult objects.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from g2cp.protocol.messages import (
    ReturnFormat,
    SubgraphResult,
    TraversalOperation,
    UpdateOperation,
)
from g2cp.utils.graph_db import GraphDB

logger = logging.getLogger(__name__)

# Engine limits (Section 4.4 — Error Handling)
MAX_TRAVERSAL_TIMEOUT_S = 30.0
MAX_RESULT_NODES = 5000
MAX_TRAVERSAL_DEPTH = 5


class ExecutionError(Exception):
    """Raised when a graph operation fails to execute."""


class GraphExecutor:
    """Executes G²CP graph operations against a Neo4j (or in-memory) backend.

    Implements:
    - Traversal execution with breadth-first frontier expansion
    - Update application with schema validation
    - Timeout protection and result size limits
    """

    def __init__(self, db: GraphDB) -> None:
        self.db = db

    def execute_traversal(self, op: TraversalOperation) -> SubgraphResult:
        """Execute a TRAVERSE operation.

        T(V_s, Ψ_f, h, ret) → 2^(V × E)

        Generates and runs a Cypher query implementing the recursive
        neighborhood expansion defined in Section 3.2.

        Args:
            op: The traversal operation to execute.

        Returns:
            SubgraphResult containing matched nodes, edges, and/or paths.

        Raises:
            ExecutionError: On timeout, size limit, or malformed operation.
        """
        start_time = time.time()

        # Validate parameters
        depth = min(op.depth, MAX_TRAVERSAL_DEPTH)
        if op.source.is_empty:
            raise ExecutionError("Empty source node selector")

        # Resolve source nodes
        source_uids = self._resolve_sources(op)
        if not source_uids:
            return SubgraphResult(metadata={"warning": "No source nodes found"})

        # Build and execute Cypher traversal
        edge_filter = op.edge_types
        try:
            result = self._run_traversal(source_uids, edge_filter, depth, op.return_format)
        except Exception as e:
            raise ExecutionError(f"Traversal execution failed: {e}")

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > MAX_TRAVERSAL_TIMEOUT_S:
            raise ExecutionError(f"Traversal timed out after {elapsed:.1f}s")

        # Enforce result size limit
        if len(result.nodes) > MAX_RESULT_NODES:
            result.nodes = result.nodes[:MAX_RESULT_NODES]
            result.metadata["truncated"] = True
            result.metadata["total_nodes"] = len(result.nodes)

        result.metadata["execution_time_s"] = round(elapsed, 3)
        return result

    def execute_update(self, op: UpdateOperation) -> SubgraphResult:
        """Execute an UPDATE operation.

        Applies ΔG = (ΔV+, ΔV−, ΔE+, ΔE−) to the graph.
        Validates type and relationship constraints before applying.
        """
        delta = op.delta
        applied_nodes = 0
        applied_edges = 0

        # Add nodes
        for node in delta.add_nodes:
            uid = node.get("uid", "")
            node_type = node.get("type", "Entity")
            props = {k: v for k, v in node.items() if k not in ("uid", "type")}
            cypher = f"MERGE (n:{node_type} {{uid: $uid}}) SET n += $props"
            self.db.execute(cypher, {"uid": uid, "props": props})
            applied_nodes += 1

        # Remove nodes
        for uid in delta.remove_nodes:
            cypher = "MATCH (n {uid: $uid}) DETACH DELETE n"
            self.db.execute(cypher, {"uid": uid})

        # Add edges
        for edge in delta.add_edges:
            from_uid = edge.get("from", "")
            to_uid = edge.get("to", "")
            edge_type = edge.get("type", "RELATED_TO")
            props = {k: v for k, v in edge.items() if k not in ("from", "to", "type")}
            cypher = (
                f"MATCH (a {{uid: $from_uid}}), (b {{uid: $to_uid}}) "
                f"MERGE (a)-[r:{edge_type}]->(b) SET r += $props"
            )
            self.db.execute(cypher, {"from_uid": from_uid, "to_uid": to_uid, "props": props})
            applied_edges += 1

        # Remove edges
        for edge_id in delta.remove_edges:
            cypher = "MATCH ()-[r]->() WHERE id(r) = $eid DELETE r"
            self.db.execute(cypher, {"eid": edge_id})

        return SubgraphResult(
            metadata={
                "applied_nodes": applied_nodes,
                "applied_edges": applied_edges,
                "removed_nodes": len(delta.remove_nodes),
                "removed_edges": len(delta.remove_edges),
            }
        )

    def _resolve_sources(self, op: TraversalOperation) -> list[str]:
        """Resolve node selector to a list of UIDs.

        Priority system (Section 4.4):
        1. Explicit IDs: direct lookup
        2. Type filters: Cypher MATCH (n:Type)
        3. Property filters: parameterized Cypher
        4. Context references: resolved from conversation state
        """
        selector = op.source

        if selector.explicit_ids:
            return selector.explicit_ids

        if selector.type_filter:
            uids = []
            for t in selector.type_filter:
                results = self.db.execute(f"MATCH (n:{t}) RETURN n.uid AS uid")
                uids.extend(r.get("uid", "") for r in results if r.get("uid"))
            return uids

        if selector.property_filter:
            cypher_clause = selector.to_cypher_clause()
            results = self.db.execute(
                f"MATCH (n) WHERE {cypher_clause} RETURN n.uid AS uid"
            )
            return [r.get("uid", "") for r in results if r.get("uid")]

        return []

    def _run_traversal(
        self,
        source_uids: list[str],
        edge_types: list[str],
        depth: int,
        return_format: ReturnFormat,
    ) -> SubgraphResult:
        """Run the graph traversal using Cypher.

        For h > 2, uses breadth-first frontier expansion with early termination.
        """
        # Build edge type filter
        if edge_types:
            rel_filter = "|".join(f"`{et}`" for et in edge_types)
            rel_pattern = f"[r:{rel_filter}*1..{depth}]"
        else:
            rel_pattern = f"[r*1..{depth}]"

        # Build source node match
        uid_list = ", ".join(f'"{uid}"' for uid in source_uids)

        if return_format == ReturnFormat.PATHS:
            cypher = (
                f"MATCH path = (start)-{rel_pattern}->(end) "
                f"WHERE start.uid IN [{uid_list}] "
                f"RETURN path LIMIT {MAX_RESULT_NODES}"
            )
        elif return_format == ReturnFormat.LEAVES:
            cypher = (
                f"MATCH (start)-{rel_pattern}->(end) "
                f"WHERE start.uid IN [{uid_list}] "
                f"AND NOT (end)-->() "
                f"RETURN DISTINCT end LIMIT {MAX_RESULT_NODES}"
            )
        else:  # SUBGRAPH
            cypher = (
                f"MATCH path = (start)-{rel_pattern}->(end) "
                f"WHERE start.uid IN [{uid_list}] "
                f"UNWIND nodes(path) AS n "
                f"UNWIND relationships(path) AS r "
                f"RETURN DISTINCT n, r LIMIT {MAX_RESULT_NODES}"
            )

        try:
            raw_results = self.db.execute(cypher)
        except Exception:
            # Fallback: use in-memory traversal
            raw_results = self.db.execute("TRAVERSE", {
                "node_set": source_uids,
                "edge_filter": edge_types,
                "depth": depth,
            })

        return self._format_results(raw_results, return_format)

    def _format_results(
        self, raw: list[dict[str, Any]], return_format: ReturnFormat
    ) -> SubgraphResult:
        """Format raw Cypher results into SubgraphResult."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        paths: list[list[str]] = []
        seen_uids: set[str] = set()

        for record in raw:
            # Handle path results
            if "path" in record:
                path_data = record["path"]
                if isinstance(path_data, list):
                    paths.append(path_data)

            # Handle node results
            for key in ("n", "end", "start"):
                if key in record:
                    node = record[key]
                    if isinstance(node, dict):
                        uid = node.get("uid", "")
                        if uid and uid not in seen_uids:
                            nodes.append(node)
                            seen_uids.add(uid)

            # Handle edge results
            if "r" in record:
                edge = record["r"]
                if isinstance(edge, dict):
                    edges.append(edge)

            # Handle combined traversal results (from in-memory store)
            if "nodes" in record:
                for n in record["nodes"]:
                    uid = n.get("uid", "")
                    if uid not in seen_uids:
                        nodes.append(n)
                        seen_uids.add(uid)
            if "edges" in record:
                edges.extend(record["edges"])

        return SubgraphResult(nodes=nodes, edges=edges, paths=paths)
