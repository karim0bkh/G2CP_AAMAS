"""Neo4j connection manager for G²CP knowledge graph operations.

Handles connection lifecycle, Cypher query execution, and graph schema validation.
Supports both real Neo4j connections and an in-memory mock for testing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GraphDB:
    """Neo4j graph database interface.

    Manages connections and provides query execution with parameterized Cypher.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database
        self._driver = None

        if password:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(uri, auth=(user, password))
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {uri}")
            except ImportError:
                logger.warning("neo4j package not installed; using in-memory mock")
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}; using in-memory mock")

        if self._driver is None:
            self._mock_store = InMemoryGraphStore()

    def execute(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        params = parameters or {}

        if self._driver is not None:
            return self._execute_neo4j(cypher, params)
        return self._mock_store.execute(cypher, params)

    def _execute_neo4j(self, cypher: str, params: dict[str, Any]) -> list[dict]:
        """Execute against real Neo4j."""
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            return [dict(record) for record in result]

    def load_nodes(self, nodes: list[dict[str, Any]]) -> int:
        """Bulk load nodes into the graph."""
        count = 0
        for node in nodes:
            node_type = node.get("type", "Entity")
            uid = node.get("uid", "")
            props = {k: v for k, v in node.items() if k not in ("type",)}
            cypher = f"MERGE (n:{node_type} {{uid: $uid}}) SET n += $props"
            self.execute(cypher, {"uid": uid, "props": props})
            count += 1
        return count

    def load_edges(self, edges: list[dict[str, Any]]) -> int:
        """Bulk load edges into the graph."""
        count = 0
        for edge in edges:
            from_uid = edge.get("from", "")
            to_uid = edge.get("to", "")
            edge_type = edge.get("type", "RELATED_TO")
            props = {k: v for k, v in edge.items() if k not in ("from", "to", "type")}
            cypher = (
                f"MATCH (a {{uid: $from_uid}}), (b {{uid: $to_uid}}) "
                f"MERGE (a)-[r:{edge_type}]->(b) SET r += $props"
            )
            self.execute(cypher, {"from_uid": from_uid, "to_uid": to_uid, "props": props})
            count += 1
        return count

    def get_schema(self) -> dict[str, Any]:
        """Retrieve graph schema (node types, edge types, counts)."""
        node_types = self.execute(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        edge_types = self.execute(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types"
        )
        counts = self.execute(
            "MATCH (n) RETURN count(n) AS node_count"
        )
        edge_counts = self.execute(
            "MATCH ()-[r]->() RETURN count(r) AS edge_count"
        )
        return {
            "node_types": node_types[0].get("labels", []) if node_types else [],
            "edge_types": edge_types[0].get("types", []) if edge_types else [],
            "node_count": counts[0].get("node_count", 0) if counts else 0,
            "edge_count": edge_counts[0].get("edge_count", 0) if edge_counts else 0,
        }

    def clear(self) -> None:
        """Delete all nodes and edges."""
        self.execute("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            self._driver.close()


class InMemoryGraphStore:
    """In-memory graph store for testing without Neo4j.

    Stores nodes and edges in dictionaries and supports basic Cypher-like
    operations for traversal and matching.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}  # uid -> {type, props...}
        self.edges: list[dict[str, Any]] = []  # [{from, to, type, props...}]

    def execute(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Execute a simplified Cypher-like query against in-memory store."""
        params = params or {}
        cypher_upper = cypher.upper().strip()

        # Handle MERGE for nodes
        if "MERGE" in cypher_upper and "uid" in params:
            uid = params.get("uid", "")
            props = params.get("props", {})
            if uid:
                if uid not in self.nodes:
                    self.nodes[uid] = {}
                self.nodes[uid].update(props)
            return []

        # Handle MATCH for edges
        if "MERGE" in cypher_upper and "from_uid" in params:
            edge = {
                "from": params["from_uid"],
                "to": params["to_uid"],
                **params.get("props", {}),
            }
            self.edges.append(edge)
            return []

        # Handle traversal pattern: MATCH path = (start)-[r*1..N]->(end)
        if "path" in cypher.lower() and "node_set" in str(params):
            return self._traverse(params)

        # Handle node lookup
        if "uid" in str(params) and "MATCH" in cypher_upper:
            uid = params.get("uid") or params.get("node_uid", "")
            if uid in self.nodes:
                return [{"n": {**self.nodes[uid], "uid": uid}}]
            return []

        # Handle schema queries
        if "db.labels" in cypher:
            types = set()
            for n in self.nodes.values():
                if "type" in n:
                    types.add(n["type"])
            return [{"labels": list(types)}]

        if "db.relationshipTypes" in cypher or "relationshipType" in cypher:
            types = set()
            for e in self.edges:
                if "type" in e:
                    types.add(e["type"])
            return [{"types": list(types)}]

        if "count(n)" in cypher:
            return [{"node_count": len(self.nodes)}]

        if "count(r)" in cypher:
            return [{"edge_count": len(self.edges)}]

        if "DETACH DELETE" in cypher_upper:
            self.nodes.clear()
            self.edges.clear()
            return []

        return []

    def _traverse(self, params: dict[str, Any]) -> list[dict]:
        """Execute a graph traversal query."""
        node_set = params.get("node_set", [])
        edge_filter = params.get("edge_filter", [])
        depth = params.get("depth", 1)

        visited: set[str] = set()
        frontier = set(node_set)
        result_nodes: list[dict] = []
        result_edges: list[dict] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                if nid in self.nodes:
                    result_nodes.append({"uid": nid, **self.nodes[nid]})
                for edge in self.edges:
                    if edge.get("from") == nid:
                        if not edge_filter or edge.get("type") in edge_filter:
                            result_edges.append(edge)
                            next_frontier.add(edge.get("to", ""))
            frontier = next_frontier - visited

        # Add remaining frontier nodes
        for nid in frontier:
            if nid in self.nodes and nid not in visited:
                result_nodes.append({"uid": nid, **self.nodes[nid]})

        return [{"nodes": result_nodes, "edges": result_edges}]

    def find_neighbors(
        self, node_uid: str, edge_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Find direct neighbors of a node, optionally filtered by edge type."""
        neighbors = []
        for edge in self.edges:
            if edge.get("from") == node_uid:
                if edge_types is None or edge.get("type") in edge_types:
                    to_uid = edge.get("to", "")
                    if to_uid in self.nodes:
                        neighbors.append({
                            "node": {"uid": to_uid, **self.nodes[to_uid]},
                            "edge": edge,
                        })
        return neighbors
