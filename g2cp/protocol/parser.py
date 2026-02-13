"""G²CP message parser — converts text-format messages to structured objects.

Implements Stage 1 of Algorithm 1 (Message Parsing and Execution) from the paper.
Supports parsing of both serialized G²CP text format and programmatic construction.
"""

from __future__ import annotations

import re
from typing import Optional

from g2cp.protocol.messages import (
    G2CPMessage,
    GraphDelta,
    MessageContext,
    NodeSelector,
    Performative,
    ReturnFormat,
    SubgraphResult,
    TraversalOperation,
    UpdateOperation,
)


class G2CPParseError(Exception):
    """Raised when a G²CP message cannot be parsed."""


class G2CPParser:
    """Parses G²CP text-format messages into structured G2CPMessage objects.

    Supports the canonical text format:
        <sender> TO <receiver>
        PERFORMATIVE: <perf>
        CONVERSATION: <conv_id>
        OPERATION:
          TRAVERSE
            FROM: {node_selector}
            VIA: {edge_types}
            DEPTH: <int>
            RETURN: <format>
    """

    # Regex patterns for parsing
    _HEADER_RE = re.compile(r"^(\S+)\s+TO\s+(\S+)$")
    _FIELD_RE = re.compile(r"^\s*(\w+):\s*(.+)$")
    _SET_RE = re.compile(r"\{(.+?)\}")

    def parse(self, raw: str) -> G2CPMessage:
        """Parse a raw G²CP text message into a G2CPMessage.

        Args:
            raw: The raw text message in G²CP format.

        Returns:
            Parsed G2CPMessage object.

        Raises:
            G2CPParseError: If the message is malformed.
        """
        lines = [ln.rstrip() for ln in raw.strip().split("\n")]
        if not lines:
            raise G2CPParseError("Empty message")

        # Parse header: <sender> TO <receiver>
        header_match = self._HEADER_RE.match(lines[0])
        if not header_match:
            raise G2CPParseError(f"Malformed header: {lines[0]}")
        sender, receiver = header_match.group(1), header_match.group(2)

        # Parse fields
        fields: dict[str, str] = {}
        operation_lines: list[str] = []
        result_lines: list[str] = []
        current_section: str | None = None

        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue

            field_match = self._FIELD_RE.match(line)
            if field_match and current_section is None:
                key, value = field_match.group(1), field_match.group(2)
                if key == "OPERATION":
                    current_section = "operation"
                    continue
                elif key == "RESULT":
                    current_section = "result"
                    continue
                fields[key] = value
            elif current_section == "operation":
                if line.strip().startswith("RESULT"):
                    current_section = "result"
                    continue
                operation_lines.append(line)
            elif current_section == "result":
                result_lines.append(line)
            elif line.strip().startswith("OPERATION"):
                current_section = "operation"
            elif line.strip().startswith("RESULT"):
                current_section = "result"
            else:
                # Try to parse as a field
                if field_match:
                    fields[field_match.group(1)] = field_match.group(2)

        # Parse performative
        perf_str = fields.get("PERFORMATIVE", "").strip()
        try:
            performative = Performative(perf_str)
        except ValueError:
            raise G2CPParseError(f"Unknown performative: {perf_str}")

        # Parse conversation context
        conv_id = fields.get("CONVERSATION", "unknown")
        context = MessageContext(conversation_id=conv_id)

        # Parse operation
        operation = self._parse_operation(operation_lines) if operation_lines else None

        # Parse result
        result = self._parse_result(result_lines) if result_lines else None

        # Parse error
        error = fields.get("ERROR")

        return G2CPMessage(
            sender=sender,
            receiver=receiver,
            performative=performative,
            operation=operation,
            result=result,
            context=context,
            error=error,
        )

    def _parse_operation(self, lines: list[str]) -> TraversalOperation | UpdateOperation | None:
        """Parse the OPERATION block."""
        op_fields: dict[str, str] = {}
        op_type: str | None = None

        for line in lines:
            stripped = line.strip()
            if stripped in ("TRAVERSE", "UPDATE", "UPDATE APPLY"):
                op_type = "TRAVERSE" if stripped == "TRAVERSE" else "UPDATE"
                continue
            field_match = self._FIELD_RE.match(line)
            if field_match:
                op_fields[field_match.group(1).strip()] = field_match.group(2).strip()

        if op_type == "TRAVERSE":
            return self._build_traversal(op_fields)
        elif op_type == "UPDATE":
            return self._build_update(op_fields, lines)
        return None

    def _build_traversal(self, fields: dict[str, str]) -> TraversalOperation:
        """Build a TraversalOperation from parsed fields."""
        # Parse FROM
        from_str = fields.get("FROM", "")
        source = self._parse_node_selector(from_str)

        # Parse VIA
        via_str = fields.get("VIA", "")
        edge_types = self._parse_set(via_str)

        # Parse DEPTH
        depth_str = fields.get("DEPTH", "1")
        try:
            depth = int(depth_str)
        except ValueError:
            depth = 1

        # Parse RETURN
        ret_str = fields.get("RETURN", "SUBGRAPH").upper()
        try:
            return_format = ReturnFormat(ret_str)
        except ValueError:
            return_format = ReturnFormat.SUBGRAPH

        # Parse CONSTRAINTS
        constraints = fields.get("CONSTRAINTS")

        return TraversalOperation(
            source=source,
            edge_types=edge_types,
            depth=depth,
            return_format=return_format,
            constraints=constraints,
        )

    def _build_update(self, fields: dict[str, str], lines: list[str]) -> UpdateOperation:
        """Build an UpdateOperation from parsed fields."""
        delta = GraphDelta()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("ADD_EDGE:"):
                edge_data = stripped[len("ADD_EDGE:"):].strip()
                delta.add_edges.append(self._parse_dict_value(edge_data))
            elif stripped.startswith("ADD_NODE:"):
                node_data = stripped[len("ADD_NODE:"):].strip()
                delta.add_nodes.append(self._parse_dict_value(node_data))
            elif stripped.startswith("REMOVE_EDGE:"):
                edge_id = stripped[len("REMOVE_EDGE:"):].strip()
                delta.remove_edges.append(edge_id)
            elif stripped.startswith("REMOVE_NODE:"):
                node_id = stripped[len("REMOVE_NODE:"):].strip()
                delta.remove_nodes.append(node_id)

        return UpdateOperation(delta=delta)

    def _parse_node_selector(self, raw: str) -> NodeSelector:
        """Parse a node selector from text."""
        raw = raw.strip().strip("{}")
        if not raw:
            return NodeSelector()

        # Check for type-based selector
        if raw.startswith("type:"):
            types = [t.strip().replace("type:", "") for t in raw.split(",")]
            return NodeSelector(type_filter=types)

        # Check for property-based selector
        if " WHERE " in raw:
            return NodeSelector(property_filter=raw)

        # Check for context reference
        if raw == "CURRENT_FOCUS":
            return NodeSelector(context_ref="CURRENT_FOCUS")

        # Default: explicit IDs
        ids = [nid.strip() for nid in raw.split(",") if nid.strip()]
        return NodeSelector(explicit_ids=ids)

    def _parse_set(self, raw: str) -> list[str]:
        """Parse a set notation {a, b, c} into a list."""
        raw = raw.strip().strip("{}")
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _parse_dict_value(self, raw: str) -> dict:
        """Parse a dict-like value from text."""
        raw = raw.strip().strip("{}")
        result: dict = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, value = pair.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                # Try numeric conversion
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                result[key] = value
        return result

    def _parse_result(self, lines: list[str]) -> Optional[SubgraphResult]:
        """Parse a RESULT block."""
        nodes: list[dict] = []
        edges: list[dict] = []
        paths: list[list[str]] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("NODES:"):
                node_strs = self._parse_set(stripped[len("NODES:"):])
                nodes = [{"uid": n} for n in node_strs]
            elif stripped.startswith("EDGES:"):
                pass  # Summary only in text format
            elif stripped.startswith("PATH:"):
                path_str = stripped[len("PATH:"):].strip()
                path = [p.strip() for p in path_str.split("->")]
                paths.append(path)

        if nodes or edges or paths:
            return SubgraphResult(nodes=nodes, edges=edges, paths=paths)
        return None
