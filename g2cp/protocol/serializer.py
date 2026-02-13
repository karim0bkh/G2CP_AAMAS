"""G²CP message serialization and deserialization utilities."""

from __future__ import annotations

import json
from typing import Any

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


def message_to_dict(msg: G2CPMessage) -> dict[str, Any]:
    """Serialize a G2CPMessage to a JSON-compatible dictionary."""
    d: dict[str, Any] = {
        "id": msg.id,
        "sender": msg.sender,
        "receiver": msg.receiver,
        "performative": msg.performative.value,
        "context": {
            "conversation_id": msg.context.conversation_id,
            "timestamp": msg.context.timestamp,
            "parent_message_id": msg.context.parent_message_id,
        },
    }
    if msg.operation:
        if isinstance(msg.operation, TraversalOperation):
            d["operation"] = {
                "type": "TRAVERSE",
                "source": msg.operation.source.model_dump(),
                "edge_types": msg.operation.edge_types,
                "depth": msg.operation.depth,
                "return_format": msg.operation.return_format.value,
                "constraints": msg.operation.constraints,
            }
        elif isinstance(msg.operation, UpdateOperation):
            d["operation"] = {
                "type": "UPDATE",
                "delta": msg.operation.delta.model_dump(),
            }
    if msg.result:
        d["result"] = msg.result.model_dump()
    if msg.error:
        d["error"] = msg.error
    if msg.signature:
        d["signature"] = msg.signature
    return d


def message_from_dict(d: dict[str, Any]) -> G2CPMessage:
    """Deserialize a G2CPMessage from a dictionary."""
    operation = None
    if "operation" in d and d["operation"]:
        op = d["operation"]
        if op["type"] == "TRAVERSE":
            operation = TraversalOperation(
                source=NodeSelector(**op["source"]),
                edge_types=op.get("edge_types", []),
                depth=op.get("depth", 1),
                return_format=ReturnFormat(op.get("return_format", "SUBGRAPH")),
                constraints=op.get("constraints"),
            )
        elif op["type"] == "UPDATE":
            operation = UpdateOperation(delta=GraphDelta(**op["delta"]))

    result = None
    if "result" in d and d["result"]:
        result = SubgraphResult(**d["result"])

    context = MessageContext(
        conversation_id=d.get("context", {}).get("conversation_id", "unknown"),
        timestamp=d.get("context", {}).get("timestamp", 0),
        parent_message_id=d.get("context", {}).get("parent_message_id"),
    )

    return G2CPMessage(
        id=d.get("id", ""),
        sender=d["sender"],
        receiver=d["receiver"],
        performative=Performative(d["performative"]),
        operation=operation,
        result=result,
        context=context,
        error=d.get("error"),
        signature=d.get("signature"),
    )


def message_to_json(msg: G2CPMessage) -> str:
    """Serialize to JSON string."""
    return json.dumps(message_to_dict(msg), default=str)


def message_from_json(raw: str) -> G2CPMessage:
    """Deserialize from JSON string."""
    return message_from_dict(json.loads(raw))
