"""Tests for G²CP protocol module — messages, operations, parser, commitments."""

import json
import pytest

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
from g2cp.protocol.operations import TraversalBuilder, UpdateBuilder
from g2cp.protocol.parser import G2CPParser
from g2cp.protocol.serializer import message_from_json, message_to_json
from g2cp.protocol.commitments import CommitmentState, CommitmentStore


# ── Message Construction ──────────────────────────────────────────────────


class TestG2CPMessage:
    def test_create_basic_message(self):
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
        )
        assert msg.sender == "dispatcher"
        assert msg.receiver == "diagnostic"
        assert msg.performative == Performative.REQUEST
        assert msg.id  # UUID generated

    def test_traversal_message(self):
        op = TraversalOperation(
            source=NodeSelector(explicit_ids=["Symptom:grinding_1200RPM"]),
            edge_types=["indicates", "causes"],
            depth=2,
            return_format=ReturnFormat.SUBGRAPH,
        )
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
            operation=op,
        )
        assert msg.is_traversal
        assert not msg.is_update
        assert msg.operation.depth == 2

    def test_update_message(self):
        op = UpdateOperation(
            delta=GraphDelta(
                add_edges=[{"from": "Fault:X", "to": "Sensor:Y", "type": "risk_indicator"}]
            )
        )
        msg = G2CPMessage(
            sender="synthesis",
            receiver="ingestion",
            performative=Performative.UPDATE,
            operation=op,
        )
        assert msg.is_update
        assert not msg.is_traversal

    def test_message_hash_deterministic(self):
        msg = G2CPMessage(
            sender="a",
            receiver="b",
            performative=Performative.INFORM,
            context=MessageContext(conversation_id="conv1"),
        )
        h1 = msg.compute_hash()
        h2 = msg.compute_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_serialize_roundtrip(self):
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
        )
        text = msg.serialize()
        assert "dispatcher TO diagnostic" in text
        assert "REQUEST" in text


# ── Node Selector ─────────────────────────────────────────────────────────


class TestNodeSelector:
    def test_explicit_ids(self):
        sel = NodeSelector(explicit_ids=["Fault:X", "Fault:Y"])
        clause = sel.to_cypher_clause()
        assert '"Fault:X"' in clause
        assert '"Fault:Y"' in clause

    def test_type_filter(self):
        sel = NodeSelector(type_filter=["Fault", "Symptom"])
        clause = sel.to_cypher_clause()
        assert "n:Fault" in clause
        assert "n:Symptom" in clause

    def test_is_empty(self):
        sel = NodeSelector()
        assert sel.is_empty
        sel2 = NodeSelector(explicit_ids=["X"])
        assert not sel2.is_empty


# ── Builder Pattern ───────────────────────────────────────────────────────


class TestBuilders:
    def test_traversal_builder(self):
        op = (
            TraversalBuilder()
            .from_nodes(["Symptom:grinding_1200RPM", "Symptom:temp_85C"])
            .via(["causes", "indicates"])
            .depth(2)
            .return_subgraph()
            .build()
        )
        assert isinstance(op, TraversalOperation)
        assert len(op.source.explicit_ids) == 2
        assert op.depth == 2
        assert op.return_format == ReturnFormat.SUBGRAPH
        assert "causes" in op.edge_types

    def test_update_builder(self):
        op = (
            UpdateBuilder()
            .add_edge("Fault:X", "Sensor:Y", "risk_indicator", weight=0.89)
            .add_node("Node:new1", "Sensor", name="New Sensor")
            .build()
        )
        assert isinstance(op, UpdateOperation)
        assert len(op.delta.add_edges) == 1
        assert len(op.delta.add_nodes) == 1
        assert op.delta.add_edges[0]["weight"] == 0.89

    def test_traversal_builder_from_type(self):
        op = TraversalBuilder().from_type("Fault", "Symptom").via(["indicates"]).depth(1).build()
        assert op.source.type_filter == ["Fault", "Symptom"]

    def test_traversal_builder_return_paths(self):
        op = TraversalBuilder().from_nodes(["X"]).return_paths().build()
        assert op.return_format == ReturnFormat.PATHS


# ── Parser ────────────────────────────────────────────────────────────────


class TestParser:
    def test_parse_request(self):
        raw = """dispatcher TO diagnostic
PERFORMATIVE: REQUEST
CONVERSATION: conv-001
OPERATION:
  TRAVERSE
    FROM: {Symptom:grinding_1200RPM}
    VIA: {indicates, causes}
    DEPTH: 2
    RETURN: SUBGRAPH"""

        parser = G2CPParser()
        msg = parser.parse(raw)
        assert msg.sender == "dispatcher"
        assert msg.receiver == "diagnostic"
        assert msg.performative == Performative.REQUEST
        assert msg.context.conversation_id == "conv-001"
        assert msg.is_traversal

    def test_parse_inform(self):
        raw = """diagnostic TO dispatcher
PERFORMATIVE: INFORM
CONVERSATION: conv-001"""

        parser = G2CPParser()
        msg = parser.parse(raw)
        assert msg.performative == Performative.INFORM

    def test_parse_all_performatives(self):
        parser = G2CPParser()
        for perf in Performative:
            raw = f"a TO b\nPERFORMATIVE: {perf.value}\nCONVERSATION: c1"
            msg = parser.parse(raw)
            assert msg.performative == perf


# ── JSON Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_json_roundtrip(self):
        op = TraversalOperation(
            source=NodeSelector(explicit_ids=["X"]),
            edge_types=["causes"],
            depth=2,
        )
        original = G2CPMessage(
            sender="a",
            receiver="b",
            performative=Performative.REQUEST,
            operation=op,
        )
        json_str = message_to_json(original)
        restored = message_from_json(json_str)
        assert restored.sender == original.sender
        assert restored.receiver == original.receiver
        assert restored.performative == original.performative

    def test_json_with_result(self):
        msg = G2CPMessage(
            sender="a",
            receiver="b",
            performative=Performative.INFORM,
            result=SubgraphResult(
                nodes=[{"uid": "X", "type": "Fault"}],
                edges=[{"from": "X", "to": "Y", "type": "causes"}],
            ),
        )
        json_str = message_to_json(msg)
        restored = message_from_json(json_str)
        assert len(restored.result.nodes) == 1


# ── Commitment Semantics ─────────────────────────────────────────────────


class TestCommitments:
    def test_request_creates_commitment(self):
        store = CommitmentStore()
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
            operation=TraversalBuilder().from_nodes(["X"]).build(),
        )
        commitment = store.create_from_message(msg)
        assert commitment is not None
        assert commitment.debtor == "diagnostic"
        assert commitment.creditor == "dispatcher"
        assert commitment.state == CommitmentState.ACTIVE

    def test_confirm_fulfills_commitment(self):
        store = CommitmentStore()
        # Create REQUEST commitment
        req = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
            operation=TraversalBuilder().from_nodes(["X"]).build(),
        )
        store.create_from_message(req)
        assert len(store.active) == 1

        # CONFIRM fulfills it
        confirm = G2CPMessage(
            sender="diagnostic",
            receiver="dispatcher",
            performative=Performative.CONFIRM,
        )
        store.create_from_message(confirm)
        # The CONFIRM itself creates a fulfilled commitment
        assert any(c.state == CommitmentState.FULFILLED for c in store.all)

    def test_reject_creates_violation_commitment(self):
        store = CommitmentStore()
        msg = G2CPMessage(
            sender="ingestion",
            receiver="synthesis",
            performative=Performative.REJECT,
            error="Invalid edge type",
        )
        c = store.create_from_message(msg)
        assert c.condition == "violated(op, constraint)"
