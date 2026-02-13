"""Tests for G²CP engine, agents, and end-to-end pipeline."""

import json
from pathlib import Path

import pytest

from g2cp.agents.diagnostic import DiagnosticAgent
from g2cp.agents.dispatcher import DispatcherAgent
from g2cp.agents.ingestion import IngestionAgent
from g2cp.agents.procedural import ProceduralAgent
from g2cp.agents.synthesis import SynthesisAgent
from g2cp.engine.audit import AuditLog
from g2cp.engine.executor import GraphExecutor
from g2cp.engine.runtime import G2CPRuntime
from g2cp.engine.security import SecurityManager
from g2cp.protocol.messages import (
    G2CPMessage,
    MessageContext,
    NodeSelector,
    Performative,
    SubgraphResult,
    TraversalOperation,
)
from g2cp.protocol.operations import TraversalBuilder, UpdateBuilder
from g2cp.utils.embeddings import EntityLinker
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient
from g2cp.utils.tokens import TokenCounter


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Create an in-memory graph database with test data."""
    db = GraphDB()  # No password → in-memory mock
    db.load_nodes([
        {"uid": "Symptom:grinding_1200RPM", "type": "Symptom", "name": "Grinding noise at 1200 RPM"},
        {"uid": "Symptom:pressure_drop", "type": "Symptom", "name": "Pressure drop"},
        {"uid": "Fault:bearing_wear_B4521", "type": "Fault", "name": "Bearing Wear B-4521", "severity": "high"},
        {"uid": "Fault:seal_degradation", "type": "Fault", "name": "Seal Degradation"},
        {"uid": "Procedure:P-205", "type": "Procedure", "name": "Bearing Replacement", "estimated_hours": 4},
        {"uid": "Part:bearing_SKF_6210", "type": "Part", "name": "SKF 6210 Bearing", "cost": 85.0},
        {"uid": "SafetyProtocol:SP-001", "type": "SafetyProtocol", "name": "Lockout/Tagout"},
    ])
    db.load_edges([
        {"from": "Symptom:grinding_1200RPM", "to": "Fault:bearing_wear_B4521", "type": "indicates", "weight": 0.92},
        {"from": "Symptom:pressure_drop", "to": "Fault:seal_degradation", "type": "indicates", "weight": 0.91},
        {"from": "Fault:bearing_wear_B4521", "to": "Procedure:P-205", "type": "addressed_by", "weight": 0.95},
        {"from": "Procedure:P-205", "to": "Part:bearing_SKF_6210", "type": "requires", "weight": 1.0},
        {"from": "Procedure:P-205", "to": "SafetyProtocol:SP-001", "type": "has_safety_protocol", "weight": 1.0},
    ])
    return db


@pytest.fixture
def mock_llm():
    """Create a mock LLM client (no API key = mock mode)."""
    return LLMClient()


@pytest.fixture
def mock_linker():
    """Create an entity linker with indexed nodes."""
    linker = EntityLinker(threshold=0.5)  # Low threshold for string matching
    linker.build_index([
        {"uid": "Symptom:grinding_1200RPM", "name": "Grinding noise at 1200 RPM"},
        {"uid": "Fault:bearing_wear_B4521", "name": "Bearing Wear B-4521"},
        {"uid": "Procedure:P-205", "name": "Bearing Replacement"},
    ])
    return linker


# ── Graph Executor ────────────────────────────────────────────────────────


class TestGraphExecutor:
    def test_traverse_explicit_ids(self, mock_db):
        executor = GraphExecutor(mock_db)
        op = (
            TraversalBuilder()
            .from_nodes(["Symptom:grinding_1200RPM"])
            .via(["indicates"])
            .depth(1)
            .return_subgraph()
            .build()
        )
        result = executor.execute_traversal(op)
        assert isinstance(result, SubgraphResult)
        # Should find at least the source node
        assert len(result.nodes) >= 1 or result.metadata

    def test_traverse_by_type(self, mock_db):
        executor = GraphExecutor(mock_db)
        op = TraversalBuilder().from_type("Symptom").via(["indicates"]).depth(1).build()
        result = executor.execute_traversal(op)
        assert isinstance(result, SubgraphResult)

    def test_update_add_edge(self, mock_db):
        executor = GraphExecutor(mock_db)
        op = (
            UpdateBuilder()
            .add_edge("Fault:bearing_wear_B4521", "Symptom:pressure_drop", "correlates_with", weight=0.7)
            .build()
        )
        result = executor.execute_update(op)
        assert result.metadata.get("applied_edges") == 1


# ── Agents ────────────────────────────────────────────────────────────────


class TestDiagnosticAgent:
    def test_handle_request(self, mock_db, mock_llm):
        agent = DiagnosticAgent(db=mock_db, llm=mock_llm)
        op = TraversalBuilder().from_nodes(["Symptom:grinding_1200RPM"]).via(["indicates"]).depth(1).build()
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
            operation=op,
        )
        result_subgraph = agent.execute_operation(op)
        response = agent.handle_message(msg, result_subgraph)
        assert "messages" in response
        assert len(response["messages"]) >= 1
        assert response["messages"][0].performative == Performative.INFORM


class TestProceduralAgent:
    def test_handle_request(self, mock_db, mock_llm):
        agent = ProceduralAgent(db=mock_db, llm=mock_llm)
        op = TraversalBuilder().from_nodes(["Fault:bearing_wear_B4521"]).via(["addressed_by"]).depth(1).build()
        msg = G2CPMessage(
            sender="dispatcher",
            receiver="procedural",
            performative=Performative.REQUEST,
            operation=op,
        )
        result_subgraph = agent.execute_operation(op)
        response = agent.handle_message(msg, result_subgraph)
        assert "messages" in response


class TestIngestionAgent:
    def test_valid_update(self, mock_db, mock_llm):
        agent = IngestionAgent(db=mock_db, llm=mock_llm)
        op = (
            UpdateBuilder()
            .add_edge("Fault:bearing_wear_B4521", "Symptom:pressure_drop", "risk_indicator", weight=0.8)
            .build()
        )
        msg = G2CPMessage(
            sender="synthesis",
            receiver="ingestion",
            performative=Performative.UPDATE,
            operation=op,
        )
        response = agent.handle_message(msg)
        assert len(response["messages"]) == 1
        assert response["messages"][0].performative == Performative.CONFIRM

    def test_reject_non_update(self, mock_db, mock_llm):
        agent = IngestionAgent(db=mock_db, llm=mock_llm)
        msg = G2CPMessage(
            sender="synthesis",
            receiver="ingestion",
            performative=Performative.REQUEST,  # Wrong performative
        )
        response = agent.handle_message(msg)
        assert response["messages"] == []


# ── Security Manager ─────────────────────────────────────────────────────


class TestSecurityManager:
    def test_register_and_authorize(self):
        sec = SecurityManager()
        sec.register_agent("diag1", "diagnostic")

        msg = G2CPMessage(
            sender="diag1",
            receiver="dispatcher",
            performative=Performative.INFORM,
            operation=TraversalBuilder().from_nodes(["Symptom:X"]).via(["indicates"]).build(),
        )
        assert sec.authorize(msg) is True

    def test_unauthorized_update(self):
        sec = SecurityManager()
        sec.register_agent("diag1", "diagnostic")

        msg = G2CPMessage(
            sender="diag1",
            receiver="ingestion",
            performative=Performative.UPDATE,
            operation=UpdateBuilder().add_node("X", "Fault").build(),
        )
        # Diagnostic agents cannot UPDATE
        assert sec.authorize(msg) is False

    def test_trust_propagation(self):
        sec = SecurityManager()
        sec.register_agent("agent1", "diagnostic")
        assert sec.get_trust("agent1") == 1.0

        # Successful verification keeps trust high
        t = sec.update_trust("agent1", True)
        assert t > 0.9

        # Failed verification reduces trust
        t = sec.update_trust("agent1", False)
        assert t < 1.0


# ── Audit Log ────────────────────────────────────────────────────────────


class TestAuditLog:
    def test_append_and_trace(self):
        log = AuditLog(conversation_id="test-conv")
        msg = G2CPMessage(
            sender="a", receiver="b", performative=Performative.REQUEST,
        )
        log.append(msg, token_count=42)
        assert log.message_count == 1
        assert log.total_tokens == 42
        trace = log.get_trace()
        assert len(trace) == 1

    def test_verify_trace(self):
        log = AuditLog(conversation_id="test")
        ctx = MessageContext(conversation_id="test")
        # Add REQUEST
        req = G2CPMessage(sender="a", receiver="b", performative=Performative.REQUEST, context=ctx)
        log.append(req)
        # Add matching INFORM
        inf = G2CPMessage(sender="b", receiver="a", performative=Performative.INFORM, context=ctx)
        log.append(inf)

        verification = log.verify_trace()
        assert verification["valid"] is True

    def test_save_load_roundtrip(self, tmp_path):
        log = AuditLog(conversation_id="roundtrip")
        msg = G2CPMessage(sender="x", receiver="y", performative=Performative.QUERY)
        log.append(msg, token_count=10)

        path = tmp_path / "audit.json"
        log.save(path)
        loaded = AuditLog.load(path)
        assert loaded.conversation_id == "roundtrip"
        assert loaded.message_count == 1


# ── Token Counter ────────────────────────────────────────────────────────


class TestTokenCounter:
    def test_count_string(self):
        counter = TokenCounter()
        count = counter.count("Hello world, this is a test message.")
        assert count > 0

    def test_count_g2cp_message(self):
        counter = TokenCounter()
        msg = G2CPMessage(sender="a", receiver="b", performative=Performative.REQUEST)
        count = counter.count_message(msg)
        assert count > 0

    def test_count_conversation(self):
        counter = TokenCounter()
        messages = [
            G2CPMessage(sender="a", receiver="b", performative=Performative.REQUEST),
            G2CPMessage(sender="b", receiver="a", performative=Performative.INFORM),
        ]
        stats = counter.count_conversation(messages)
        assert stats["n_messages"] == 2
        assert stats["total"] > 0


# ── End-to-End Runtime ───────────────────────────────────────────────────


class TestRuntime:
    def test_process_query(self, mock_db, mock_llm, mock_linker):
        agents = {
            "dispatcher": DispatcherAgent(db=mock_db, llm=mock_llm, linker=mock_linker),
            "diagnostic": DiagnosticAgent(db=mock_db, llm=mock_llm),
            "procedural": ProceduralAgent(db=mock_db, llm=mock_llm),
            "synthesis": SynthesisAgent(db=mock_db, llm=mock_llm),
            "ingestion": IngestionAgent(db=mock_db, llm=mock_llm),
        }
        runtime = G2CPRuntime(db=mock_db, agents=agents)
        result = runtime.process_query("Why is there grinding noise?")

        assert "response" in result
        assert "metrics" in result
        assert result["metrics"]["n_messages"] >= 1
        assert result["audit_log"] is not None

    def test_process_without_agents(self, mock_db):
        # Should work with fallback operations
        runtime = G2CPRuntime(db=mock_db)
        result = runtime.process_query("test query")
        assert "response" in result
