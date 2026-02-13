"""Tests for evaluation metrics."""

import pytest

from evaluation.metrics import (
    aggregate_metrics,
    compute_accuracy,
    compute_auditability,
    compute_cascading_error_rate,
    compute_hallucination_rate,
)


class TestAccuracy:
    def test_perfect_match(self):
        score = compute_accuracy(
            "Bearing Wear B-4521 is the fault",
            {"answer": "Bearing Wear B-4521", "entities": ["Fault:bearing_wear_B4521"]},
        )
        assert score > 0.5

    def test_no_match(self):
        score = compute_accuracy(
            "The weather is nice today",
            {"answer": "Bearing Wear", "entities": ["Fault:bearing_wear"]},
        )
        assert score < 0.3

    def test_partial_match(self):
        score = compute_accuracy(
            "The bearing might be worn",
            {"answer": "Bearing wear confirmed", "entities": ["Fault:bearing_wear"]},
        )
        assert 0.0 < score < 1.0

    def test_empty_ground_truth(self):
        score = compute_accuracy("Some response", {"answer": "", "entities": []})
        assert score == 0.0 or score == 1.0  # Edge case


class TestHallucinationRate:
    def test_fully_grounded(self):
        rate = compute_hallucination_rate(
            "Bearing B-4521 has wear damage.",
            [{"uid": "Fault:bearing_wear_B4521", "name": "Bearing Wear B-4521"}],
            [{"from": "Symptom:X", "to": "Fault:bearing_wear_B4521", "type": "indicates"}],
        )
        assert rate < 0.5

    def test_ungrounded(self):
        rate = compute_hallucination_rate(
            "The quantum flux capacitor needs replacement.",
            [{"uid": "Fault:bearing_wear", "name": "Bearing Wear"}],
            [],
        )
        assert rate > 0.0


class TestCascadingErrorRate:
    def test_no_errors(self):
        trace = [
            {"message": {"performative": "REQUEST", "error": None}, "result": {"nodes": ["X"]}},
            {"message": {"performative": "INFORM", "error": None}, "result": {"nodes": ["Y"]}},
        ]
        rate = compute_cascading_error_rate(trace)
        assert rate == 0.0

    def test_with_reject(self):
        trace = [
            {"message": {"performative": "REQUEST", "error": None}, "result": {"nodes": []}},
            {"message": {"performative": "REJECT", "error": "bad"}, "result": None},
            {"message": {"performative": "REQUEST", "error": None}, "result": {"nodes": []}},
        ]
        rate = compute_cascading_error_rate(trace)
        assert rate > 0.0

    def test_empty_trace(self):
        assert compute_cascading_error_rate([]) == 0.0


class TestAuditability:
    def test_fully_auditable(self):
        trace = [
            {"message": {"performative": "INFORM", "operation": {"type": "TRAVERSE"}}, "result": {"nodes": ["X"]}},
            {"message": {"performative": "INFORM", "operation": {"type": "TRAVERSE"}}, "result": {"nodes": ["Y"]}},
        ]
        score = compute_auditability(trace)
        assert score == 1.0

    def test_no_informs(self):
        trace = [
            {"message": {"performative": "REQUEST", "operation": {"type": "TRAVERSE"}}, "result": None},
        ]
        score = compute_auditability(trace)
        assert score == 1.0  # Has operation

    def test_empty(self):
        assert compute_auditability([]) == 0.0


class TestAggregation:
    def test_aggregate(self):
        results = [
            {"accuracy": 0.8, "tokens": 100},
            {"accuracy": 0.9, "tokens": 200},
            {"accuracy": 1.0, "tokens": 300},
        ]
        agg = aggregate_metrics(results)
        assert abs(agg["accuracy_mean"] - 0.9) < 0.001
        assert agg["tokens_mean"] == 200.0
        assert agg["accuracy_min"] == 0.8
        assert agg["tokens_max"] == 300
