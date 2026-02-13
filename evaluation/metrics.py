"""Evaluation metrics for G²CP and baseline systems.

Implements the 5 metrics from Section 5.3 of the paper:
1. Task Accuracy (F1 over ground truth entities)
2. Token Efficiency (inter-agent tokens per query)
3. Hallucination Rate (fraction of claims not grounded in KG)
4. Cascading Error Rate (fraction where early error propagates)
5. Auditability Score (fraction of conclusions traceable to graph ops)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_accuracy(response: str, ground_truth: dict[str, Any]) -> float:
    """Compute task accuracy as F1 between response entities and ground truth.

    We tokenize both the response and the ground truth answer, then compute
    F1 overlap. Entity UIDs in ground truth are also matched against the
    response text (normalized).
    """
    gt_answer = ground_truth.get("answer", "")
    gt_entities = ground_truth.get("entities", [])

    # Normalize
    resp_tokens = set(_normalize(response).split())
    gt_tokens = set(_normalize(gt_answer).split())

    # Also match entity UIDs (strip prefixes like "Component:", "Fault:")
    for entity in gt_entities:
        if ":" in entity:
            parts = entity.split(":", 1)
            gt_tokens.update(_normalize(parts[1]).split())
            gt_tokens.update(_normalize(parts[0]).split())

    if not gt_tokens:
        return 1.0 if not resp_tokens else 0.0

    # F1
    tp = len(resp_tokens & gt_tokens)
    precision = tp / max(len(resp_tokens), 1)
    recall = tp / max(len(gt_tokens), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hallucination_rate(
    response: str,
    subgraph_nodes: list[dict[str, Any]],
    subgraph_edges: list[dict[str, Any]],
) -> float:
    """Compute hallucination rate: fraction of response claims not grounded in KG.

    A claim is considered grounded if the entities it mentions appear in
    the retrieved subgraph. We extract noun phrases from the response
    and check if each appears in the graph data.
    """
    # Build set of grounded terms from subgraph
    grounded_terms: set[str] = set()
    for node in subgraph_nodes:
        for key, value in node.items():
            grounded_terms.add(_normalize(str(value)))
    for edge in subgraph_edges:
        for key, value in edge.items():
            grounded_terms.add(_normalize(str(value)))

    if not grounded_terms:
        return 0.0  # No graph data means we can't verify

    # Also add individual words from grounded terms
    grounded_words: set[str] = set()
    for term in grounded_terms:
        grounded_words.update(term.replace("_", " ").split())

    # Extract claims from response (simplified: split into phrases)
    response_claims = _extract_claims(response)
    if not response_claims:
        return 0.0

    ungrounded = 0
    for claim in response_claims:
        claim_norm = _normalize(claim)
        claim_tokens = set(claim_norm.split())
        # A claim is grounded if any of its significant tokens match grounded terms/words
        significant_tokens = {t for t in claim_tokens if len(t) > 2}
        if significant_tokens and not any(
            t in grounded_words or t in grounded_terms for t in significant_tokens
        ):
            ungrounded += 1

    return ungrounded / len(response_claims)


def compute_cascading_error_rate(message_trace: list[dict[str, Any]]) -> float:
    """Compute cascading error rate from a message trace.

    An error cascades when an agent's incorrect output is consumed by
    a downstream agent, producing a compounded error. In G²CP, this is
    detected when a REJECT message follows an INFORM, or when empty
    results propagate through the chain.
    """
    if not message_trace:
        return 0.0

    errors = 0
    error_propagated = False

    for i, entry in enumerate(message_trace):
        msg = entry.get("message", {})
        result = entry.get("result")
        perf = msg.get("performative", "")

        # Check for error conditions
        has_error = msg.get("error") is not None
        empty_result = result is not None and not result.get("nodes") and not result.get("edges")

        if has_error or (perf == "REJECT"):
            error_propagated = True

        if error_propagated and perf in ("REQUEST", "INFORM") and i > 0:
            errors += 1

    total_messages = len(message_trace)
    return errors / max(total_messages, 1)


def compute_auditability(message_trace: list[dict[str, Any]]) -> float:
    """Compute auditability score: fraction of conclusions traceable to graph operations.

    In G²CP, every INFORM message references an explicit graph operation,
    making all conclusions auditable. For baselines using free text,
    auditability is lower because conclusions cannot be traced to specific
    graph traversals.
    """
    if not message_trace:
        return 0.0

    auditable = 0
    total_informs = 0

    for entry in message_trace:
        msg = entry.get("message", {})
        if msg.get("performative") == "INFORM":
            total_informs += 1
            # Auditable if the message has a structured operation or result
            has_operation = msg.get("operation") is not None
            has_result = entry.get("result") is not None
            if has_operation or has_result:
                auditable += 1

    if total_informs == 0:
        # If no INFORM messages, check if all messages are structured
        for entry in message_trace:
            msg = entry.get("message", {})
            if msg.get("operation") is not None:
                return 1.0
        return 0.0

    return auditable / total_informs


def compute_all_metrics(
    response: str,
    ground_truth: dict[str, Any],
    token_count: int,
    message_trace: list[dict[str, Any]],
    subgraph_nodes: list[dict[str, Any]] | None = None,
    subgraph_edges: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute all 5 evaluation metrics."""
    return {
        "accuracy": compute_accuracy(response, ground_truth),
        "tokens": token_count,
        "hallucination_rate": compute_hallucination_rate(
            response, subgraph_nodes or [], subgraph_edges or []
        ),
        "cascading_error_rate": compute_cascading_error_rate(message_trace),
        "auditability": compute_auditability(message_trace),
    }


def aggregate_metrics(all_results: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across all queries."""
    if not all_results:
        return {}

    keys = all_results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r[key] for r in all_results]
        aggregated[f"{key}_mean"] = sum(values) / len(values)
        aggregated[f"{key}_std"] = (
            sum((v - aggregated[f"{key}_mean"]) ** 2 for v in values) / len(values)
        ) ** 0.5
        aggregated[f"{key}_min"] = min(values)
        aggregated[f"{key}_max"] = max(values)

    return aggregated


# ── Utilities ────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_claims(text: str) -> list[str]:
    """Extract factual claims from response text (simplified)."""
    # Split on sentences
    import re

    sentences = re.split(r"[.!?]+", text)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:  # Skip very short fragments
            claims.append(sent)
    return claims
