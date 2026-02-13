"""Single-Agent RAG Baseline.

A monolithic retrieval-augmented generation system with no agent decomposition.
Same graph and LLMs as G²CP — uses a single prompt combining all capabilities.
"""

from __future__ import annotations

import time
from typing import Any

from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient
from g2cp.utils.tokens import TokenCounter

SINGLE_AGENT_PROMPT = """You are an industrial maintenance AI assistant with access to a knowledge graph.
You can query the graph using Cypher. Analyze the user's query, retrieve relevant information,
and provide a comprehensive response covering diagnosis, procedures, and safety considerations.

Available node types: Component, Fault, Procedure, WorkOrder, Part, Sensor, SafetyProtocol
Available edge types: causes, indicates, correlates_with, addressed_by, requires, precedes,
has_safety_protocol, occurred_in, replaced_in, failed_after"""


class SingleAgentSystem:
    """Single-agent RAG baseline with no multi-agent decomposition."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm
        self.token_counter = TokenCounter()

    def process_query(self, query: str) -> dict[str, Any]:
        """Process query with a single monolithic agent."""
        start_time = time.time()

        # Single LLM call combining all agent capabilities
        prompt = (
            f"System: {SINGLE_AGENT_PROMPT}\n\n"
            f"User query: {query}\n\n"
            f"Provide diagnosis, recommended procedure, and safety warnings."
        )
        response = self.llm.complete(prompt)
        token_count = self.token_counter.count(prompt)

        elapsed = time.time() - start_time
        return {
            "response": response,
            "messages": [],  # No inter-agent messages
            "metrics": {
                "total_tokens": token_count,
                "total_time_s": round(elapsed, 3),
                "n_messages": 0,
            },
        }
