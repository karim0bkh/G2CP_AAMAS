"""JSON-Structured Multi-Agent (JSMA) Baseline.

Agents exchange JSON objects with typed fields, but without graph-grounded semantics.
Same graph and LLMs as G²CP — communication uses structured JSON instead of NL or graph ops.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient
from g2cp.utils.tokens import TokenCounter

logger = logging.getLogger(__name__)

JSMA_DIAGNOSTIC_PROMPT = """You are a diagnostic specialist. Communicate using JSON messages.
Input format: {{"action": "diagnose", "symptoms": ["symptom1", ...]}}
Output format: {{"faults": [{{"id": "...", "name": "...", "confidence": 0.0}}], "reasoning": "..."}}
Use graph_query tool for Cypher queries."""

JSMA_PROCEDURAL_PROMPT = """You are a procedure specialist. Communicate using JSON.
Input: {{"action": "get_procedure", "fault_id": "..."}}
Output: {{"procedures": [{{"id": "...", "steps": [...], "parts": [...]}}]}}"""


class JSMASystem:
    """JSON-Structured Multi-Agent baseline system."""

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm
        self.token_counter = TokenCounter()

    def process_query(self, query: str) -> dict[str, Any]:
        """Process a query using JSON-structured multi-agent communication."""
        start_time = time.time()
        messages: list[str] = []
        total_tokens = 0

        # Step 1: Dispatcher creates JSON request
        dispatch_msg = json.dumps({
            "action": "diagnose",
            "query": query,
            "symptoms": self._extract_symptoms(query),
        })
        messages.append(dispatch_msg)
        total_tokens += self.token_counter.count(dispatch_msg)

        # Step 2: Diagnostic processes JSON
        diag_prompt = (
            f"System: {JSMA_DIAGNOSTIC_PROMPT}\n\n"
            f"Input: {dispatch_msg}\n\n"
            f"Respond with a valid JSON object only."
        )
        diag_response = self.llm.complete(diag_prompt)
        messages.append(diag_response)
        total_tokens += self.token_counter.count(diag_response)

        # Step 3: Parse diagnostic results, send to procedural
        try:
            diag_data = json.loads(diag_response)
            faults = diag_data.get("faults", [])
        except json.JSONDecodeError:
            faults = []

        proc_msg = json.dumps({
            "action": "get_procedure",
            "fault_id": faults[0].get("id", "unknown") if faults else "unknown",
        })
        messages.append(proc_msg)
        total_tokens += self.token_counter.count(proc_msg)

        # Step 4: Procedural responds
        proc_prompt = (
            f"System: {JSMA_PROCEDURAL_PROMPT}\n\n"
            f"Input: {proc_msg}\n\n"
            f"Respond with a valid JSON object only."
        )
        proc_response = self.llm.complete(proc_prompt)
        messages.append(proc_response)
        total_tokens += self.token_counter.count(proc_response)

        # Step 5: Final synthesis
        final_response = self.llm.complete(
            f"Synthesize a user-facing response from:\n"
            f"Query: {query}\nDiagnostic: {diag_response}\nProcedural: {proc_response}"
        )

        elapsed = time.time() - start_time
        return {
            "response": final_response,
            "messages": messages,
            "metrics": {
                "total_tokens": total_tokens,
                "total_time_s": round(elapsed, 3),
                "n_messages": len(messages),
            },
        }

    def _extract_symptoms(self, query: str) -> list[str]:
        """Simple symptom extraction from query text."""
        symptom_keywords = [
            "noise", "grinding", "vibration", "pressure", "temperature",
            "leak", "failure", "drop", "fluctuation", "overheat",
        ]
        found = []
        q_lower = query.lower()
        for kw in symptom_keywords:
            if kw in q_lower:
                found.append(kw)
        return found if found else [query]
