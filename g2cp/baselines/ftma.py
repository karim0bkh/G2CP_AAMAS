"""Free-Text Multi-Agent (FTMA) Baseline.

Agents communicate through natural language using GPT-4.
Same knowledge graph and LLMs as G²CP — only communication format differs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient
from g2cp.utils.tokens import TokenCounter

logger = logging.getLogger(__name__)

FTMA_DIAGNOSTIC_PROMPT = """You are a diagnostic specialist for industrial equipment maintenance.
You have access to a graph_query tool that executes Cypher queries on the industrial knowledge graph.
When another agent asks you to diagnose a problem, analyze the symptoms, query the graph for 
relevant fault information, and respond in natural language with your findings."""

FTMA_PROCEDURAL_PROMPT = """You are a maintenance procedure specialist. When given a diagnosed fault,
search the knowledge graph for repair procedures, required parts, and safety protocols.
Communicate findings in natural language."""

FTMA_SYNTHESIS_PROMPT = """You are a pattern analysis specialist. When asked about historical trends,
query the knowledge graph for work order history and identify recurring patterns.
Report findings in natural language."""

FTMA_DISPATCHER_PROMPT = """You are a dispatcher coordinating a team of maintenance AI agents.
Analyze user queries and route them to the appropriate specialist.
Communicate with other agents in natural language.
Synthesize their responses into a final answer for the user."""


class FTMASystem:
    """Free-Text Multi-Agent baseline system.

    All inter-agent communication uses natural language (GPT-4).
    Same graph, same LLMs — only the communication format differs from G²CP.
    """

    def __init__(self, db: GraphDB, llm: LLMClient) -> None:
        self.db = db
        self.llm = llm
        self.token_counter = TokenCounter()

    def process_query(self, query: str) -> dict[str, Any]:
        """Process a query using free-text multi-agent communication."""
        start_time = time.time()
        messages: list[str] = []
        total_tokens = 0

        # Step 1: Dispatcher analyzes and routes
        dispatch_prompt = (
            f"System: {FTMA_DISPATCHER_PROMPT}\n\n"
            f"User query: {query}\n\n"
            f"Analyze this query and describe what information the diagnostic agent "
            f"should look for. Respond in natural language."
        )
        dispatch_response = self.llm.complete(dispatch_prompt)
        messages.append(dispatch_response)
        total_tokens += self.token_counter.count(dispatch_response)

        # Step 2: Diagnostic agent processes
        diag_prompt = (
            f"System: {FTMA_DIAGNOSTIC_PROMPT}\n\n"
            f"Request from dispatcher: {dispatch_response}\n\n"
            f"Query the knowledge graph and provide your diagnostic analysis "
            f"in natural language."
        )
        diag_response = self.llm.complete(diag_prompt)
        messages.append(diag_response)
        total_tokens += self.token_counter.count(diag_response)

        # Step 3: Procedural agent retrieves procedures
        proc_prompt = (
            f"System: {FTMA_PROCEDURAL_PROMPT}\n\n"
            f"Diagnostic findings: {diag_response}\n\n"
            f"Find the appropriate repair procedures and respond in natural language."
        )
        proc_response = self.llm.complete(proc_prompt)
        messages.append(proc_response)
        total_tokens += self.token_counter.count(proc_response)

        # Step 4: Dispatcher synthesizes
        final_prompt = (
            f"System: {FTMA_DISPATCHER_PROMPT}\n\n"
            f"Original query: {query}\n"
            f"Diagnostic response: {diag_response}\n"
            f"Procedural response: {proc_response}\n\n"
            f"Synthesize a final response for the user."
        )
        final_response = self.llm.complete(final_prompt)

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
