"""LLM interface for G²CP agents.

Provides a unified interface to OpenAI GPT-4 (for query understanding)
and local models (Llama 3 for response generation). The LLM is used ONLY
for: (1) parsing user queries, (2) entity extraction, (3) intent classification,
(4) depth estimation, and (5) generating final natural language responses.
Inter-agent communication uses G²CP exclusively — no LLM involved.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Prompt Templates (from Section 4.5 / Appendix B of the paper) ────────

ENTITY_EXTRACTION_PROMPT = """Given the query: "{query}"

Extract all named entities that correspond to industrial equipment, symptoms, 
faults, or parts. Return as JSON: {{"entities": [...]}}

Rules:
- Include specific equipment identifiers (e.g., "B-4521", "HC-3", "P-101")
- Include symptoms (e.g., "grinding noise", "pressure drop", "high temperature")
- Include fault types (e.g., "bearing wear", "seal degradation")
- Include part names (e.g., "main pump", "hydraulic circuit")
- Normalize descriptions: "grinding noise at 1200 RPM" → "grinding_1200RPM"

Return ONLY valid JSON, no other text."""

INTENT_CLASSIFICATION_PROMPT = """Classify the query intent into exactly one of:
- diagnostic: symptom → cause analysis (e.g., "what's wrong with...", "why is there...")
- procedural: fault → fix retrieval (e.g., "how do I fix...", "repair procedure for...")
- predictive: pattern → forecast (e.g., "predict next failure...", "what's at risk...")
- factoid: direct lookup (e.g., "what is the rated pressure of...", "list all sensors on...")

Query: "{query}"

Return ONLY one word: diagnostic, procedural, predictive, or factoid."""

DEPTH_ESTIMATION_PROMPT = """Given {n_entities} source entities and {n_edge_types} edge types,
estimate the number of graph traversal hops (1-3) needed to answer this query.

Guidelines:
- Simple lookups (single entity, direct property): 1
- Causal chains (symptom → fault, fault → procedure): 2
- Complex multi-factor analysis (patterns, cross-references): 3

Query: "{query}"

Return ONLY a single integer: 1, 2, or 3."""

RESPONSE_GENERATION_PROMPT = """You are an industrial maintenance assistant. Generate a clear,
actionable response to the user's query based EXCLUSIVELY on the following graph data.

User query: {query}

Retrieved graph data:
{subgraph_data}

Rules:
- ONLY use information present in the graph data above
- Reference specific part numbers, procedure IDs, and fault codes
- Include safety warnings when safety protocol nodes are present
- Be concise and actionable
- If the data is insufficient, say so explicitly

Response:"""


class LLMClient:
    """Unified LLM client for G²CP agents.

    Supports OpenAI API and compatible endpoints.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                logger.warning("openai package not installed; LLM calls will use mock mode")

    def complete(self, prompt: str, system: str = "") -> str:
        """Send a completion request to the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The LLM's response text.
        """
        if self._client is None:
            return self._mock_complete(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._mock_complete(prompt, system)

    def extract_entities(self, query: str) -> list[str]:
        """Extract named entities from a user query (Algorithm 2, Step 1)."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        response = self.complete(prompt)
        try:
            data = json.loads(response.strip())
            return data.get("entities", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: split on commas or common delimiters
            logger.warning(f"Failed to parse entity extraction response: {response}")
            return [e.strip() for e in query.split(",") if e.strip()]

    def classify_intent(self, query: str) -> str:
        """Classify query intent (Algorithm 2, Step 3)."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        response = self.complete(prompt).strip().lower()
        valid_intents = {"diagnostic", "procedural", "predictive", "factoid"}
        if response in valid_intents:
            return response
        # Fallback heuristics
        q_lower = query.lower()
        if any(w in q_lower for w in ["why", "cause", "wrong", "diagnos"]):
            return "diagnostic"
        if any(w in q_lower for w in ["how", "fix", "repair", "procedure", "replace"]):
            return "procedural"
        if any(w in q_lower for w in ["predict", "risk", "next failure", "trend"]):
            return "predictive"
        return "factoid"

    def estimate_depth(self, query: str, n_entities: int, n_edge_types: int) -> int:
        """Estimate traversal hop depth (Algorithm 2, Step 5)."""
        prompt = DEPTH_ESTIMATION_PROMPT.format(
            query=query, n_entities=n_entities, n_edge_types=n_edge_types
        )
        response = self.complete(prompt).strip()
        try:
            depth = int(response)
            return max(1, min(3, depth))
        except ValueError:
            # Heuristic fallback
            if n_entities <= 1 and n_edge_types <= 2:
                return 1
            elif n_entities <= 3:
                return 2
            return 3

    def generate_response(self, query: str, subgraph_data: str) -> str:
        """Generate final natural language response from graph data."""
        prompt = RESPONSE_GENERATION_PROMPT.format(
            query=query, subgraph_data=subgraph_data
        )
        return self.complete(prompt)

    def _mock_complete(self, prompt: str, system: str = "") -> str:
        """Mock completion for testing without API access."""
        if "entities" in prompt.lower() and "json" in prompt.lower():
            # Mock entity extraction
            return '{"entities": ["mock_entity_1", "mock_entity_2"]}'
        if "classify" in prompt.lower() or "intent" in prompt.lower():
            return "diagnostic"
        if "depth" in prompt.lower() or "hops" in prompt.lower():
            return "2"
        return "Mock LLM response — configure API key for real completions."
