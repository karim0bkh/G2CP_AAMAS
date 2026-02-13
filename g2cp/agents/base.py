"""Base agent class for G²CP multi-agent system.

All specialized agents inherit from BaseG2CPAgent, which provides
common functionality: message handling, operation execution, LLM interface,
and audit integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from g2cp.engine.executor import GraphExecutor
from g2cp.protocol.messages import (
    G2CPMessage,
    MessageContext,
    Performative,
    SubgraphResult,
    TraversalOperation,
)
from g2cp.protocol.operations import TraversalBuilder
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class BaseG2CPAgent(ABC):
    """Abstract base class for G²CP agents.

    Each agent maintains:
    - Local graph cache (materialized view optimized for its edge specialization)
    - Message queue (pending messages with priority ordering)
    - Execution engine (Cypher-based traversal executor)
    - LLM interface (only for parsing queries and formatting responses)
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        db: GraphDB,
        llm: LLMClient,
        specialized_edge_types: set[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.db = db
        self.llm = llm
        self.executor = GraphExecutor(db)
        self.specialized_edge_types = specialized_edge_types or set()
        self._message_queue: list[G2CPMessage] = []

    @abstractmethod
    def handle_message(
        self, msg: G2CPMessage, prior_result: SubgraphResult | None = None
    ) -> dict[str, Any]:
        """Handle an incoming G²CP message.

        Args:
            msg: The incoming G²CP message.
            prior_result: Result from executing the message's operation (if any).

        Returns:
            Dict with 'messages' (list of response G²CP messages) and
            optional 'result' (SubgraphResult).
        """

    def execute_operation(self, op: TraversalOperation) -> SubgraphResult:
        """Execute a traversal operation using the graph executor."""
        return self.executor.execute_traversal(op)

    def create_message(
        self,
        receiver: str,
        performative: Performative,
        operation: Any = None,
        result: SubgraphResult | None = None,
        context: MessageContext | None = None,
        error: str | None = None,
    ) -> G2CPMessage:
        """Create a G²CP message from this agent."""
        return G2CPMessage(
            sender=self.agent_id,
            receiver=receiver,
            performative=performative,
            operation=operation,
            result=result,
            context=context or MessageContext(),
            error=error,
        )

    def create_inform(
        self,
        receiver: str,
        result: SubgraphResult,
        context: MessageContext | None = None,
    ) -> G2CPMessage:
        """Convenience: create an INFORM message with results."""
        return self.create_message(
            receiver=receiver,
            performative=Performative.INFORM,
            result=result,
            context=context,
        )

    def create_request(
        self,
        receiver: str,
        operation: TraversalOperation,
        context: MessageContext | None = None,
    ) -> G2CPMessage:
        """Convenience: create a REQUEST message with an operation."""
        return self.create_message(
            receiver=receiver,
            performative=Performative.REQUEST,
            operation=operation,
            context=context,
        )
