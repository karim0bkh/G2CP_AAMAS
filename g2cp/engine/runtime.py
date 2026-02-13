"""G²CP Runtime — the central orchestrator that processes queries end-to-end.

Coordinates the full pipeline: user query → entity resolution → G²CP message
generation → agent execution → result aggregation → final response.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from g2cp.engine.audit import AuditLog
from g2cp.engine.executor import GraphExecutor
from g2cp.engine.security import SecurityManager
from g2cp.protocol.commitments import CommitmentStore
from g2cp.protocol.messages import (
    G2CPMessage,
    MessageContext,
    Performative,
    SubgraphResult,
    TraversalOperation,
)
from g2cp.protocol.operations import TraversalBuilder
from g2cp.utils.graph_db import GraphDB
from g2cp.utils.tokens import TokenCounter

logger = logging.getLogger(__name__)


class G2CPRuntime:
    """Main G²CP runtime engine.

    Orchestrates the multi-agent system:
    1. Receives user queries
    2. Resolves entities and builds G²CP operations
    3. Dispatches to specialist agents
    4. Collects and aggregates results
    5. Generates final response
    """

    def __init__(
        self,
        db: GraphDB,
        agents: dict[str, Any] | None = None,
        security: SecurityManager | None = None,
    ) -> None:
        self.db = db
        self.executor = GraphExecutor(db)
        self.security = security or SecurityManager()
        self.token_counter = TokenCounter()
        self.agents: dict[str, Any] = agents or {}
        self._audit_logs: dict[str, AuditLog] = {}

    def process_query(
        self,
        query: str,
        conversation_id: str = "",
    ) -> dict[str, Any]:
        """Process a user query through the G²CP pipeline.

        This is the main entry point. Implements the full flow from
        Figure 1 in the paper.

        Args:
            query: Natural language user query.
            conversation_id: Optional conversation ID for context tracking.

        Returns:
            Dict with: response, audit_log, metrics (tokens, time, messages).
        """
        start_time = time.time()
        conv_id = conversation_id or f"conv_{int(time.time())}"
        audit = AuditLog(conversation_id=conv_id)
        commitments = CommitmentStore()
        context = MessageContext(conversation_id=conv_id)

        collected_subgraphs: list[SubgraphResult] = []
        messages: list[G2CPMessage] = []

        try:
            # Phase 1: Dispatch to agents
            dispatcher = self.agents.get("dispatcher")
            if dispatcher:
                dispatch_result = dispatcher.handle_query(query, context)
                g2cp_messages = dispatch_result.get("messages", [])
            else:
                # Fallback: build operations directly
                g2cp_messages = self._build_default_operations(query, context)

            # Phase 2: Execute G²CP messages
            for msg in g2cp_messages:
                # Security check
                if not self.security.authorize(msg):
                    logger.warning(f"Unauthorized: {msg.sender} → {msg.serialize()}")
                    continue

                # Sign message
                msg.signature = self.security.sign_message(msg)

                # Track commitment
                commitments.create_from_message(msg)

                # Execute operation
                result = self._execute_message(msg)

                # Count tokens
                tokens = self.token_counter.count_message(msg)

                # Audit
                audit.append(msg, result, token_count=tokens)
                messages.append(msg)

                if result and not result.is_empty:
                    collected_subgraphs.append(result)

                # Route to target agent for further processing
                target_agent = self.agents.get(msg.receiver)
                if target_agent and msg.performative == Performative.REQUEST:
                    agent_result = target_agent.handle_message(msg, result)
                    if agent_result:
                        for response_msg in agent_result.get("messages", []):
                            resp_result = self._execute_message(response_msg)
                            resp_tokens = self.token_counter.count_message(response_msg)
                            audit.append(response_msg, resp_result, token_count=resp_tokens)
                            messages.append(response_msg)
                            if resp_result and not resp_result.is_empty:
                                collected_subgraphs.append(resp_result)

            # Phase 3: Generate response
            response = self._generate_response(query, collected_subgraphs)

        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
            response = f"Error processing query: {e}"

        elapsed = time.time() - start_time
        self._audit_logs[conv_id] = audit

        return {
            "response": response,
            "conversation_id": conv_id,
            "audit_log": audit,
            "metrics": {
                "total_tokens": audit.total_tokens,
                "total_time_s": round(elapsed, 3),
                "n_messages": audit.message_count,
                "n_subgraphs": len(collected_subgraphs),
                "commitments": {
                    "total": len(commitments.all),
                    "active": len(commitments.active),
                    "violated": len(commitments.violated),
                },
            },
        }

    def _execute_message(self, msg: G2CPMessage) -> SubgraphResult | None:
        """Execute the graph operation in a G²CP message."""
        if msg.operation is None:
            return None

        try:
            if isinstance(msg.operation, TraversalOperation):
                return self.executor.execute_traversal(msg.operation)
            else:
                return self.executor.execute_update(msg.operation)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return SubgraphResult(metadata={"error": str(e)})

    def _build_default_operations(
        self, query: str, context: MessageContext
    ) -> list[G2CPMessage]:
        """Build default G²CP operations when no dispatcher agent is configured.

        Uses simple heuristics for demonstration/testing.
        """
        # Simple heuristic: extract keywords, build traversal
        keywords = [w for w in query.lower().split() if len(w) > 3]

        op = (
            TraversalBuilder()
            .from_type("Component", "Fault", "Symptom")
            .via(["causes", "indicates", "addressed_by"])
            .depth(2)
            .return_subgraph()
            .build()
        )

        msg = G2CPMessage(
            sender="dispatcher",
            receiver="diagnostic",
            performative=Performative.REQUEST,
            operation=op,
            context=context,
        )
        return [msg]

    def _generate_response(
        self, query: str, subgraphs: list[SubgraphResult]
    ) -> str:
        """Generate final natural language response from collected subgraphs."""
        if not subgraphs:
            return "No relevant information found in the knowledge graph."

        # Aggregate subgraph data
        all_nodes = []
        all_edges = []
        for sg in subgraphs:
            all_nodes.extend(sg.nodes)
            all_edges.extend(sg.edges)

        # Try LLM-based generation
        dispatcher = self.agents.get("dispatcher")
        if dispatcher and hasattr(dispatcher, "generate_response"):
            return dispatcher.generate_response(query, all_nodes, all_edges)

        # Fallback: structured summary
        lines = [f"Found {len(all_nodes)} relevant entities and {len(all_edges)} relationships."]
        for node in all_nodes[:10]:
            uid = node.get("uid", "unknown")
            name = node.get("name", uid)
            ntype = node.get("type", "Entity")
            lines.append(f"  - [{ntype}] {name}")

        return "\n".join(lines)

    def get_audit_log(self, conversation_id: str) -> AuditLog | None:
        """Retrieve the audit log for a conversation."""
        return self._audit_logs.get(conversation_id)
