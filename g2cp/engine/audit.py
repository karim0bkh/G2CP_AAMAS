"""Audit logging for G²CP message traces.

Implements the auditability guarantee (Theorem 2): any agent conclusion
can be verified by replaying the message sequence over the graph.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from g2cp.protocol.messages import G2CPMessage, SubgraphResult
from g2cp.protocol.serializer import message_to_dict


@dataclass
class AuditEntry:
    """Single entry in the audit log."""

    timestamp: float
    message: dict[str, Any]
    result: Optional[dict[str, Any]]
    execution_time_s: float
    token_count: int = 0


class AuditLog:
    """Append-only audit log for G²CP conversations.

    Stores complete message traces enabling:
    - Replay verification (Theorem 2)
    - Non-hallucination checking (Theorem 6)
    - Performance analysis
    """

    def __init__(self, conversation_id: str = "") -> None:
        self.conversation_id = conversation_id
        self._entries: list[AuditEntry] = []
        self._start_time: float = time.time()

    def append(
        self,
        message: G2CPMessage,
        result: SubgraphResult | None = None,
        execution_time_s: float = 0.0,
        token_count: int = 0,
    ) -> None:
        """Append a message and its result to the audit log."""
        entry = AuditEntry(
            timestamp=time.time(),
            message=message_to_dict(message),
            result=result.model_dump() if result else None,
            execution_time_s=execution_time_s,
            token_count=token_count,
        )
        self._entries.append(entry)

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self._entries)

    @property
    def total_time_s(self) -> float:
        return time.time() - self._start_time

    @property
    def message_count(self) -> int:
        return len(self._entries)

    def get_trace(self) -> list[dict[str, Any]]:
        """Get the full message trace for replay verification."""
        return [
            {
                "timestamp": e.timestamp,
                "message": e.message,
                "result": e.result,
            }
            for e in self._entries
        ]

    def verify_trace(self) -> dict[str, Any]:
        """Verify the audit trace for consistency.

        Checks:
        1. Timestamps are monotonically increasing
        2. Conversation IDs are consistent
        3. REQUEST messages have corresponding INFORM responses
        """
        issues = []

        # Check timestamp ordering
        for i in range(1, len(self._entries)):
            if self._entries[i].timestamp < self._entries[i - 1].timestamp:
                issues.append(f"Non-monotonic timestamp at entry {i}")

        # Check conversation consistency
        conv_ids = set()
        for e in self._entries:
            cid = e.message.get("context", {}).get("conversation_id", "")
            if cid:
                conv_ids.add(cid)
        if len(conv_ids) > 1:
            issues.append(f"Multiple conversation IDs: {conv_ids}")

        # Check REQUEST/INFORM pairing
        pending_requests: dict[str, int] = {}
        for i, e in enumerate(self._entries):
            perf = e.message.get("performative", "")
            sender = e.message.get("sender", "")
            receiver = e.message.get("receiver", "")
            if perf == "REQUEST":
                key = f"{sender}->{receiver}"
                pending_requests[key] = i
            elif perf == "INFORM":
                key = f"{receiver}->{sender}"
                if key in pending_requests:
                    del pending_requests[key]
                # Also try matching by reverse direction
                alt_key = f"{sender}->{receiver}"
                pending_requests.pop(alt_key, None)

        for key, idx in pending_requests.items():
            issues.append(f"Unfulfilled REQUEST at entry {idx}: {key}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_entries": len(self._entries),
            "total_tokens": self.total_tokens,
        }

    def save(self, path: str | Path) -> None:
        """Save audit log to JSON file."""
        data = {
            "conversation_id": self.conversation_id,
            "start_time": self._start_time,
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "message": e.message,
                    "result": e.result,
                    "execution_time_s": e.execution_time_s,
                    "token_count": e.token_count,
                }
                for e in self._entries
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> AuditLog:
        """Load audit log from JSON file."""
        data = json.loads(Path(path).read_text())
        log = cls(conversation_id=data.get("conversation_id", ""))
        log._start_time = data.get("start_time", time.time())
        for entry_data in data.get("entries", []):
            log._entries.append(AuditEntry(
                timestamp=entry_data["timestamp"],
                message=entry_data["message"],
                result=entry_data.get("result"),
                execution_time_s=entry_data.get("execution_time_s", 0),
                token_count=entry_data.get("token_count", 0),
            ))
        return log
