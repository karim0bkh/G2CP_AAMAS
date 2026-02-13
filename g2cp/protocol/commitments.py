"""Social commitment semantics for G²CP performatives.

Implements Section 3.6 of the paper: each performative creates verifiable
social commitments C(x, y, p) where agent x commits to agent y that p holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from g2cp.protocol.messages import G2CPMessage, Performative


class CommitmentState(str, Enum):
    """Lifecycle states for social commitments."""

    CREATED = "created"
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    VIOLATED = "violated"
    CANCELLED = "cancelled"


@dataclass
class SocialCommitment:
    """A social commitment C(debtor, creditor, condition).

    Attributes:
        debtor: Agent who must satisfy the commitment.
        creditor: Agent to whom the commitment is owed.
        condition: What must hold for the commitment to be fulfilled.
        state: Current lifecycle state.
        source_message_id: The G²CP message that created this commitment.
        created_at: Timestamp of creation.
        fulfilled_at: Timestamp of fulfillment (if any).
    """

    debtor: str
    creditor: str
    condition: str
    state: CommitmentState = CommitmentState.CREATED
    source_message_id: str = ""
    created_at: float = field(default_factory=time.time)
    fulfilled_at: Optional[float] = None

    def fulfill(self) -> None:
        self.state = CommitmentState.FULFILLED
        self.fulfilled_at = time.time()

    def violate(self) -> None:
        self.state = CommitmentState.VIOLATED

    def cancel(self) -> None:
        self.state = CommitmentState.CANCELLED


class CommitmentStore:
    """Tracks all social commitments in a G²CP conversation.

    Commitments are publicly observable through the audit log,
    satisfying Singh's (1998) criterion that meaning should be
    grounded in social facts rather than private mental states.
    """

    def __init__(self) -> None:
        self._commitments: list[SocialCommitment] = []

    @property
    def all(self) -> list[SocialCommitment]:
        return list(self._commitments)

    @property
    def active(self) -> list[SocialCommitment]:
        return [c for c in self._commitments if c.state == CommitmentState.ACTIVE]

    @property
    def violated(self) -> list[SocialCommitment]:
        return [c for c in self._commitments if c.state == CommitmentState.VIOLATED]

    def create_from_message(self, msg: G2CPMessage) -> SocialCommitment | None:
        """Create social commitment(s) from a G²CP message.

        Implements the commitment semantics defined in Section 3.6:
        - REQUEST: C(receiver, sender, execute_and_return(op))
        - INFORM: C(sender, receiver, grounded(subgraph, G))
        - QUERY: C(receiver, sender, truthful_response(op))
        - PROPOSE: C(receiver, sender, evaluate_and_respond(op))
        - CONFIRM: C(sender, receiver, verified(result))
        - REJECT: C(sender, receiver, violated(op, constraint))
        - UPDATE: C(receiver, sender, apply_if_valid(ΔG))
        """
        commitment = None
        op_desc = msg.operation.op_type if msg.operation else "none"

        match msg.performative:
            case Performative.REQUEST:
                commitment = SocialCommitment(
                    debtor=msg.receiver,
                    creditor=msg.sender,
                    condition=f"execute_and_return({op_desc})",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )
            case Performative.INFORM:
                commitment = SocialCommitment(
                    debtor=msg.sender,
                    creditor=msg.receiver,
                    condition=f"grounded(subgraph, G)",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )
            case Performative.QUERY:
                commitment = SocialCommitment(
                    debtor=msg.receiver,
                    creditor=msg.sender,
                    condition=f"truthful_response({op_desc})",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )
            case Performative.PROPOSE:
                commitment = SocialCommitment(
                    debtor=msg.receiver,
                    creditor=msg.sender,
                    condition=f"evaluate_and_respond({op_desc})",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )
            case Performative.CONFIRM:
                commitment = SocialCommitment(
                    debtor=msg.sender,
                    creditor=msg.receiver,
                    condition="verified(result)",
                    state=CommitmentState.FULFILLED,
                    source_message_id=msg.id,
                )
                # Fulfill the matching REQUEST commitment
                self._fulfill_matching(msg.receiver, msg.sender, "execute_and_return")
            case Performative.REJECT:
                commitment = SocialCommitment(
                    debtor=msg.sender,
                    creditor=msg.receiver,
                    condition="violated(op, constraint)",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )
            case Performative.UPDATE:
                commitment = SocialCommitment(
                    debtor=msg.receiver,
                    creditor=msg.sender,
                    condition=f"apply_if_valid(delta_G)",
                    state=CommitmentState.ACTIVE,
                    source_message_id=msg.id,
                )

        if commitment:
            self._commitments.append(commitment)
        return commitment

    def fulfill_commitment(self, debtor: str, creditor: str, condition_prefix: str) -> bool:
        """Mark a matching commitment as fulfilled."""
        return self._fulfill_matching(debtor, creditor, condition_prefix)

    def _fulfill_matching(self, debtor: str, creditor: str, condition_prefix: str) -> bool:
        """Find and fulfill the first matching active commitment."""
        for c in self._commitments:
            if (
                c.debtor == debtor
                and c.creditor == creditor
                and c.condition.startswith(condition_prefix)
                and c.state == CommitmentState.ACTIVE
            ):
                c.fulfill()
                return True
        return False

    def check_violations(self, timeout_seconds: float = 30.0) -> list[SocialCommitment]:
        """Check for timed-out commitments and mark them as violated."""
        now = time.time()
        violations = []
        for c in self._commitments:
            if c.state == CommitmentState.ACTIVE and (now - c.created_at) > timeout_seconds:
                c.violate()
                violations.append(c)
        return violations
