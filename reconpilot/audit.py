from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from reconpilot.db import session_scope
from reconpilot.models import AuditLog


def log_audit_event(
    *,
    run_id: str,
    case_id: str,
    stage: str,
    action: str,
    decision: str,
    reason: str,
    confidence: float | None = None,
    error_type: str | None = None,
    recovery_action: str | None = None,
    candidate_ids: Iterable[str] | None = None,
) -> AuditLog:
    """
    Persist one auditable reconciliation event.

    Every decision can therefore be traced to its stage,
    evidence, confidence, candidates, and recovery behavior.
    """

    candidate_ids_json = None

    if candidate_ids is not None:
        candidate_ids_json = json.dumps(list(candidate_ids))

    event = AuditLog(
        run_id=run_id,
        case_id=case_id,
        timestamp=datetime.now(timezone.utc),
        stage=stage,
        action=action,
        decision=decision,
        reason=reason,
        confidence=confidence,
        error_type=error_type,
        recovery_action=recovery_action,
        candidate_ids_json=candidate_ids_json,
    )

    with session_scope() as session:
        session.add(event)
        session.flush()
        session.refresh(event)

    return event


def get_case_audit(case_id: str) -> list[AuditLog]:
    """Return the complete audit history for one case."""

    with session_scope() as session:
        return (
            session.query(AuditLog)
            .filter(AuditLog.case_id == case_id)
            .order_by(AuditLog.timestamp.asc(), AuditLog.id.asc())
            .all()
        )


def get_run_audit(run_id: str) -> list[AuditLog]:
    """Return all audit events belonging to a reconciliation run."""

    with session_scope() as session:
        return (
            session.query(AuditLog)
            .filter(AuditLog.run_id == run_id)
            .order_by(AuditLog.timestamp.asc(), AuditLog.id.asc())
            .all()
        )