from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Case(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")


class GroundTruthRow(Base):
    __tablename__ = "ground_truth"

    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), primary_key=True
    )
    scenario_type: Mapped[str] = mapped_column(String(64))
    expected_outcome: Mapped[str] = mapped_column(String(32))
    expected_invoice_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_payment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_settlement_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_bank_txn_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class CaseResultRow(Base):
    __tablename__ = "case_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), index=True)
    case_id: Mapped[str] = mapped_column(String(64), ForeignKey("cases.case_id"), index=True)
    terminal_state: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_invoice_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    matched_payment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    matched_settlement_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    matched_bank_txn_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exception_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ai_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ai_classification: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ai_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_uncertainty_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_selected_candidate_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ExceptionRow(Base):
    __tablename__ = "exceptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), index=True)
    case_id: Mapped[str] = mapped_column(String(64), ForeignKey("cases.case_id"), index=True)
    reason_code: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text, default="")
    recommended_action: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), index=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stage: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recovery_action: Mapped[str | None] = mapped_column(String(128), nullable=True)
    candidate_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
