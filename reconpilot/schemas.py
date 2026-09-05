from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TerminalState(str, Enum):
    matched_deterministically = "matched_deterministically"
    matched_by_ai = "matched_by_ai"
    exception = "exception"
    failed_safely = "failed_safely"


class ExpectedOutcome(str, Enum):
    match = "match"
    exception = "exception"


class AuditStage(str, Enum):
    load = "load"
    deterministic = "deterministic"
    candidates = "candidates"
    llm = "llm"
    validate = "validate"
    persist = "persist"
    recover = "recover"


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str
    customer_name: str
    amount: Decimal = Field(gt=Decimal("0"))
    currency: str = "INR"
    issued_at: datetime
    reference: str | None = None


class Payment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payment_id: str
    invoice_id: str | None = None
    amount: Decimal | None = None
    currency: str = "INR"
    paid_at: datetime | None = None
    utr: str | None = None
    method: str | None = None
    status: str | None = None


class Settlement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settlement_id: str
    payment_id: str | None = None
    gross_amount: Decimal | None = None
    fee_amount: Decimal | None = None
    net_amount: Decimal | None = None
    currency: str = "INR"
    settled_at: datetime | None = None
    utr: str | None = None


class BankTxn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank_txn_id: str
    amount: Decimal | None = None
    currency: str = "INR"
    booked_at: datetime | None = None
    narration: str | None = None
    utr: str | None = None
    counterparty: str | None = None


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    scenario_type: str
    expected_outcome: ExpectedOutcome
    expected_invoice_id: str | None = None
    expected_payment_id: str | None = None
    expected_settlement_id: str | None = None
    expected_bank_txn_id: str | None = None


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    terminal_state: TerminalState
    reason: str
    confidence: float | None = None
    matched_invoice_id: str | None = None
    matched_payment_id: str | None = None
    matched_settlement_id: str | None = None
    matched_bank_txn_id: str | None = None
    exception_reason: str | None = None
    error_type: str | None = None


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    timestamp: datetime
    stage: AuditStage
    action: str
    decision: str
    reason: str
    confidence: float | None = None
    error_type: str | None = None
    recovery_action: str | None = None
    candidate_ids_json: str | None = None


TerminalStateLiteral = Literal[
    "matched_deterministically",
    "matched_by_ai",
    "exception",
    "failed_safely",
]
