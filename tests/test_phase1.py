from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect

from reconpilot import __version__
from reconpilot.db import REQUIRED_TABLES, init_db
from reconpilot.schemas import (
    AuditEvent,
    AuditStage,
    BankTxn,
    CaseResult,
    GroundTruth,
    Invoice,
    Payment,
    Settlement,
    TerminalState,
)


def test_package_import():
    assert __version__
    import reconpilot.config  # noqa: F401
    import reconpilot.db  # noqa: F401
    import reconpilot.models  # noqa: F401
    import reconpilot.schemas  # noqa: F401


def test_invoice_payment_settlement_bank_valid():
    issued = datetime(2026, 1, 15, tzinfo=timezone.utc)
    invoice = Invoice(
        invoice_id="inv_1",
        customer_name="Acme",
        amount=Decimal("1000.00"),
        issued_at=issued,
        reference="INV-1",
    )
    payment = Payment(
        payment_id="pay_1",
        invoice_id="inv_1",
        amount=Decimal("1000.00"),
        paid_at=issued,
        utr="UTR1",
    )
    settlement = Settlement(
        settlement_id="set_1",
        payment_id="pay_1",
        gross_amount=Decimal("1000.00"),
        fee_amount=Decimal("20.00"),
        net_amount=Decimal("980.00"),
        settled_at=issued,
        utr="UTR1",
    )
    bank = BankTxn(
        bank_txn_id="bnk_1",
        amount=Decimal("980.00"),
        booked_at=issued,
        narration="UTR1 Acme",
        utr="UTR1",
    )
    assert invoice.invoice_id == "inv_1"
    assert payment.payment_id == "pay_1"
    assert settlement.net_amount == Decimal("980.00")
    assert bank.utr == "UTR1"


def test_invoice_rejects_non_positive_amount():
    with pytest.raises(ValidationError):
        Invoice(
            invoice_id="inv_bad",
            customer_name="Acme",
            amount=Decimal("0"),
            issued_at=datetime.now(timezone.utc),
        )


def test_ground_truth_and_case_result_terminal_states():
    gt = GroundTruth(
        case_id="c1",
        scenario_type="exact_match",
        expected_outcome="match",
        expected_invoice_id="inv_1",
        expected_payment_id="pay_1",
    )
    assert gt.scenario_type == "exact_match"
    for state in TerminalState:
        result = CaseResult(
            run_id="run-1",
            case_id="c1",
            terminal_state=state,
            reason="test",
            confidence=0.9,
        )
        assert result.terminal_state == state
    assert {s.value for s in TerminalState} == {
        "matched_deterministically",
        "matched_by_ai",
        "exception",
        "failed_safely",
    }


def test_case_result_rejects_unknown_terminal_state():
    with pytest.raises(ValidationError):
        CaseResult(
            run_id="run-1",
            case_id="c1",
            terminal_state="auto_match",
            reason="no",
        )


def test_audit_event_required_fields():
    event = AuditEvent(
        run_id="run-1",
        case_id="c1",
        timestamp=datetime.now(timezone.utc),
        stage=AuditStage.load,
        action="load_case",
        decision="continue",
        reason="loaded",
        confidence=None,
        error_type=None,
        recovery_action=None,
        candidate_ids_json=None,
    )
    dumped = event.model_dump()
    for key in (
        "run_id",
        "case_id",
        "timestamp",
        "stage",
        "action",
        "decision",
        "reason",
        "confidence",
        "error_type",
        "recovery_action",
        "candidate_ids_json",
    ):
        assert key in dumped


def test_init_db_creates_required_tables(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = init_db(url)
    tables = set(inspect(engine).get_table_names())
    assert set(REQUIRED_TABLES).issubset(tables)
    audit_cols = {c["name"] for c in inspect(engine).get_columns("audit_log")}
    for col in (
        "run_id",
        "case_id",
        "timestamp",
        "stage",
        "action",
        "decision",
        "reason",
        "confidence",
        "error_type",
        "recovery_action",
        "candidate_ids_json",
    ):
        assert col in audit_cols
    engine.dispose()
