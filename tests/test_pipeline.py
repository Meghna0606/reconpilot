from __future__ import annotations

import json

import pytest

import reconpilot.pipeline as pipeline
from reconpilot.ai_investigator import AIInvestigation, AIInvestigationResult
from reconpilot.db import init_db
from reconpilot.models import (
    AuditLog,
    CaseResultRow,
    ExceptionRow,
    GroundTruthRow,
    Run,
)
from reconpilot.schemas import Invoice, Payment, Settlement, BankTxn


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """
    Give each integration test its own SQLite database.
    """

    db_path = tmp_path / "integration.db"
    url = f"sqlite:///{db_path}"

    engine = init_db(url)

    # pipeline.session_scope ultimately uses the global DB session.
    # Rebind it to the temporary database.
    import reconpilot.db as db

    from sqlalchemy.orm import sessionmaker

    test_session = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        future=True,
    )

    monkeypatch.setattr(db, "_SessionLocal", test_session)
    monkeypatch.setattr(pipeline, "init_db", lambda: engine)

    yield engine

    engine.dispose()


def _load_case_payload(engine, case_id: str) -> dict:
    import reconpilot.db as db

    with db.session_scope() as session:
        case = session.query(pipeline.Case).filter(
            pipeline.Case.case_id == case_id
        ).first()

        assert case is not None

        return json.loads(case.payload_json)


def _assert_complete_audit_trail(
    engine,
    run_id: str,
    case_id: str,
):
    import reconpilot.db as db

    with db.session_scope() as session:
        events = (
            session.query(AuditLog)
            .filter(
                AuditLog.run_id == run_id,
                AuditLog.case_id == case_id,
            )
            .all()
        )

        assert events, "Every processed case must have audit events."


def test_full_generated_batch_runs_end_to_end(
    isolated_db,
    monkeypatch,
):
    """
    Generate the complete synthetic dataset and run the actual
    reconciliation pipeline.

    The LLM is mocked so this test never makes an API request.
    """

    def fake_llm(target, candidates):
        # The generated ambiguous case should select the real bank
        # transaction supplied as the candidate.
        selected = candidates[0].candidate_id if candidates else None

        return AIInvestigationResult(
            success=True,
            investigation=AIInvestigation(
                classification="LIKELY_MATCH",
                selected_candidate_id=selected,
                confidence=0.95,
                evidence=["candidate evidence supplied by pipeline"],
                recommended_action="AUTO_RESOLVE",
                uncertainty=[],
            ),
        )

    monkeypatch.setattr(
        "reconpilot.ai_investigator.investigate_with_llm",
        fake_llm,
    )

    run_id, metrics = pipeline.run_reconciliation(
        generate_demo=True,
        seed=42,
    )

    assert run_id
    assert metrics.total_cases >= 50
    assert metrics.total_source_records == 408

    # The pipeline must actually produce terminal outcomes.
    assert (
        metrics.deterministic_resolutions
        + metrics.ai_assisted_resolutions
        + metrics.unresolved_exceptions
        == metrics.total_cases
    )

    assert metrics.throughput_cases_per_second > 0

    # Safety invariant: integration must never produce a false
    # automatic match.
    assert metrics.false_auto_match_count == 0

    import reconpilot.db as db

    with db.session_scope() as session:
        run = session.query(Run).filter(
            Run.run_id == run_id
        ).first()

        assert run is not None
        assert run.status == "completed"
        assert run.finished_at is not None

        result_count = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id
        ).count()

        assert result_count == metrics.total_cases


def test_exact_match_resolves_deterministically(
    isolated_db,
):
    """
    An obvious exact match must never require AI.
    """

    from reconpilot.generate import scenario_exact_match

    case_id = "CASE-INTEGRATION-EXACT"

    payload, gt = scenario_exact_match(case_id)

    import reconpilot.db as db
    from reconpilot.models import Case

    with db.session_scope() as session:
        session.add(
            Case(
                case_id=case_id,
                payload_json=json.dumps(payload, default=str),
            )
        )
        session.add(
            GroundTruthRow(
                case_id=case_id,
                scenario_type=gt["scenario_type"],
                expected_outcome=gt["expected_outcome"],
                expected_invoice_id=gt["expected_invoice_id"],
                expected_payment_id=gt["expected_payment_id"],
                expected_settlement_id=gt["expected_settlement_id"],
                expected_bank_txn_id=gt["expected_bank_txn_id"],
            )
        )

    run_id, metrics = pipeline.run_reconciliation()

    assert metrics.total_cases == 1
    assert metrics.deterministic_resolutions == 1
    assert metrics.ai_assisted_resolutions == 0
    assert metrics.unresolved_exceptions == 0
    assert metrics.overall_accuracy == 1.0

    with db.session_scope() as session:
        result = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id,
            CaseResultRow.case_id == case_id,
        ).first()

        assert result is not None
        assert result.terminal_state == "matched_deterministically"
        assert result.matched_invoice_id == gt["expected_invoice_id"]
        assert result.matched_payment_id == gt["expected_payment_id"]
        assert result.matched_settlement_id == gt["expected_settlement_id"]
        assert result.matched_bank_txn_id == gt["expected_bank_txn_id"]

    _assert_complete_audit_trail(
        isolated_db,
        run_id,
        case_id,
    )


def test_no_valid_match_becomes_safe_exception(
    isolated_db,
):
    """
    A case with no meaningful candidate must terminate as an exception.
    """

    from reconpilot.generate import scenario_no_valid_match

    case_id = "CASE-INTEGRATION-NONE"

    payload, gt = scenario_no_valid_match(case_id)

    import reconpilot.db as db
    from reconpilot.models import Case

    with db.session_scope() as session:
        session.add(
            Case(
                case_id=case_id,
                payload_json=json.dumps(payload, default=str),
            )
        )
        session.add(
            GroundTruthRow(
                case_id=case_id,
                scenario_type=gt["scenario_type"],
                expected_outcome=gt["expected_outcome"],
                expected_invoice_id=gt["expected_invoice_id"],
                expected_payment_id=gt["expected_payment_id"],
                expected_settlement_id=gt["expected_settlement_id"],
                expected_bank_txn_id=gt["expected_bank_txn_id"],
            )
        )

    run_id, metrics = pipeline.run_reconciliation()

    assert metrics.total_cases == 1
    assert metrics.unresolved_exceptions == 1
    assert metrics.deterministic_resolutions == 0
    assert metrics.ai_assisted_resolutions == 0

    import reconpilot.db as db

    with db.session_scope() as session:
        result = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id,
            CaseResultRow.case_id == case_id,
        ).first()

        assert result is not None
        assert result.terminal_state == "exception"
        assert result.exception_reason == "NO_CANDIDATES"

        exception = session.query(ExceptionRow).filter(
            ExceptionRow.run_id == run_id,
            ExceptionRow.case_id == case_id,
        ).first()

        assert exception is not None
        assert exception.reason_code == "NO_CANDIDATES"


def test_llm_failure_fails_safely(
    isolated_db,
    monkeypatch,
):
    """
    A failed AI investigation must not crash the reconciliation run.
    """

    from reconpilot.generate import scenario_ambiguous_match

    case_id = "CASE-INTEGRATION-LLM-FAIL"

    payload, gt = scenario_ambiguous_match(case_id)

    import reconpilot.db as db
    from reconpilot.models import Case

    with db.session_scope() as session:
        session.add(
            Case(
                case_id=case_id,
                payload_json=json.dumps(payload, default=str),
            )
        )
        session.add(
            GroundTruthRow(
                case_id=case_id,
                scenario_type=gt["scenario_type"],
                expected_outcome=gt["expected_outcome"],
                expected_invoice_id=gt["expected_invoice_id"],
                expected_payment_id=gt["expected_payment_id"],
                expected_settlement_id=gt["expected_settlement_id"],
                expected_bank_txn_id=gt["expected_bank_txn_id"],
            )
        )

    def failing_llm(target, candidates):
        from reconpilot.ai_investigator import AIFailure

        return AIInvestigationResult(
            success=False,
            failure=AIFailure(
                failure_type="LLM_FAILURE",
                detail="simulated LLM outage",
                retryable=True,
            ),
        )

    monkeypatch.setattr(
        "reconpilot.ai_investigator.investigate_with_llm",
        failing_llm,
    )

    run_id, metrics = pipeline.run_reconciliation()

    assert metrics.total_cases == 1
    assert metrics.unresolved_exceptions == 1
    assert metrics.ai_assisted_resolutions == 0

    import reconpilot.db as db

    with db.session_scope() as session:
        result = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id,
            CaseResultRow.case_id == case_id,
        ).first()

        assert result is not None
        assert result.terminal_state == "failed_safely"
        assert result.error_type == "LLM_FAILURE"

        exception = session.query(ExceptionRow).filter(
            ExceptionRow.run_id == run_id,
            ExceptionRow.case_id == case_id,
        ).first()

        assert exception is not None
        assert exception.reason_code == "LLM_FAILURE"


def test_ai_high_confidence_cannot_bypass_deterministic_validation(
    isolated_db,
    monkeypatch,
):
    """
    Critical safety integration test:

    Even if AI returns 0.99 confidence and a candidate,
    invalid deterministic evidence must prevent AUTO_RESOLVE.
    """

    from reconpilot.generate import scenario_ambiguous_match

    case_id = "CASE-INTEGRATION-AI-SAFETY"

    payload, gt = scenario_ambiguous_match(case_id)

    import reconpilot.db as db
    from reconpilot.models import Case

    with db.session_scope() as session:
        session.add(
            Case(
                case_id=case_id,
                payload_json=json.dumps(payload, default=str),
            )
        )
        session.add(
            GroundTruthRow(
                case_id=case_id,
                scenario_type=gt["scenario_type"],
                expected_outcome=gt["expected_outcome"],
                expected_invoice_id=gt["expected_invoice_id"],
                expected_payment_id=gt["expected_payment_id"],
                expected_settlement_id=gt["expected_settlement_id"],
                expected_bank_txn_id=gt["expected_bank_txn_id"],
            )
        )

    def overconfident_llm(target, candidates):
        assert candidates

        # Deliberately bypass the normal parser and return a fabricated
        # candidate ID. The pipeline itself must still fail closed.
        return AIInvestigationResult(
            success=True,
            investigation=AIInvestigation(
                classification="LIKELY_MATCH",
                selected_candidate_id="BANK-FABRICATED-NOT-SUPPLIED",
                confidence=0.99,
                evidence=["simulated strong AI confidence"],
                recommended_action="AUTO_RESOLVE",
                uncertainty=[],
            ),
        )

    monkeypatch.setattr(
        "reconpilot.ai_investigator.investigate_with_llm",
        overconfident_llm,
    )

    run_id, metrics = pipeline.run_reconciliation()

    # The AI must NOT be allowed to force a match.
    assert metrics.ai_assisted_resolutions == 0
    assert metrics.unresolved_exceptions == 1
    assert metrics.false_auto_match_count == 0

    import reconpilot.db as db

    with db.session_scope() as session:
        result = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id,
            CaseResultRow.case_id == case_id,
        ).first()

        assert result is not None
        assert result.terminal_state == "exception"
        assert result.exception_reason == "AI_SAFETY_BLOCK"

def test_valid_ai_selected_bank_candidate_can_auto_resolve(
    isolated_db,
    monkeypatch,
):
    from reconpilot.generate import scenario_ambiguous_match

    case_id = "CASE-INTEGRATION-AI-BANK-VALID"
    payload, gt = scenario_ambiguous_match(case_id)

    import reconpilot.db as db
    from reconpilot.models import Case

    with db.session_scope() as session:
        session.add(Case(case_id=case_id, payload_json=json.dumps(payload, default=str)))
        session.add(GroundTruthRow(
            case_id=case_id,
            scenario_type=gt["scenario_type"],
            expected_outcome=gt["expected_outcome"],
            expected_invoice_id=gt["expected_invoice_id"],
            expected_payment_id=gt["expected_payment_id"],
            expected_settlement_id=gt["expected_settlement_id"],
            expected_bank_txn_id=gt["expected_bank_txn_id"],
        ))

    def fake_llm(target, candidates):
        assert candidates
        return AIInvestigationResult(
            success=True,
            investigation=AIInvestigation(
                classification="LIKELY_MATCH",
                selected_candidate_id=candidates[0].candidate_id,
                confidence=0.95,
                evidence=["candidate has independent deterministic support"],
                recommended_action="AUTO_RESOLVE",
                uncertainty=[],
            ),
        )

    monkeypatch.setattr("reconpilot.ai_investigator.investigate_with_llm", fake_llm)

    run_id, metrics = pipeline.run_reconciliation()

    assert metrics.total_cases == 1
    assert metrics.ai_assisted_resolutions == 1
    assert metrics.false_auto_match_count == 0
    assert metrics.overall_accuracy == 1.0

    with db.session_scope() as session:
        result = session.query(CaseResultRow).filter(
            CaseResultRow.run_id == run_id,
            CaseResultRow.case_id == case_id,
        ).first()
        assert result is not None
        assert result.terminal_state == "matched_by_ai"
        assert result.matched_bank_txn_id == gt["expected_bank_txn_id"]
