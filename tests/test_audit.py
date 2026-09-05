from reconpilot.audit import (
    get_case_audit,
    get_run_audit,
    log_audit_event,
)


def test_log_and_read_audit_event():
    event = log_audit_event(
        run_id="run-test-001",
        case_id="case-test-001",
        stage="deterministic",
        action="match_invoice_payment",
        decision="matched",
        reason="exact invoice ID and amount",
        confidence=1.0,
        candidate_ids=["PAY-001"],
    )

    assert event.run_id == "run-test-001"
    assert event.case_id == "case-test-001"
    assert event.stage == "deterministic"
    assert event.decision == "matched"
    assert event.confidence == 1.0
    assert event.candidate_ids_json == '["PAY-001"]'


def test_case_audit_returns_events():
    log_audit_event(
        run_id="run-test-002",
        case_id="case-test-002",
        stage="candidates",
        action="generate_candidates",
        decision="deferred",
        reason="multiple candidates",
        candidate_ids=["PAY-001", "PAY-002"],
    )

    events = get_case_audit("case-test-002")

    assert len(events) >= 1
    assert events[0].case_id == "case-test-002"
    assert events[0].stage == "candidates"


def test_run_audit_returns_events():
    log_audit_event(
        run_id="run-test-003",
        case_id="case-test-003",
        stage="validate",
        action="validate_ai_result",
        decision="blocked",
        reason="conflicting deterministic evidence",
        confidence=0.97,
        error_type="CONFLICTING_EVIDENCE",
    )

    events = get_run_audit("run-test-003")

    assert len(events) >= 1
    assert events[0].run_id == "run-test-003"
    assert events[0].decision == "blocked"
    assert events[0].error_type == "CONFLICTING_EVIDENCE"