from reconpilot.ai_investigator import AIInvestigation
from reconpilot.decision_gate import apply_decision_gate


def make_investigation(
    classification="LIKELY_MATCH",
    candidate_id="PAY-001",
    confidence=0.95,
):
    return AIInvestigation(
        classification=classification,
        selected_candidate_id=candidate_id,
        confidence=confidence,
        evidence=["strong amount agreement"],
        recommended_action="AUTO_RESOLVE",
        uncertainty=[],
    )


def test_high_confidence_valid_match_is_auto_resolved():
    investigation = make_investigation(confidence=0.95)

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "AUTO_RESOLVE"
    assert result.selected_candidate_id == "PAY-001"


def test_high_confidence_but_invalid_deterministic_evidence_is_not_auto_resolved():
    investigation = make_investigation(confidence=0.97)

    result = apply_decision_gate(
        investigation,
        deterministic_valid=False,
    )

    assert result.decision == "SUGGEST_MATCH"
    assert result.decision != "AUTO_RESOLVE"


def test_conflicting_candidate_blocks_auto_resolution():
    investigation = make_investigation(confidence=0.98)

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
        conflicting_candidate=True,
    )

    assert result.decision == "SUGGEST_MATCH"
    assert result.decision != "AUTO_RESOLVE"


def test_medium_confidence_requires_human_review():
    investigation = make_investigation(confidence=0.80)

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "SUGGEST_MATCH"


def test_low_confidence_becomes_exception():
    investigation = make_investigation(confidence=0.50)

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "EXCEPTION"


def test_ambiguous_classification_becomes_exception():
    investigation = make_investigation(
        classification="AMBIGUOUS",
        candidate_id=None,
        confidence=0.95,
    )

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "EXCEPTION"


def test_partial_payment_cannot_be_auto_resolved():
    investigation = make_investigation(
        classification="PARTIAL_PAYMENT",
        candidate_id=None,
        confidence=0.95,
    )

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "EXCEPTION"


def test_match_without_candidate_becomes_exception():
    investigation = make_investigation(
        classification="LIKELY_MATCH",
        candidate_id=None,
        confidence=0.95,
    )

    result = apply_decision_gate(
        investigation,
        deterministic_valid=True,
    )

    assert result.decision == "EXCEPTION"