from datetime import datetime
from decimal import Decimal

from reconpilot.ai_investigator import (
    AIInvestigation,
    build_investigation_prompt,
    parse_ai_response,
)
from reconpilot.candidates import Candidate


def make_candidate(candidate_id="INV-001"):
    return Candidate(
        candidate_id=candidate_id,
        entity_type="invoice",
        score=0.91,
        reasons=("amount_exact", "reference_high_similarity"),
    )


def test_valid_ai_response_is_accepted():
    candidates = [make_candidate()]

    response = {
        "classification": "LIKELY_MATCH",
        "selected_candidate_id": "INV-001",
        "confidence": 0.92,
        "evidence": [
            "Amount agrees",
            "Reference similarity is high",
        ],
        "recommended_action": "AUTO_RESOLVE",
        "uncertainty": [],
    }

    result = parse_ai_response(response, candidates)

    assert isinstance(result, AIInvestigation)
    assert result.classification == "LIKELY_MATCH"
    assert result.selected_candidate_id == "INV-001"
    assert result.confidence == 0.92


def test_ai_cannot_select_unknown_candidate():
    candidates = [make_candidate("INV-001")]

    response = {
        "classification": "LIKELY_MATCH",
        "selected_candidate_id": "INV-999",
        "confidence": 0.95,
        "evidence": ["Strong evidence"],
        "recommended_action": "AUTO_RESOLVE",
        "uncertainty": [],
    }

    try:
        parse_ai_response(response, candidates)
        assert False, "Expected validation failure"
    except ValueError:
        pass


def test_ambiguous_result_cannot_select_candidate():
    candidates = [make_candidate("INV-001")]

    response = {
        "classification": "AMBIGUOUS",
        "selected_candidate_id": "INV-001",
        "confidence": 0.75,
        "evidence": ["Conflicting evidence"],
        "recommended_action": "HUMAN_REVIEW",
        "uncertainty": ["Two candidates are plausible"],
    }

    try:
        parse_ai_response(response, candidates)
        assert False, "Expected validation failure"
    except ValueError:
        pass


def test_invalid_classification_is_rejected():
    candidates = [make_candidate()]

    response = {
        "classification": "FORCE_MATCH",
        "selected_candidate_id": "INV-001",
        "confidence": 0.99,
        "evidence": ["Looks correct"],
        "recommended_action": "AUTO_RESOLVE",
        "uncertainty": [],
    }

    try:
        parse_ai_response(response, candidates)
        assert False, "Expected validation failure"
    except ValueError:
        pass


def test_prompt_contains_only_target_and_top_three_candidates():
    candidates = [
        make_candidate("INV-001"),
        make_candidate("INV-002"),
        make_candidate("INV-003"),
        make_candidate("INV-004"),
    ]

    target = {
        "bank_txn_id": "BANK-001",
        "amount": "9764.00",
        "reference": "RZP-4501",
    }

    prompt = build_investigation_prompt(target, candidates)

    assert "BANK-001" in prompt
    assert "INV-001" in prompt
    assert "INV-002" in prompt
    assert "INV-003" in prompt
    assert "INV-004" not in prompt


def test_no_candidate_prompt_is_safe():
    prompt = build_investigation_prompt(
        {"bank_txn_id": "BANK-001"},
        [],
    )

    assert "NO CANDIDATES" in prompt