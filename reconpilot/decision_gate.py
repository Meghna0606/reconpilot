from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from reconpilot.ai_investigator import AIInvestigation

AUTO_RESOLVE_THRESHOLD = 0.90
SUGGEST_MATCH_THRESHOLD = 0.70

@dataclass(frozen=True)
class DecisionResult:
    decision: str
    reason: str
    selected_candidate_id: Optional[str] = None
    confidence: Optional[float] = None

def apply_decision_gate(investigation: AIInvestigation, *, deterministic_valid: bool,
                        conflicting_candidate: bool = False,
                        auto_resolve_threshold: float = AUTO_RESOLVE_THRESHOLD,
                        suggest_match_threshold: float = SUGGEST_MATCH_THRESHOLD) -> DecisionResult:
    confidence = investigation.confidence
    classification = investigation.classification
    candidate_id = investigation.selected_candidate_id
    if classification in {"AMBIGUOUS", "UNRESOLVABLE", "PARTIAL_PAYMENT", "DUPLICATE", "MISSING_RECORD"}:
        return DecisionResult("EXCEPTION", f"unsafe_ai_classification:{classification}", None, confidence)
    if classification in {"LIKELY_MATCH", "LIKELY_FEE_ADJUSTMENT"} and candidate_id is None:
        return DecisionResult("EXCEPTION", "match_classification_without_selected_candidate", None, confidence)
    if conflicting_candidate:
        if confidence >= suggest_match_threshold:
            return DecisionResult("SUGGEST_MATCH", "strong_conflicting_candidate_requires_review", candidate_id, confidence)
        return DecisionResult("EXCEPTION", "conflicting_evidence", None, confidence)
    if not deterministic_valid:
        if confidence >= suggest_match_threshold:
            return DecisionResult("SUGGEST_MATCH", "deterministic_validation_failed_requires_review", candidate_id, confidence)
        return DecisionResult("EXCEPTION", "deterministic_validation_failed", None, confidence)
    if confidence >= auto_resolve_threshold:
        return DecisionResult("AUTO_RESOLVE", "high_confidence_and_deterministic_validation_passed", candidate_id, confidence)
    if confidence >= suggest_match_threshold:
        return DecisionResult("SUGGEST_MATCH", "medium_confidence_requires_human_review", candidate_id, confidence)
    return DecisionResult("EXCEPTION", "low_confidence", None, confidence)
