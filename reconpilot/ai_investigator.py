from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import re
import time

from pydantic import BaseModel, ConfigDict, Field

from reconpilot.candidates import Candidate
from reconpilot.config import AI_MODE, LITELLM_MODEL

# ============================================================
# PHASE 4 — AI INVESTIGATION
# ============================================================


ALLOWED_CLASSIFICATIONS = {
    "LIKELY_MATCH",
    "LIKELY_FEE_ADJUSTMENT",
    "PARTIAL_PAYMENT",
    "DUPLICATE",
    "MISSING_RECORD",
    "AMBIGUOUS",
    "UNRESOLVABLE",
}


class AIInvestigation(BaseModel):
    """
    Strict structured result returned by the AI investigator.
    """

    model_config = ConfigDict(extra="forbid")

    classification: str
    selected_candidate_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str
    uncertainty: list[str] = Field(default_factory=list)


class AIFailure(BaseModel):
    """
    Typed failure result.

    AI failures must never crash the reconciliation batch.
    """

    model_config = ConfigDict(extra="forbid")

    failure_type: str
    detail: str
    retryable: bool = False


@dataclass(frozen=True)
class AIInvestigationResult:
    """
    Wrapper around either a valid AI investigation or a safe failure.
    """

    success: bool
    investigation: Optional[AIInvestigation] = None
    failure: Optional[AIFailure] = None


def build_investigation_prompt(
    target: dict[str, Any],
    candidates: list[Candidate],
) -> str:
    """
    Build a compact prompt containing only the target record
    and deterministic candidates.

    The AI must not receive the complete database.
    """

    limited_candidates = candidates[:3]

    candidate_lines = []

    for candidate in limited_candidates:
        candidate_lines.append(
            f"""
Candidate ID: {candidate.candidate_id}
Entity type: {candidate.entity_type}
Deterministic score: {candidate.score}
Signals: {", ".join(candidate.reasons) or "none"}
""".strip()
        )

    candidates_text = (
        "\n\n".join(candidate_lines)
        if candidate_lines
        else "NO CANDIDATES"
    )

    return f"""
You are investigating an ambiguous financial reconciliation case.

Your task is to interpret conflicting evidence conservatively.

TARGET RECORD:
{target}

CANDIDATES:
{candidates_text}

Allowed classifications:
LIKELY_MATCH
LIKELY_FEE_ADJUSTMENT
PARTIAL_PAYMENT
DUPLICATE
MISSING_RECORD
AMBIGUOUS
UNRESOLVABLE

Rules:
- Never invent evidence.
- Never create a candidate ID that was not supplied.
- If evidence conflicts, prefer AMBIGUOUS.
- A high confidence score does not by itself authorize an automatic match.
- Keep evidence concise and based only on the supplied information.

Return ONLY valid JSON matching this schema:

{{
  "classification": "LIKELY_MATCH",
  "selected_candidate_id": "candidate-id-or-null",
  "confidence": 0.0,
  "evidence": ["reason 1", "reason 2"],
  "recommended_action": "AUTO_RESOLVE",
  "uncertainty": ["remaining uncertainty"]
}}
""".strip()


def _validate_investigation(
    investigation: AIInvestigation,
    candidates: list[Candidate],
) -> AIInvestigation:
    """
    Apply application-level validation after Pydantic validation.
    """

    if investigation.classification not in ALLOWED_CLASSIFICATIONS:
        raise ValueError(
            f"unsupported classification: {investigation.classification}"
        )

    candidate_ids = {candidate.candidate_id for candidate in candidates}

    if investigation.selected_candidate_id is not None:
        if investigation.selected_candidate_id not in candidate_ids:
            raise ValueError(
                "AI selected a candidate that was not supplied"
            )

    if investigation.classification in {
        "AMBIGUOUS",
        "UNRESOLVABLE",
        "MISSING_RECORD",
    }:
        if investigation.selected_candidate_id is not None:
            raise ValueError(
                "uncertain classification cannot select a candidate"
            )

    return investigation


def parse_ai_response(
    response: Any,
    candidates: list[Candidate],
) -> AIInvestigation:
    """
    Parse and validate an AI response.

    Accepts:
    - a JSON string
    - a Python dictionary
    - a Pydantic-compatible object
    """

    if isinstance(response, str):
        import json

        response = json.loads(response)

    investigation = AIInvestigation.model_validate(response)

    return _validate_investigation(
        investigation,
        candidates,
    )


def investigate_with_llm(
    target: dict[str, Any],
    candidates: list[Candidate],
    *,
    model: str = LITELLM_MODEL,
    max_retries: int = 2,
) -> AIInvestigationResult:
    """
    Investigate an unresolved case using LiteLLM.

    The function is deliberately isolated from the rest of the
    reconciliation pipeline so API failures can be handled safely.

    The actual LLM call is imported lazily, which keeps the module
    easy to test without making an API request.
    """

    if not candidates:
        return AIInvestigationResult(
            success=False,
            failure=AIFailure(
                failure_type="NO_CANDIDATES",
                detail="No deterministic candidates were generated.",
                retryable=False,
            ),
        )

    prompt = build_investigation_prompt(
        target,
        candidates[:3],
    )

    try:
        from litellm import completion
    except Exception as exc:
        return AIInvestigationResult(
            success=False,
            failure=AIFailure(
                failure_type="LLM_IMPORT_ERROR",
                detail=str(exc),
                retryable=False,
            ),
        )
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative financial "
                            "reconciliation investigator."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            content = response.choices[0].message.content

            investigation = parse_ai_response(
                content,
                candidates[:3],
            )

            return AIInvestigationResult(
                success=True,
                investigation=investigation,
            )

        except Exception as exc:
            last_error = exc

            # Retry only when another attempt is available.
            if attempt >= max_retries:
                break

            error_text = str(exc)

            # Groq/LiteLLM rate-limit errors often contain
            # a recommended retry delay such as:
            # "try again in 4.48s"
            match = re.search(
                r"try again in\s+([\d.]+)s",
                error_text,
                re.IGNORECASE,
            )

            if match:
                delay = float(match.group(1)) + 0.5
            else:
                # Conservative exponential backoff:
                # 2s, 4s, ...
                delay = 2 ** (attempt + 1)

            time.sleep(delay)

    return AIInvestigationResult(
        success=False,
        failure=AIFailure(
            failure_type="AI_UNAVAILABLE",
            detail=str(last_error),
            retryable=True,
        ),
    )

def investigate_locally(
    target: dict[str, Any],
    candidates: list[Candidate],
) -> AIInvestigationResult:
    """Deterministic local evaluator used for reproducible offline demos.

    This is deliberately labeled LOCAL_EVALUATOR, not an LLM. It selects only
    from supplied candidates and uses their deterministic evidence scores.
    """
    if not candidates:
        return AIInvestigationResult(
            success=False,
            failure=AIFailure(failure_type="NO_CANDIDATES", detail="No candidates supplied.", retryable=False),
        )
    ranked = sorted(candidates, key=lambda c: (-c.score, c.candidate_id))
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    gap = top.score - second.score if second else 1.0
    if top.score < 0.60:
        classification = "UNRESOLVABLE"
        action = "MANUAL_REVIEW"
        selected = None
        confidence = min(0.69, top.score)
        uncertainty = ["Deterministic candidate evidence is weak."]
    elif second is not None and gap < 0.05:
        classification = "AMBIGUOUS"
        action = "MANUAL_REVIEW"
        selected = None
        confidence = 0.60 + min(gap, 0.05)
        uncertainty = ["Top candidates have similar deterministic scores."]
    else:
        classification = "LIKELY_MATCH"
        action = "AUTO_RESOLVE" if top.score >= 0.90 else "MANUAL_REVIEW"
        selected = top.candidate_id
        confidence = min(0.99, max(0.90, top.score))
        uncertainty = [] if gap >= 0.15 else ["Candidate ranking margin is modest."]
    evidence = [f"Top candidate deterministic score={top.score:.3f}", *top.reasons[:3]]
    investigation = AIInvestigation(
        classification=classification,
        selected_candidate_id=selected,
        confidence=round(confidence, 4),
        evidence=evidence,
        recommended_action=action,
        uncertainty=uncertainty,
    )
    return AIInvestigationResult(success=True, investigation=investigation)


def investigate(
    target: dict[str, Any],
    candidates: list[Candidate],
    *,
    mode: str = AI_MODE,
    model: str = LITELLM_MODEL,
) -> AIInvestigationResult:
    """Route investigation through explicit LOCAL, LIVE, or OFF modes."""
    normalized = mode.lower()
    if normalized in {"local", "mock", "evaluation"}:
        return investigate_locally(target, candidates)
    if normalized in {"live", "llm", "real_llm", "openai", "groq"}:
        return investigate_with_llm(target, candidates)
    return AIInvestigationResult(
        success=False,
        failure=AIFailure(failure_type="AI_DISABLED", detail="AI mode is off; ambiguous cases require review.", retryable=False),
    )
