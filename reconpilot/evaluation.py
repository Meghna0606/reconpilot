from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvaluationMetrics:
    total_cases: int
    total_source_records: int
    deterministic_resolutions: int
    ai_assisted_resolutions: int
    suggested_matches: int
    unresolved_exceptions: int
    resolution_rate: float
    resolved_accuracy: float
    overall_accuracy: float
    false_auto_match_count: int
    throughput_cases_per_second: float
    exception_breakdown: dict[str, int]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(
    results: Iterable[dict],
    *,
    total_source_records: int = 0,
    processing_time_seconds: float = 0.0,
) -> EvaluationMetrics:
    """
    Calculate honest reconciliation metrics from case-level results.

    Expected result fields:

        outcome:
            matched
            suggested_match
            exception

        method:
            deterministic
            ai
            none

        correct:
            True / False / None

        auto_resolved:
            True / False

        exception_reason:
            optional string

    Exceptions are never counted as successful matches.
    """

    results = list(results)

    total_cases = len(results)

    deterministic_resolutions = 0
    ai_assisted_resolutions = 0
    suggested_matches = 0
    unresolved_exceptions = 0

    resolved_correct = 0
    resolved_cases = 0
    overall_correct = 0

    false_auto_match_count = 0
    exception_breakdown: dict[str, int] = {}

    for result in results:
        outcome = result.get("outcome")
        method = result.get("method")
        correct = result.get("correct")
        auto_resolved = bool(result.get("auto_resolved", False))

        if outcome == "matched":
            if method == "deterministic":
                deterministic_resolutions += 1
            elif method == "ai":
                ai_assisted_resolutions += 1

            resolved_cases += 1

            if correct is True:
                resolved_correct += 1
                overall_correct += 1

            if auto_resolved and correct is False:
                false_auto_match_count += 1

        elif outcome == "suggested_match":
            suggested_matches += 1

        elif outcome == "exception":
            unresolved_exceptions += 1

            reason = result.get("exception_reason") or "unknown"
            exception_breakdown[reason] = (
                exception_breakdown.get(reason, 0) + 1
            )

        # A false auto-match is important even if the caller
        # accidentally labels the outcome differently.
        if (
            auto_resolved
            and correct is False
            and outcome != "matched"
        ):
            false_auto_match_count += 1

    resolution_rate = _safe_rate(
        resolved_cases,
        total_cases,
    )

    resolved_accuracy = _safe_rate(
        resolved_correct,
        resolved_cases,
    )

    overall_accuracy = _safe_rate(
        overall_correct,
        total_cases,
    )

    throughput = _safe_rate(
        total_cases,
        processing_time_seconds,
    )

    return EvaluationMetrics(
        total_cases=total_cases,
        total_source_records=total_source_records,
        deterministic_resolutions=deterministic_resolutions,
        ai_assisted_resolutions=ai_assisted_resolutions,
        suggested_matches=suggested_matches,
        unresolved_exceptions=unresolved_exceptions,
        resolution_rate=round(resolution_rate, 4),
        resolved_accuracy=round(resolved_accuracy, 4),
        overall_accuracy=round(overall_accuracy, 4),
        false_auto_match_count=false_auto_match_count,
        throughput_cases_per_second=round(throughput, 4),
        exception_breakdown=exception_breakdown,
    )