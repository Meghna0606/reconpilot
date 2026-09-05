from reconpilot.evaluation import calculate_metrics


def test_metrics_separate_resolution_and_accuracy():
    results = [
        {
            "outcome": "matched",
            "method": "deterministic",
            "correct": True,
            "auto_resolved": True,
        },
        {
            "outcome": "matched",
            "method": "ai",
            "correct": True,
            "auto_resolved": True,
        },
        {
            "outcome": "suggested_match",
            "method": "ai",
            "correct": None,
            "auto_resolved": False,
        },
        {
            "outcome": "exception",
            "method": "none",
            "correct": None,
            "auto_resolved": False,
            "exception_reason": "AMBIGUOUS",
        },
    ]

    metrics = calculate_metrics(
        results,
        total_source_records=16,
        processing_time_seconds=2.0,
    )

    assert metrics.total_cases == 4
    assert metrics.total_source_records == 16
    assert metrics.deterministic_resolutions == 1
    assert metrics.ai_assisted_resolutions == 1
    assert metrics.suggested_matches == 1
    assert metrics.unresolved_exceptions == 1

    assert metrics.resolution_rate == 0.5
    assert metrics.resolved_accuracy == 1.0
    assert metrics.overall_accuracy == 0.5

    assert metrics.false_auto_match_count == 0
    assert metrics.throughput_cases_per_second == 2.0
    assert metrics.exception_breakdown == {"AMBIGUOUS": 1}


def test_false_auto_match_is_counted():
    results = [
        {
            "outcome": "matched",
            "method": "ai",
            "correct": False,
            "auto_resolved": True,
        },
        {
            "outcome": "exception",
            "method": "none",
            "correct": None,
            "auto_resolved": False,
            "exception_reason": "CONFLICTING_EVIDENCE",
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics.total_cases == 2
    assert metrics.resolution_rate == 0.5
    assert metrics.resolved_accuracy == 0.0
    assert metrics.overall_accuracy == 0.0
    assert metrics.false_auto_match_count == 1
    assert metrics.exception_breakdown == {
        "CONFLICTING_EVIDENCE": 1
    }


def test_empty_batch_is_safe():
    metrics = calculate_metrics([])

    assert metrics.total_cases == 0
    assert metrics.resolution_rate == 0.0
    assert metrics.resolved_accuracy == 0.0
    assert metrics.overall_accuracy == 0.0
    assert metrics.throughput_cases_per_second == 0.0


def test_exceptions_are_not_counted_as_matches():
    results = [
        {
            "outcome": "exception",
            "method": "none",
            "correct": False,
            "auto_resolved": False,
            "exception_reason": "NO_CANDIDATES",
        },
        {
            "outcome": "exception",
            "method": "none",
            "correct": False,
            "auto_resolved": False,
            "exception_reason": "LLM_FAILURE",
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics.deterministic_resolutions == 0
    assert metrics.ai_assisted_resolutions == 0
    assert metrics.unresolved_exceptions == 2
    assert metrics.resolution_rate == 0.0
    assert metrics.overall_accuracy == 0.0