import pytest

from reconpilot.recovery import (
    execute_with_recovery,
)


def test_successful_operation_requires_no_retry():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        return "success"

    result = execute_with_recovery(operation)

    assert result.status == "success"
    assert result.result == "success"
    assert result.retry_count == 0
    assert calls == 1


def test_timeout_is_retried():
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1

        if calls < 3:
            raise TimeoutError("LLM timeout")

        return "success"

    result = execute_with_recovery(
        operation,
        max_retries=2,
        sleep=sleeps.append,
    )

    assert result.status == "recovered"
    assert result.result == "success"
    assert result.retry_count == 2
    assert calls == 3
    assert len(sleeps) == 2


def test_retry_uses_exponential_backoff():
    sleeps = []

    def operation():
        raise TimeoutError("timeout")

    result = execute_with_recovery(
        operation,
        max_retries=2,
        base_backoff_seconds=0.1,
        sleep=sleeps.append,
    )

    assert result.status == "exception"
    assert result.retry_count == 2
    assert sleeps == pytest.approx([0.1, 0.2])


def test_validation_error_is_not_retried():
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1
        raise ValueError("malformed AI response")

    result = execute_with_recovery(
        operation,
        max_retries=2,
        sleep=sleeps.append,
    )

    assert result.status == "exception"
    assert result.failure_type == "VALIDATION_FAILURE"
    assert result.retry_count == 0
    assert calls == 1
    assert sleeps == []


def test_fallback_runs_after_retries_fail():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise TimeoutError("LLM unavailable")

    def fallback():
        return "deterministic result"

    result = execute_with_recovery(
        operation,
        fallback=fallback,
        max_retries=2,
        sleep=lambda _: None,
    )

    assert result.status == "recovered"
    assert result.result == "deterministic result"
    assert result.retry_count == 2
    assert result.fallback_used == "deterministic_fallback"


def test_failed_fallback_becomes_exception():
    def operation():
        raise TimeoutError("LLM unavailable")

    def fallback():
        raise ValueError("deterministic validation failed")

    result = execute_with_recovery(
        operation,
        fallback=fallback,
        max_retries=1,
        sleep=lambda _: None,
    )

    assert result.status == "exception"
    assert result.fallback_used == "deterministic_fallback_failed"


def test_no_candidate_failure_is_handled_safely():
    def operation():
        raise ValueError("NO_CANDIDATES")

    result = execute_with_recovery(
        operation,
        max_retries=2,
        sleep=lambda _: None,
    )

    assert result.status == "exception"
    assert result.failure_type == "VALIDATION_FAILURE"
    assert result.retry_count == 0


def test_batch_continues_after_one_failed_operation():
    results = []

    def failing_operation():
        raise TimeoutError("temporary failure")

    def successful_operation():
        return "next case processed"

    results.append(
        execute_with_recovery(
            failing_operation,
            max_retries=1,
            sleep=lambda _: None,
        )
    )

    results.append(
        execute_with_recovery(
            successful_operation,
        )
    )

    assert results[0].status == "exception"
    assert results[1].status == "success"
    assert results[1].result == "next case processed"