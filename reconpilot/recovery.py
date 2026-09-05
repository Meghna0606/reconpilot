from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import time


MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 0.1


@dataclass(frozen=True)
class RecoveryResult:
    """
    Safe result for an operation that may fail.

    status:
        success   -> operation succeeded immediately
        recovered -> operation succeeded after retry/fallback
        exception -> operation could not be completed safely
    """

    status: str
    result: Any = None
    failure_type: Optional[str] = None
    detail: Optional[str] = None
    retry_count: int = 0
    fallback_used: Optional[str] = None


def classify_failure(exc: Exception) -> tuple[str, bool]:
    """
    Classify an error and determine whether retrying is appropriate.
    """

    name = type(exc).__name__.lower()
    message = str(exc).lower()

    if "timeout" in name or "timeout" in message:
        return "LLM_TIMEOUT", True

    if "ratelimit" in name or "rate limit" in message:
        return "RATE_LIMIT", True

    if "connection" in name or "temporary" in message:
        return "TEMPORARY_API_ERROR", True

    if isinstance(exc, (ValueError, TypeError)):
        return "VALIDATION_FAILURE", False

    return "LLM_FAILURE", True


def execute_with_recovery(
    operation: Callable[[], Any],
    *,
    fallback: Optional[Callable[[], Any]] = None,
    max_retries: int = MAX_RETRIES,
    base_backoff_seconds: float = BASE_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> RecoveryResult:
    """
    Execute an operation with bounded retries and optional fallback.

    Recoverable failures are retried using exponential backoff.

    If all retries fail:
        - use the fallback if supplied
        - otherwise return a safe typed exception

    Exceptions never escape this function.
    """

    retry_count = 0
    last_failure_type: Optional[str] = None
    last_detail: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            result = operation()

            if retry_count == 0:
                return RecoveryResult(
                    status="success",
                    result=result,
                    retry_count=0,
                )

            return RecoveryResult(
                status="recovered",
                result=result,
                failure_type=last_failure_type,
                detail=last_detail,
                retry_count=retry_count,
            )

        except Exception as exc:
            failure_type, retryable = classify_failure(exc)

            last_failure_type = failure_type
            last_detail = str(exc)

            # Validation/malformed-data errors should not be retried.
            if not retryable:
                break

            # No attempts remaining.
            if attempt >= max_retries:
                break

            retry_count += 1

            # Exponential backoff:
            # first retry -> base delay
            # second retry -> base * 2
            delay = base_backoff_seconds * (2**attempt)
            sleep(delay)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if fallback is not None:
        try:
            fallback_result = fallback()

            return RecoveryResult(
                status="recovered",
                result=fallback_result,
                failure_type=last_failure_type,
                detail=last_detail,
                retry_count=retry_count,
                fallback_used="deterministic_fallback",
            )

        except Exception as exc:
            fallback_type, _ = classify_failure(exc)

            return RecoveryResult(
                status="exception",
                failure_type=fallback_type,
                detail=str(exc),
                retry_count=retry_count,
                fallback_used="deterministic_fallback_failed",
            )

    # --------------------------------------------------------
    # No fallback -> conservative exception
    # --------------------------------------------------------

    return RecoveryResult(
        status="exception",
        failure_type=last_failure_type,
        detail=last_detail,
        retry_count=retry_count,
    )