"""Backward-compatible import for the canonical recovery implementation."""
from reconpilot.recovery import (  # noqa: F401
    BASE_BACKOFF_SECONDS,
    MAX_RETRIES,
    RecoveryResult,
    classify_failure,
    execute_with_recovery,
)

__all__ = ["BASE_BACKOFF_SECONDS","MAX_RETRIES","RecoveryResult","classify_failure","execute_with_recovery"]
