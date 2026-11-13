
import time

import pytest

from src.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    OperationTimeoutError,
    call_with_retry,
    run_with_timeout,
)


def test_retry_succeeds_after_temporary_failure():
    attempts = {"count": 0}

    def flaky_operation():
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise RuntimeError("Temporary failure")

        return "success"

    result = call_with_retry(
        flaky_operation,
        max_attempts=3,
        base_delay=0.01,
        timeout_seconds=2,
    )

    assert result == "success"
    assert attempts["count"] == 3


def test_retry_stops_after_maximum_attempts():
    attempts = {"count": 0}

    def failing_operation():
        attempts["count"] += 1
        raise RuntimeError("Service unavailable")

    with pytest.raises(RuntimeError, match="Service unavailable"):
        call_with_retry(
            failing_operation,
            max_attempts=2,
            base_delay=0,
            timeout_seconds=2,
        )

    assert attempts["count"] == 2


def test_timeout_raises_error():
    def slow_operation():
        time.sleep(0.2)
        return "finished"

    started = time.monotonic()

    with pytest.raises(OperationTimeoutError):
        run_with_timeout(slow_operation, timeout_seconds=0.02)

    elapsed = time.monotonic() - started
    assert elapsed < 0.15


def test_circuit_breaker_opens_after_repeated_failures():
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=30,
    )

    def failing_operation():
        raise RuntimeError("Service unavailable")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(failing_operation)

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not execute")


def test_circuit_breaker_resets_after_success():
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=30,
    )

    assert breaker.call(lambda: "success") == "success"

    with pytest.raises(RuntimeError):
        breaker.call(
            lambda: (_ for _ in ()).throw(RuntimeError("Temporary"))
        )

    assert breaker.call(lambda: "recovered") == "recovered"
