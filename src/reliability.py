
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationTimeoutError(TimeoutError):
    """Raised when an operation exceeds its allowed time."""


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open."""


def run_with_timeout(
    operation: Callable[[], T],
    timeout_seconds: float = 10.0,
) -> T:
    """Wait for a synchronous operation for at most timeout_seconds.

    Python cannot forcibly terminate a running worker thread. A timed-out
    operation may continue in the background, so use this for bounded
    waiting, not as a process-level kill mechanism.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(operation)

    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout as error:
        future.cancel()
        raise OperationTimeoutError(
            f"Operation exceeded {timeout_seconds} seconds."
        ) from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.25,
    timeout_seconds: float = 10.0,
) -> T:
    """Run an operation with bounded attempts and exponential backoff."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if base_delay < 0:
        raise ValueError("base_delay cannot be negative.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        remaining = timeout_seconds - (time.monotonic() - started)

        if remaining <= 0:
            raise OperationTimeoutError(
                "Operation exceeded its total time budget."
            ) from last_error

        try:
            return run_with_timeout(operation, remaining)
        except Exception as error:
            last_error = error

            if attempt == max_attempts - 1:
                break

            delay = base_delay * (2 ** attempt)
            remaining = timeout_seconds - (time.monotonic() - started)

            if remaining <= 0:
                break

            time.sleep(min(delay, remaining))

    if time.monotonic() - started >= timeout_seconds:
        raise OperationTimeoutError(
            "Operation exceeded its total time budget."
        ) from last_error

    if last_error is not None:
        raise last_error

    raise RuntimeError("Operation failed without an exception.")


class CircuitBreaker:
    """Simple thread-safe circuit breaker with a recovery timeout."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive.")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def call(self, operation: Callable[[], T]) -> T:
        """Call the operation only when the circuit allows it."""
        with self._lock:
            if self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at

                if elapsed < self.recovery_timeout:
                    raise CircuitOpenError(
                        "Circuit is open; retry after recovery timeout."
                    )

                # Permit a trial call after the recovery period.
                self._opened_at = None

        try:
            result = operation()
        except Exception:
            with self._lock:
                self._failures += 1

                if self._failures >= self.failure_threshold:
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit breaker opened after %s failures.",
                        self._failures,
                    )
            raise

        with self._lock:
            self._failures = 0
            self._opened_at = None

        return result
