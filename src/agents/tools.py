
import logging
import threading
from typing import Any

from src.admission_classifier import (
    classify_admission,
    extract_request_entities,
)
from src.decision_router import route_pre_authorisation
from src.evidence_verifier import verify_evidence
from src.policy_retriever import PolicyRetriever
from src.reliability import CircuitBreaker, call_with_retry


logger = logging.getLogger(__name__)

RETRIEVAL_TIMEOUT_SECONDS = 120.0
RETRIEVAL_MAX_ATTEMPTS = 1

_retrieval_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30.0,
)

_policy_retriever = None
_retriever_lock = threading.Lock()


def get_policy_retriever() -> PolicyRetriever:
    """Load the policy retriever once and reuse it."""
    global _policy_retriever

    if _policy_retriever is None:
        with _retriever_lock:
            if _policy_retriever is None:
                logger.info(
                    "Initializing policy retriever and embedding model."
                )
                _policy_retriever = PolicyRetriever()
                logger.info(
                    "Policy retriever initialized successfully."
                )

    return _policy_retriever


def intake_tool(request_data: dict[str, Any]) -> dict[str, Any]:
    """Extract request entities and identify admission-type clues."""
    entities = extract_request_entities(request_data)

    admission_type = request_data.get("admission_type")
    if hasattr(admission_type, "value"):
        admission_type = admission_type.value

    classification = classify_admission(
        procedure_name=request_data["procedure_name"],
        admission_type=admission_type,
    )

    return {
        "entities": entities,
        "classification": classification,
    }


def retrieve_policy_tool(
    question: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve policy passages with timeout, retries, and circuit breaker."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Policy question must be non-empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    def retrieve() -> list[dict[str, Any]]:
        retriever = get_policy_retriever()
        evidence = retriever.search(question, top_k=top_k)

        return [
            {
                "document_title": item.get("document_title"),
                "source_file": item.get("source_file"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "text": item.get("text"),
            }
            for item in evidence
        ]

    try:
        return _retrieval_circuit_breaker.call(
            lambda: call_with_retry(
                retrieve,
                max_attempts=RETRIEVAL_MAX_ATTEMPTS,
                base_delay=0.25,
                timeout_seconds=RETRIEVAL_TIMEOUT_SECONDS,
            )
        )
    except Exception:
        logger.exception("Policy retrieval failed.")
        raise


def verify_policy_claim_tool(
    claim: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check a claim against retrieved policy passages."""
    return verify_evidence(
        claim=claim,
        evidence=evidence,
    )


def route_request_tool(
    request_data: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Recommend a workflow route without making a final decision."""
    return route_pre_authorisation(
        request_data=request_data,
        verification=verification,
    )
