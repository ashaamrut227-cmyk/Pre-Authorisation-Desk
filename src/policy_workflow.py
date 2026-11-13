
from typing import Any

from src.evidence_verifier import verify_evidence
from src.policy_retriever import PolicyRetriever


class PolicyWorkflow:
    """Connect policy retrieval and evidence verification."""

    def __init__(self):
        self.retriever = PolicyRetriever()

    def run(
        self,
        question: str,
        claim: str,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Retrieve policy evidence and verify a claim."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer.")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        evidence = self.retriever.search(question, top_k=top_k)

        verification = verify_evidence(
            claim=claim,
            evidence=evidence,
        )

        citations = [
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

        status = verification["status"]

        if status == "supported":
            message = (
                "The retrieved evidence supports the recognized claim "
                "facts. Human review is still required before any "
                "pre-authorisation decision."
            )
        elif status == "contradicted":
            message = (
                "The claim conflicts with retrieved policy evidence. "
                "Human review is required."
            )
        else:
            message = (
                "The retrieved evidence does not establish the claim. "
                "Additional policy evidence and human review are required."
            )

        return {
            "question": question,
            "claim": claim,
            "status": status,
            "message": message,
            "evidence": citations,
            "verification": verification,
            "human_review_required": True,
            "final_coverage_decision": None,
        }


def run_policy_workflow(
    question: str,
    claim: str,
    top_k: int = 8,
) -> dict[str, Any]:
    """Convenience function for scripts and API integration."""
    workflow = PolicyWorkflow()
    return workflow.run(
        question=question,
        claim=claim,
        top_k=top_k,
    )
