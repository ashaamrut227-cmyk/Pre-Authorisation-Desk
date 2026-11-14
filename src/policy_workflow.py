
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

        # 1. Validate input
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer.")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        # 2. Prepare multiple retrieval queries
        queries = [
            question.strip(),
            claim.strip(),
            f"waiting period and exclusions for {question}",
            f"coverage benefits and limits for {question}",
            f"procedure-specific restrictions for {question}",
        ]

        queries = list(dict.fromkeys(queries))

        # 3. Retrieve evidence independently for each query
        evidence_by_id = {}

        for query in queries:
            results = self.retriever.search(
                query,
                top_k=top_k,
            )

            for item in results:
                chunk_id = item.get("chunk_id")

                if chunk_id is None:
                    chunk_id = (
                        item.get("document_id"),
                        item.get("page_number"),
                        item.get("text"),
                    )

                if (
                    chunk_id not in evidence_by_id
                    or item["score"] > evidence_by_id[chunk_id]["score"]
                ):
                    evidence_by_id[chunk_id] = item

        # 4. Combine and rank unique evidence passages
        evidence = sorted(
            evidence_by_id.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        # 5. Verify the claim against retrieved policy evidence
        verification = verify_evidence(
            claim=claim,
            evidence=evidence,
        )

        # 6. Prepare traceable citations
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

        # 7. Determine workflow status and routing
        status = verification.get("status", "unverified")
        eligibility = verification.get("eligibility_assessment", {})
        eligibility_status = eligibility.get("status")

        # Default to human medical review.
        # The system must never make the final coverage decision.
        route = "medical_review"

        if eligibility_status == "waiting_period_not_completed":
            policy_age = eligibility.get("policy_age_days")
            required_period = eligibility.get(
                "required_waiting_period_days"
            )

            if policy_age is not None and required_period is not None:
                policy_age_months = policy_age / 30
                required_months = required_period / 30

                message = (
                    f"The policy has been active for approximately "
                    f"{policy_age_months:.0f} months, while cataract "
                    f"treatment is subject to a waiting period of "
                    f"approximately {required_months:.0f} months. "
                    f"The waiting period appears incomplete. "
                    f"Prior continuous coverage, portability and "
                    f"applicable policy exceptions must be checked "
                    f"by a human medical reviewer."
                )
            else:
                message = (
                    "The retrieved policy evidence indicates that "
                    "the waiting period may not be completed. "
                    "Additional checks and human medical review "
                    "are required."
                )

            route_reason = (
                "The policy age is shorter than the specified "
                "waiting period. Prior coverage and applicable "
                "exceptions require human verification."
            )

        elif status == "supported":
            message = (
                "The retrieved evidence supports the recognized "
                "claim facts. Human medical review is still required "
                "before any pre-authorisation decision."
            )

            route_reason = (
                "The claim facts are supported by retrieved evidence, "
                "but final coverage assessment requires human review."
            )

        elif status == "contradicted":
            message = (
                "The claim conflicts with retrieved policy evidence. "
                "The case has been routed for human medical review."
            )

            route_reason = (
                "A conflict was identified between the claim "
                "and retrieved policy evidence."
            )

        else:
            message = (
                "The retrieved evidence does not establish the claim. "
                "Additional policy evidence and human medical review "
                "are required."
            )

            route_reason = (
                "The available evidence is insufficient to establish "
                "the claim."
            )

        # 8. Return the complete workflow result
        return {
            "question": question,
            "claim": claim,
            "status": status,
            "message": message,
            "route": route,
            "route_reason": route_reason,
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
