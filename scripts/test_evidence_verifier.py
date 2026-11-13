
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_verifier import verify_evidence


CATARACT_WAITING_EVIDENCE = [
    {
        "page_number": 12,
        "chunk_id": "page-12-part-4",
        "text": (
            "24 Months waiting period. "
            "Cataract and age-related eye ailments."
        ),
    }
]


UNRELATED_EVIDENCE = [
    {
        "page_number": 9,
        "chunk_id": "page-9-part-5",
        "text": (
            "Expenses incurred on treatment of cataract "
            "subject to the sub limits."
        ),
    },
    {
        "page_number": 25,
        "chunk_id": "page-25-part-5",
        "text": (
            "24 months initial waiting period is applicable "
            "for the pre-existing disability covered under the policy."
        ),
    },
]


def check(name, claim, evidence, expected_status):
    result = verify_evidence(claim, evidence)

    print(f"\n{name}")
    print(f"Expected status: {expected_status}")
    print(f"Actual status:   {result['status']}")
    print(f"Reason: {result['reason']}")

    assert result["status"] == expected_status, (
        f"{name} failed: expected {expected_status}, "
        f"received {result['status']}"
    )

    assert result["verified"] == (expected_status == "supported")
    return result


def main():
    check(
        "TEST 1 - Condition and duration match",
        "Cataract and age-related eye ailments have a "
        "24 months waiting period.",
        CATARACT_WAITING_EVIDENCE,
        "supported",
    )

    check(
        "TEST 2 - Incorrect duration",
        "Cataract has a 12 months waiting period.",
        CATARACT_WAITING_EVIDENCE,
        "contradicted",
    )

    check(
        "TEST 3 - No waiting period claim",
        "Cataract has no waiting period.",
        CATARACT_WAITING_EVIDENCE,
        "contradicted",
    )

    check(
        "TEST 4 - Unsupported procedure",
        "Dental implants have no waiting period.",
        CATARACT_WAITING_EVIDENCE,
        "unverified",
    )

    check(
        "TEST 5 - Approval guarantee",
        "Cataract is always approved after 24 months.",
        CATARACT_WAITING_EVIDENCE,
        "unverified",
    )

    # The cataract sub-limit passage and pre-existing disability
    # waiting-period passage must not be combined.
    check(
        "TEST 6 - Unrelated policy clauses",
        "Cataract has a 24 months waiting period.",
        UNRELATED_EVIDENCE,
        "unverified",
    )

    missing = verify_evidence(
        "Cataract has a 24 months waiting period.",
        [],
    )
    print("\nTEST 7 - Missing evidence")
    print(f"Actual status: {missing['status']}")
    assert missing["status"] == "unverified"
    assert missing["verified"] is False

    print("\nTEST 8 - Empty claim")
    try:
        verify_evidence("", CATARACT_WAITING_EVIDENCE)
    except ValueError as error:
        print(f"Correctly rejected: {error}")
    else:
        raise AssertionError("An empty claim should have been rejected.")

    print("\nAll 8 evidence-verifier tests passed!")


if __name__ == "__main__":
    main()
