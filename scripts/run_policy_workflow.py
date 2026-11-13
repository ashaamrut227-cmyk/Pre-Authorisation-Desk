
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_workflow import run_policy_workflow


def main():
    print("=" * 65)
    print("HEALTH INSURANCE POLICY WORKFLOW")
    print("=" * 65)

    question = input(
        "\nEnter your policy question "
        "(example: What is the waiting period for cataract treatment?): "
    ).strip()

    claim = input(
        "\nEnter the claim to verify "
        "(example: Cataract has a 24 months waiting period.): "
    ).strip()

    if not question or not claim:
        print("\nError: Both the question and claim are required.")
        return

    try:
        result = run_policy_workflow(
            question=question,
            claim=claim,
            top_k=8,
        )
    except Exception as error:
        print(f"\nWorkflow failed: {error}")
        print(
            "Check that the policy index exists and that the virtual "
            "environment has the required packages installed."
        )
        return

    print("\n" + "=" * 65)
    print("WORKFLOW RESULT")
    print("=" * 65)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("SOURCE CITATIONS")
    print("=" * 65)

    for citation in result.get("evidence", []):
        print(
            f"\nDocument: {citation.get('document_title')}\n"
            f"PDF page: {citation.get('page_number')}\n"
            f"Chunk ID: {citation.get('chunk_id')}\n"
            f"Text: {citation.get('text')}"
        )

    print("\nFinal status:", result["status"])
    print("Human review required:", result["human_review_required"])


if __name__ == "__main__":
    main()
