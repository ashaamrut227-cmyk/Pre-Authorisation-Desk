
from src.policy_workflow import PolicyWorkflow


class FakeRetriever:
    def search(self, query, top_k=8):
        return [{
            "score": 0.9,
            "chunk_id": "page-12-part-4",
            "document_id": "equicover-policy",
            "document_title": "EquiCover Health Policy Wording",
            "source_file": "policy_wording.pdf",
            "page_number": 12,
            "text": (
                "24 Months waiting period. "
                "Cataract and age-related eye ailments."
            ),
        }]


def test_workflow_returns_supported_claim(monkeypatch):
    monkeypatch.setattr(
        "src.policy_workflow.PolicyRetriever",
        FakeRetriever,
    )

    workflow = PolicyWorkflow()
    result = workflow.run(
        question="What is the waiting period for cataract?",
        claim="Cataract has a 24 months waiting period.",
    )

    assert result["status"] == "supported"
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None
    assert result["evidence"][0]["page_number"] == 12


def test_workflow_flags_contradicted_claim(monkeypatch):
    monkeypatch.setattr(
        "src.policy_workflow.PolicyRetriever",
        FakeRetriever,
    )

    workflow = PolicyWorkflow()
    result = workflow.run(
        question="What is the waiting period for cataract?",
        claim="Cataract has a 12 months waiting period.",
    )

    assert result["status"] == "contradicted"
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None


def test_workflow_flags_unsupported_procedure(monkeypatch):
    monkeypatch.setattr(
        "src.policy_workflow.PolicyRetriever",
        FakeRetriever,
    )

    workflow = PolicyWorkflow()
    result = workflow.run(
        question="What is the waiting period for cataract?",
        claim="Dental implants have no waiting period.",
    )

    assert result["status"] == "unverified"
    assert result["human_review_required"] is True


def test_workflow_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(
        "src.policy_workflow.PolicyRetriever",
        FakeRetriever,
    )

    workflow = PolicyWorkflow()

    try:
        workflow.run(
            question="",
            claim="Cataract has a 24 months waiting period.",
        )
    except ValueError as error:
        assert "Question" in str(error)
    else:
        raise AssertionError("An empty question should be rejected.")
