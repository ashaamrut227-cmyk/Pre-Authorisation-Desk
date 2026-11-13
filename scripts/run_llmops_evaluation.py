
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter


# Resolve the project root regardless of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

DATASET_PATH = EVALUATION_DIR / "grounded_scenarios.json"
OUTPUT_PATH = EVALUATION_DIR / "llmops_results.json"
REPORT_PATH = EVALUATION_DIR / "llmops_report.txt"

# These are prompt templates for comparison.
# This script does not call an LLM. It evaluates the answer fixtures
# stored in grounded_scenarios.json.
PROMPT_VARIANTS = {
    "baseline": (
        "Answer the user's insurance question using your available knowledge. "
        "Provide a direct answer."
    ),
    "evidence_first": (
        "Answer only from the supplied policy evidence. Do not invent clauses, "
        "amounts, waiting periods, or coverage decisions. If the evidence is "
        "insufficient, say so and request more evidence or human review."
    ),
}

# Common words are excluded so that trivial word overlap does not count
# as meaningful evidence support.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "from", "by",
    "with", "as", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "why", "how", "does",
    "do", "did", "can", "could", "should", "would", "will", "may",
    "might", "must", "has", "have", "had", "not", "no", "yes",
    "under", "after", "before", "into", "than", "then", "also",
    "only", "every", "all", "any", "each", "their", "them", "they",
    "your", "you", "we", "our", "he", "she", "his", "her",
    "policy", "supplied", "evidence", "excerpt", "illustrative",
    "test", "states", "state", "says", "said", "excerpt"
}


def tokenize(text):
    """Return normalized, meaningful tokens from text."""
    words = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", str(text).lower())
    return [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def safe_divide(numerator, denominator):
    """Avoid division by zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def sentence_claims(answer):
    """
    Split an answer into simple claims.

    This is a lightweight heuristic, not a full natural-language
    claim extractor. Semicolons and sentence-ending punctuation split claims.
    """
    pieces = re.split(r"(?<=[.!?])\s+|;\s*", str(answer).strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def claim_support_score(claim, evidence):
    """
    Estimate lexical support for a claim.

    Score = meaningful claim tokens also present in the evidence /
            all meaningful claim tokens.

    This is an overlap proxy. It cannot establish semantic entailment.
    """
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))

    if not claim_tokens:
        return 0.0

    return safe_divide(
        len(claim_tokens & evidence_tokens),
        len(claim_tokens)
    )


def evaluate_answer(question, evidence, answer):
    """
    Calculate four transparent proxy metrics:

    1. Groundedness:
       Fraction of answer claims meeting the lexical-support threshold.

    2. Context relevance:
       Fraction of meaningful question tokens found in the evidence.

    3. Faithfulness:
       Fraction of meaningful answer tokens found in the evidence.

    4. Hallucination rate:
       Fraction of answer claims failing the lexical-support threshold.

    A claim is treated as supported when at least 50% of its meaningful
    tokens overlap with the evidence. This threshold is a heuristic.
    """
    claim_threshold = 0.50

    claims = sentence_claims(answer)
    claim_scores = [
        claim_support_score(claim, evidence)
        for claim in claims
    ]

    supported_claims = sum(
        score >= claim_threshold for score in claim_scores
    )

    answer_tokens = tokenize(answer)
    evidence_tokens = set(tokenize(evidence))
    question_tokens = set(tokenize(question))

    answer_token_matches = sum(
        token in evidence_tokens for token in answer_tokens
    )

    question_token_matches = sum(
        token in evidence_tokens for token in question_tokens
    )

    groundedness = safe_divide(supported_claims, len(claims))

    context_relevance = safe_divide(
        question_token_matches, len(question_tokens)
    )

    faithfulness = safe_divide(
        answer_token_matches, len(answer_tokens)
    )

    hallucination_rate = (
        1.0 - groundedness if claims else 0.0
    )

    return {
        "groundedness": round(groundedness, 4),
        "context_relevance": round(context_relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "claims_total": len(claims),
        "claims_supported": supported_claims,
        "claim_support_scores": [
            round(score, 4) for score in claim_scores
        ],
        "answer": answer,
    }


def average_metric(results, metric):
    """Average a metric over all evaluated scenarios."""
    values = [item[metric] for item in results]
    return round(safe_divide(sum(values), len(values)), 4)


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Create evaluation/grounded_scenarios.json first."
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        scenarios = json.load(file)

    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    required_fields = {
        "id",
        "question",
        "evidence",
        "reference_answer",
        "baseline_answer",
        "evidence_first_answer",
    }

    seen_ids = set()

    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario {index} must be a JSON object.")

        missing = required_fields - set(scenario.keys())
        if missing:
            raise ValueError(
                f"Scenario {index} is missing fields: {sorted(missing)}"
            )

        for field in required_fields:
            if not isinstance(scenario[field], str) or not scenario[field].strip():
                raise ValueError(
                    f"Scenario {index}: '{field}' must be a non-empty string."
                )

        scenario_id = scenario["id"]
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)

    return scenarios


def main():
    try:
        scenarios = load_dataset()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        "baseline": "baseline_answer",
        "evidence_first": "evidence_first_answer",
    }

    results_by_variant = {}

    for variant_name, answer_field in variants.items():
        scenario_results = []

        for scenario in scenarios:
            result = evaluate_answer(
                question=scenario["question"],
                evidence=scenario["evidence"],
                answer=scenario[answer_field],
            )

            scenario_results.append({
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "evidence": scenario["evidence"],
                "reference_answer": scenario["reference_answer"],
                **result,
            })

        results_by_variant[variant_name] = scenario_results

    metric_names = [
        "groundedness",
        "context_relevance",
        "faithfulness",
        "hallucination_rate",
    ]

    aggregate_results = {}

    for variant_name, variant_results in results_by_variant.items():
        aggregate_results[variant_name] = {
            metric: average_metric(variant_results, metric)
            for metric in metric_names
        }

    baseline = aggregate_results["baseline"]
    evidence_first = aggregate_results["evidence_first"]

    # Positive improvement is good for the first three metrics.
    # For hallucination rate, a negative change is an improvement.
    changes = {
        metric: round(
            evidence_first[metric] - baseline[metric], 4
        )
        for metric in metric_names
    }

    report = {
        "report_type": "offline_lexical_proxy_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Health Insurance Pre-Authorisation Desk",
        "scenario_count": len(scenarios),
        "prompt_variants": {
            name: {
                "template": template,
                "answer_source": variants[name],
            }
            for name, template in PROMPT_VARIANTS.items()
        },
        "evaluation_method": {
            "execution_mode": "offline_fixture_evaluation",
            "llm_called": False,
            "groundedness": (
                "Fraction of answer claims with at least 50% meaningful "
                "token overlap with the evidence."
            ),
            "context_relevance": (
                "Fraction of meaningful question tokens found in the evidence."
            ),
            "faithfulness": (
                "Fraction of meaningful answer tokens found in the evidence."
            ),
            "hallucination_rate": (
                "Fraction of answer claims below the lexical-support threshold."
            ),
            "limitations": [
                "These are lexical overlap proxies, not semantic LLM-judge metrics.",
                "Token overlap cannot reliably establish entailment or factual correctness.",
                "Prompt variants use prewritten answer fixtures; prompts are not executed by an LLM.",
                "Illustrative evidence must be replaced with verified policy excerpts before policy-validation claims are made.",
                "The reference answer is recorded for review but is not used to calculate these four metrics."
            ],
        },
        "aggregate_metrics": aggregate_results,
        "changes_evidence_first_minus_baseline": changes,
        "scenario_results": results_by_variant,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Create a human-readable report as well.
    lines = [
        "LLMOps OFFLINE EVALUATION REPORT",
        "=" * 38,
        f"Scenarios: {len(scenarios)}",
        "LLM API called: No",
        "Evaluation type: Lexical proxy using fixed answer fixtures",
        "",
        "AGGREGATE METRICS (0.0 to 1.0)",
        "-" * 38,
    ]

    labels = {
        "groundedness": "Groundedness",
        "context_relevance": "Context relevance",
        "faithfulness": "Faithfulness",
        "hallucination_rate": "Hallucination rate",
    }

    for metric in metric_names:
        lines.append(
            f"{labels[metric]}: "
            f"baseline={baseline[metric]:.4f}, "
            f"evidence-first={evidence_first[metric]:.4f}, "
            f"change={changes[metric]:+.4f}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 38,
        "Higher is better: groundedness, context relevance, faithfulness.",
        "Lower is better: hallucination rate.",
        "A lexical score is only a screening signal, not proof of correctness.",
        "",
        "LIMITATIONS",
        "-" * 38,
        "- No real LLM was called.",
        "- Answers are prewritten test fixtures.",
        "- Evidence is illustrative and must be verified against the actual policy.",
        "- Review individual scenario results before drawing conclusions.",
        "",
        f"JSON results: {OUTPUT_PATH}",
        f"Text report: {REPORT_PATH}",
    ])

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nDetailed scenario results saved.")
    print(f"JSON: {OUTPUT_PATH}")
    print(f"Text: {REPORT_PATH}")


if __name__ == "__main__":
    main()
