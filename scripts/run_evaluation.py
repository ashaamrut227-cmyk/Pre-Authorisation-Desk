import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_router import route_pre_authorisation


SCENARIOS_FILE = PROJECT_ROOT / "evaluation" / "scenarios.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results.json"


def evaluate_scenario(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one deterministic decision-routing scenario."""

    request_data = {
        "required_documents_complete": scenario[
            "documents_complete"
        ]
    }

    verification = {
        "status": scenario["verification_status"]
    }

    try:
        actual = route_pre_authorisation(
            request_data=request_data,
            verification=verification,
        )

        expected_route = scenario["expected_route"]
        actual_route = actual.get("route")

        route_matches = actual_route == expected_route
        human_review_is_required = (
            actual.get("human_review_required") is True
        )
        no_final_decision = (
            actual.get("final_coverage_decision") is None
        )

        passed = (
            route_matches
            and human_review_is_required
            and no_final_decision
        )

        return {
            "id": scenario["id"],
            "name": scenario["name"],
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_matches": route_matches,
            "human_review_required": (
                actual.get("human_review_required")
            ),
            "final_coverage_decision": (
                actual.get("final_coverage_decision")
            ),
            "passed": passed,
            "reason": actual.get("reason"),
            "error": None,
        }

    except Exception as error:
        return {
            "id": scenario.get("id", "UNKNOWN"),
            "name": scenario.get("name", "Unnamed scenario"),
            "expected_route": scenario.get("expected_route"),
            "actual_route": None,
            "route_matches": False,
            "human_review_required": None,
            "final_coverage_decision": None,
            "passed": False,
            "reason": None,
            "error": str(error),
        }


def main() -> int:
    """Run all scenarios and save a JSON evaluation report."""

    if not SCENARIOS_FILE.exists():
        print(f"Scenario file not found: {SCENARIOS_FILE}")
        return 1

    try:
        scenarios = json.loads(
            SCENARIOS_FILE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read scenarios: {error}")
        return 1

    if not isinstance(scenarios, list) or not scenarios:
        print("The scenario file must contain a non-empty JSON list.")
        return 1

    results = [
        evaluate_scenario(scenario)
        for scenario in scenarios
    ]

    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed

    report = {
        "evaluation_name": "Pre-authorisation Decision Router",
        "evaluation_type": "deterministic_component",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate_percent": round(
            passed / len(results) * 100, 2
        ),
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['id']} - {result['name']}"
        )

        if not result["passed"]:
            print(
                f"       Expected: {result['expected_route']}; "
                f"Actual: {result['actual_route']}"
            )
            if result["error"]:
                print(f"       Error: {result['error']}")

    print("\nEvaluation summary")
    print("------------------")
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {report['pass_rate_percent']}%")
    print(f"Report: {RESULTS_FILE}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())