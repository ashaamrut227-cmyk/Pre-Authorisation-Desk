
import time
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError

from src.agent_metrics import agent_metrics
from src.agents.intake_agent import run_intake_agent
from src.agents.policy_agent import run_policy_agent

MAX_AGENT_STEPS = 2
GRAPH_RECURSION_LIMIT = 5


class AgentState(TypedDict, total=False):
    request_data: dict[str, Any]
    intake_result: dict[str, Any]
    policy_result: dict[str, Any]
    final_result: dict[str, Any]
    agent_steps: int


def _run_agent_with_metrics(
    agent_name: str,
    agent_function: Callable[[dict[str, Any]], dict[str, Any]],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()

    try:
        result = agent_function(request_data)

        if not isinstance(result, dict):
            raise TypeError(f"{agent_name} must return a dictionary.")

        agent_metrics.record_execution(
            agent_name,
            time.perf_counter() - started_at,
            success=result.get("status") not in {"error", "failed"},
            human_review_required=bool(
                result.get("human_review_required", True)
            ),
        )
        return result

    except Exception:
        agent_metrics.record_execution(
            agent_name,
            time.perf_counter() - started_at,
            success=False,
            human_review_required=True,
        )
        raise


def intake_node(state: AgentState) -> dict[str, Any]:
    result = _run_agent_with_metrics(
        "intake_agent",
        run_intake_agent,
        state.get("request_data", {}),
    )
    return {
        "intake_result": result,
        "agent_steps": state.get("agent_steps", 0) + 1,
    }


def policy_node(state: AgentState) -> dict[str, Any]:
    if state.get("agent_steps", 0) >= MAX_AGENT_STEPS:
        raise RuntimeError("Maximum agent-step limit reached.")

    result = _run_agent_with_metrics(
        "policy_agent",
        run_policy_agent,
        state.get("request_data", {}),
    )
    return {
        "policy_result": result,
        "agent_steps": state.get("agent_steps", 0) + 1,
    }


def finalize_intake(state: AgentState) -> dict[str, Any]:
    intake = state["intake_result"]
    return {
        "final_result": {
            "status": intake["status"],
            "message": intake["reason"],
            "intake": intake,
            "policy_review": None,
            "route": intake.get("handoff_to"),
            "human_review_required": True,
            "final_coverage_decision": None,
        }
    }


def finalize_policy(state: AgentState) -> dict[str, Any]:
    policy = state["policy_result"]
    return {
        "final_result": {
            "status": policy["status"],
            "message": policy["reason"],
            "intake": state["intake_result"],
            "policy_review": policy,
            "route": (policy.get("routing") or {}).get(
                "route", "medical-review"
            ),
            "human_review_required": True,
            "final_coverage_decision": None,
        }
    }


def choose_next(
    state: AgentState,
) -> Literal["policy", "intake_finish"]:
    if state.get("agent_steps", 0) >= MAX_AGENT_STEPS:
        return "intake_finish"

    intake = state.get("intake_result", {})

    if (
        intake.get("status") == "ready_for_policy_review"
        and intake.get("handoff_to") == "policy_agent"
    ):
        return "policy"

    return "intake_finish"


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("policy", policy_node)
    graph.add_node("intake_finish", finalize_intake)
    graph.add_node("policy_finish", finalize_policy)

    graph.add_edge(START, "intake")

    graph.add_conditional_edges(
        "intake",
        choose_next,
        {
            "policy": "policy",
            "intake_finish": "intake_finish",
        },
    )

    graph.add_edge("policy", "policy_finish")
    graph.add_edge("intake_finish", END)
    graph.add_edge("policy_finish", END)

    return graph.compile()


def run_agent_workflow(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(request_data, dict):
        raise ValueError("request_data must be a dictionary.")

    graph = build_agent_graph()

    try:
        result = graph.invoke(
            {
                "request_data": request_data,
                "agent_steps": 0,
            },
            config={
                "recursion_limit": GRAPH_RECURSION_LIMIT,
            },
        )
        return result["final_result"]

    except GraphRecursionError:
        return {
            "status": "error",
            "message": "Agent execution limit reached. Human review required.",
            "intake": None,
            "policy_review": None,
            "route": "medical-review",
            "human_review_required": True,
            "final_coverage_decision": None,
        }
