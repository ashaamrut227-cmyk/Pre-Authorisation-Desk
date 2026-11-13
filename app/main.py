
import logging

from fastapi import FastAPI, HTTPException

from app.schemas import PreAuthRequest
from src.agent_metrics import agent_metrics
from src.agents.agent_graph import run_agent_workflow


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Insurance Pre-Authorisation Desk",
    description=(
        "A policy-grounded, two-agent pre-authorisation support system. "
        "All recommendations require human review."
    ),
    version="0.7.0",
)


@app.get("/")
def home():
    return {
        "message": "Health Insurance Pre-Authorisation Desk API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
def get_metrics():
    """Return in-memory agent execution metrics."""
    return {
        "metrics_type": "in_memory",
        "note": "Metrics reset when the application restarts.",
        "agents": agent_metrics.snapshot(),
    }


@app.post("/preauth")
def submit_preauthorisation(request: PreAuthRequest):
    return {
        "request_id": request.request_id,
        "status": "received",
        "admission_type": request.admission_type.value,
        "message": "Request validated successfully.",
        "human_review_required": True,
        "final_coverage_decision": None,
    }


@app.post("/preauth/evaluate")
def evaluate_preauthorisation(request: PreAuthRequest):
    request_data = request.model_dump(mode="json")

    try:
        result = run_agent_workflow(request_data)

        logger.info(
            "Pre-authorisation workflow completed",
            extra={
                "request_id": request.request_id,
                "workflow_status": result.get("status"),
                "route": result.get("route"),
            },
        )

        return {
            "request_id": request.request_id,
            "workflow_result": result,
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    except ValueError as error:
        logger.warning(
            "Invalid pre-authorisation workflow input: %s",
            error,
        )
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Pre-authorisation workflow failed for request %s",
            request.request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Agent workflow failed. Check the server logs.",
        ) from error

# Frontend served by the same FastAPI application.
from pathlib import Path
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/ui", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
