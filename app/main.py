
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.schemas import PreAuthRequest
from src.agent_metrics import agent_metrics
from src.agents.agent_graph import run_agent_workflow
from src.task_queue import enqueue_run, get_run


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

APP_STARTED_AT = time.time()

HTTP_REQUESTS = Counter(
    "preauth_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "preauth_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

WORKFLOW_EXECUTIONS = Counter(
    "preauth_workflow_executions_total",
    "Total synchronous workflow executions",
    ["status"],
)

WORKFLOW_DURATION = Histogram(
    "preauth_workflow_duration_seconds",
    "End-to-end synchronous workflow duration",
)

CONTAINER_UPTIME = Gauge(
    "preauth_container_uptime_seconds",
    "API process uptime in seconds",
)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    started_at = time.perf_counter()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - started_at

        HTTP_REQUESTS.labels(
            method=method,
            path=path,
            status_code=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=method,
            path=path,
        ).observe(duration)


@app.get("/")
def root():
    return {
        "message": "Health Insurance Pre-Authorisation Desk API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
def get_metrics():
    CONTAINER_UPTIME.set(time.time() - APP_STARTED_AT)

    return {
        "metrics_type": "redis_shared",
        "container_uptime_seconds": round(
            time.time() - APP_STARTED_AT, 2
        ),
        "agents": agent_metrics.snapshot(),
    }


@app.get("/metrics/prometheus", include_in_schema=False)
def prometheus_metrics():
    CONTAINER_UPTIME.set(time.time() - APP_STARTED_AT)

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/preauth")
def submit_preauthorisation(request: PreAuthRequest):
    return {
        "request_id": request.request_id,
        "status": "received",
        "admission_type": request.admission_type.value,
        "message": "Request received. Use /preauth/evaluate for evaluation.",
        "human_review_required": True,
    }


@app.post("/preauth/evaluate")
def evaluate_preauthorisation(request: PreAuthRequest):
    started_at = time.perf_counter()

    try:
        result = run_agent_workflow(
            request.model_dump(mode="json")
        )

        WORKFLOW_EXECUTIONS.labels(status="success").inc()
        return result

    except Exception as error:
        WORKFLOW_EXECUTIONS.labels(status="failure").inc()
        logger.exception("Pre-authorisation workflow failed")

        raise HTTPException(
            status_code=500,
            detail="Workflow failed. Human review required.",
        ) from error

    finally:
        WORKFLOW_DURATION.observe(
            time.perf_counter() - started_at
        )


@app.post("/preauth/async", status_code=202)
def submit_async_preauthorisation(request: PreAuthRequest):
    request_data = request.model_dump(mode="json")

    try:
        run_id = enqueue_run(request_data)

        return {
            "run_id": run_id,
            "request_id": request.request_id,
            "status": "queued",
            "status_url": f"/preauth/status/{run_id}",
            "human_review_required": True,
        }

    except Exception as error:
        logger.exception("Could not enqueue pre-authorisation request")

        raise HTTPException(
            status_code=503,
            detail="Async queue unavailable.",
        ) from error


@app.get("/preauth/status/{run_id}")
def get_preauthorisation_status(run_id: str):
    try:
        result = get_run(run_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Run ID not found",
            )

        return result

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Could not retrieve run status")

        raise HTTPException(
            status_code=503,
            detail="Status service unavailable.",
        ) from error


@app.get("/ui", include_in_schema=False)
def web_interface():
    frontend_file = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "index.html"
    )

    if not frontend_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found.",
        )

    return FileResponse(frontend_file)
