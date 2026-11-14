# Health Insurance Pre-Authorisation Desk – MLOps

## 1. Project Overview

The Health Insurance Pre-Authorisation Desk is an AI-assisted system designed to support health insurance pre-authorisation assessment.

It uses a two-agent workflow to process admission information, retrieve relevant policy evidence, verify evidence and generate a recommendation for human review.

The system is designed to support decision-making. It does not issue final insurance coverage approvals or denials.

## 2. Key Features

- **Two-agent architecture:**
  - Intake Agent: processes request details and classifies admission information.
  - Policy Agent: retrieves relevant policy evidence and supports verification.
- Policy document retrieval using FAISS and sentence-transformer embeddings.
- Evidence verification and decision routing.
- Human-review routing for uncertain or unverified cases.
- Redis-backed asynchronous task queue.
- Three background worker containers.
- Reliability controls including timeout, retry and circuit-breaker functionality.
- Redis-shared agent execution metrics.
- FastAPI backend and browser-based frontend.
- Automated tests using Pytest.
- Load testing using Locust.
- Docker-based deployment using Docker Compose.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Data validation | Pydantic |
| Agent orchestration | LangGraph |
| Document processing | PyPDF |
| Retrieval | FAISS |
| Embeddings | Sentence Transformers |
| Queue and shared metrics | Redis |
| Monitoring | Prometheus Client |
| Frontend | HTML, CSS and JavaScript |
| Containerisation | Docker and Docker Compose |
| Testing | Pytest, HTTPX |
| Load testing | Locust |

## 4. Project Architecture

The application follows a two-agent architecture supported by a FastAPI backend and Redis-based asynchronous processing.

### Synchronous workflow

1. User submits a pre-authorisation request through the frontend.
2. FastAPI validates the request.
3. The Intake Agent processes and classifies the request.
4. The Policy Agent retrieves relevant policy evidence.
5. Evidence verification evaluates whether the retrieved information supports the requested assessment.
6. The decision router determines the appropriate recommendation or human-review route.
7. Results and execution metrics are recorded.
8. The system presents the result while retaining the requirement for authorised human review.

### Asynchronous workflow

For asynchronous requests:

1. FastAPI receives and validates the request.
2. The request is added to a Redis-backed task queue.
3. A background worker retrieves the queued job.
4. The agent workflow processes the request.
5. The run status and result are stored in Redis.
6. The user can retrieve the result using the asynchronous status endpoint.

Three background worker containers are configured in Docker Compose to process jobs.

## 5. Project Structure

```text
mlops_preauth/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── corpus/
│   ├── policy_wording.pdf
│   ├── policy_extracted.txt
│   ├── policy_pages.jsonl
│   └── policy_index/
│
├── evaluation/
│
├── frontend/
│   └── index.html
│
├── scripts/
│
├── src/
│   ├── agents/
│   │   ├── agent_graph.py
│   │   ├── intake_agent.py
│   │   ├── policy_agent.py
│   │   ├── tools.py
│   │   └── __init__.py
│   │
│   ├── agent_metrics.py
│   ├── admission_classifier.py
│   ├── decision_router.py
│   ├── evidence_verifier.py
│   ├── policy_retriever.py
│   ├── policy_workflow.py
│   ├── reliability.py
│   ├── task_queue.py
│   └── worker.py
│
├── tests/
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── project_files.txt
├── README.md
└── requirements.txt
```

## 6. Running the Project on macOS

### 6.1 Create and activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.3 Run automated tests

```bash
pytest -q
```

### 6.4 Start the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

When running locally, the frontend can be accessed through the API's `/ui` route if the required policy files and dependencies are available.

Open:

```text
http://localhost:8000/ui
```

For the Docker deployment, use port 8001 as described in the next section.

## 7. Running with Docker Compose

Make sure Docker Desktop is running.

From the project root, build and start the services:

```bash
docker compose up --build -d
```

### Check container status

```bash
docker compose ps
```

The Compose configuration starts:

- FastAPI application container.
- Redis container.
- Three agent-worker containers.

### Application URLs

The current Docker Compose configuration exposes the API on host port 8001.

| Service | URL |
|---|---|
| API root | http://localhost:8001/ |
| Frontend | http://localhost:8001/ui |
| Health check | http://localhost:8001/health |
| Agent metrics | http://localhost:8001/metrics |
| Prometheus metrics | http://localhost:8001/metrics/prometheus |
| API documentation | http://localhost:8001/docs |

### Check API health

```bash
curl http://localhost:8001/health
```

### View agent metrics

```bash
curl -s http://localhost:8001/metrics | python3 -m json.tool
```

### View worker logs

```bash
docker compose logs --tail=50 agent-worker
```

### Check Redis queue length

```bash
docker compose exec redis redis-cli LLEN preauth:jobs
```

### Stop the services

```bash
docker compose down
```

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API root |
| GET | `/health` | Health check |
| GET | `/metrics` | Redis-shared agent execution metrics |
| GET | `/metrics/prometheus` | Prometheus metrics |
| GET | `/ui` | Frontend interface |
| POST | `/preauth` | Receive a pre-authorisation request |
| POST | `/preauth/evaluate` | Run synchronous evaluation |
| POST | `/preauth/async` | Submit asynchronous evaluation |
| GET | `/preauth/status/{run_id}` | Retrieve asynchronous run status |

## 9. Testing and Validation

The following results were observed during project verification.

### 9.1 Automated testing

The project test suite completed successfully:

```text
56 passed in 8.52s
```

### 9.2 Docker Compose verification

The following services were observed running:

- FastAPI API container.
- Redis container with healthy status.
- Three agent-worker containers.

### 9.3 Redis queue verification

The Redis queue length returned zero after the observed load test processing:

```text
(integer) 0
```

### 9.4 Asynchronous workflow verification

An asynchronous pre-authorisation request was submitted successfully.

The API returned a run ID and queued status. The background worker processed the request, and the status endpoint subsequently returned a completed result.

### 9.5 Frontend integration verification

The browser-based frontend successfully communicated with the backend and displayed the evaluation response.

The workflow retrieved cataract-related policy evidence and returned an `unverified` result requiring human review.

### 9.6 Agent metrics verification

The Redis-shared metrics endpoint returned execution statistics for both agents.

An observed metrics snapshot recorded:

| Metric | Intake Agent | Policy Agent |
|---|---:|---:|
| Executions | 1,768 | 1,768 |
| Successes | 1,768 | 1,768 |
| Failures | 0 | 0 |
| Human-review routes | 1,768 | 1,768 |

These values represent the observed metrics snapshot during testing.

### 9.7 Load testing

Locust was used to exercise the asynchronous pre-authorisation endpoint.

An observed test screenshot recorded:

| Metric | Observed result |
|---|---:|
| Concurrent users reached | 30 |
| Requests | 670 |
| HTTP failures | 0 |
| Median response time | 9 ms |
| 95th percentile | 18 ms |
| 99th percentile | 32 ms |

These figures represent an observed load-test snapshot, not a guarantee of production capacity. HTTP request success does not by itself establish successful completion of every asynchronous workflow.

## 10. Load Testing with Locust

Make sure the Docker Compose services are running before starting the load test.

From the project root, activate the virtual environment and run:

```bash
source .venv/bin/activate
locust -f locustfile.py --host http://localhost:8001
```

Open the Locust web interface:

```text
http://localhost:8089
```

Configure the number of users and ramp-up rate in the Locust interface, then start the test.

The project was exercised with up to 30 concurrent users. One observed test snapshot recorded 670 requests and zero HTTP failures.

## 11. Human Review and Safety

The application is intended as a decision-support system.

An `unverified` result does not represent an approval or denial. It indicates that the available evidence did not establish a verified coverage conclusion.

The system retrieves policy evidence, evaluates the available information and routes uncertain cases for human review.

All recommendations must be reviewed by an authorised person.

The application does not issue final insurance coverage decisions.

## 12. Configuration

Use `.env.example` as a reference for environment variables.

The Docker Compose configuration sets the Redis service URL to:

```text
redis://redis:6379/0
```

The API and worker containers use the same Redis service so that queued jobs, run statuses and shared agent metrics are accessible across containers.

Do not commit API keys, credentials or other secrets to version control.

## 13. Future Improvements

- Expand automated tests for failure and recovery scenarios.
- Add more comprehensive asynchronous workflow monitoring.
- Improve load testing with longer-duration and higher-concurrency tests.
- Add persistent evaluation and audit reporting.
- Improve deployment security and production observability.
- Add more comprehensive policy coverage and eligibility verification scenarios.

## 14. Project Status

The core application, two-agent workflow, policy retrieval, evidence verification, asynchronous processing infrastructure, frontend integration, automated tests and initial load testing have been implemented and exercised.

The project has been tested locally and through Docker Compose.

Final submission should include:

- Complete project source code.
- README and setup instructions.
- Architecture diagram.
- Frontend and backend screenshots.
- Automated test evidence.
- Agent metrics evidence.
- Locust load-testing evidence.
- Evaluation documentation.

The system is an AI-assisted pre-authorisation decision-support application. All recommendations remain subject to authorised human review.
