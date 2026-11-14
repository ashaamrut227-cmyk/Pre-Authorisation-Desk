# System Architecture

```mermaid
flowchart TD
    U[User] --> F[Frontend - HTML CSS JavaScript]
    F --> API[FastAPI Backend]

    API --> V[Request Validation - Pydantic]
    V --> IA[Intake Agent]
    IA --> PA[Policy Agent]

    PA --> PR[Policy Retriever]
    PR --> DB[Policy Documents and FAISS Index]
    PA --> EV[Evidence Verification]
    EV --> DR[Decision Router]

    DR --> HR[Human Review Required]
    DR --> R[Assessment Result]
    R --> F

    API --> Q[Redis Task Queue]
    Q --> W1[Worker 1]
    Q --> W2[Worker 2]
    Q --> W3[Worker 3]

    W1 --> AG[Agent Workflow]
    W2 --> AG
    W3 --> AG

    AG --> IA
    AG --> PA
    AG --> RS[Redis Run Status]

    IA --> MET[Agent Metrics]
    PA --> MET
    MET --> RS
    RS --> API
