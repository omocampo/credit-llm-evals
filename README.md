# `credit-llm-evals`: Financial AI Compliance, Guardrails & Evaluation Harness

An end-to-end automated evaluation, guardrail, and human-in-the-loop QA framework built in Python for consumer credit AI assistants. Designed around regulated financial domain standards (**FCRA**, **ECOA**, **UDAAP**), this project treats LLM quality, regulatory safety, and operational cost/latency as measurable software engineering metrics.

## Architecture & Core Components

```text
User Prompt
     │
     ▼
┌──────────────────────────────────────────┐
│ 1. Deterministic Input Guardrails        │ ──► Blocks SSN/Credit Card PII & Prompt Injections
└──────────────────────────────────────────┘     (Zero LLM token cost, <1ms latency)
     │ (Safe)
     ▼
┌──────────────────────────────────────────┐
│ 2. Credit Education Assistant (SUT)      │ ──► Enforces FCRA/ECOA compliance & anti-hallucination
└──────────────────────────────────────────┘     Tracks latency (ms), tokens, & Model Routing
     │
     ▼
┌──────────────────────────────────────────┐
│ 3. LLM-as-a-Judge (Structured Outputs)   │ ──► Audits responses via Pydantic JSON schemas
└──────────────────────────────────────────┘     Benchmarks Precision & Recall vs. Human Labels
     │
     ▼ (eval_results.json)
┌──────────────────────────────────────────┐
│ 4. QA & Compliance Dashboard (Streamlit) │ ──► Transcript Viewer, Failure Filtering &
└──────────────────────────────────────────┘     Human-in-the-Loop Golden Dataset Labeling
```

### Repository Structure
* **`golden_dataset.json`**: Curated regression test suite covering 8 real-world and adversarial scenarios:
  * *Valid Educational & FCRA Rights* (`TC-01`, `TC-02`)
  * *Financial Advice Compliance Traps* (`TC-03`)
  * *FICO Score Numerical Hallucination Traps* (`TC-04`)
  * *PII Leak & Prompt Injection Attacks* (`TC-05`, `TC-06`)
  * *Fairness & Bias Auditing under ECOA* — ZIP code, marital status, and age neutrality (`TC-07`, `TC-08`)
* **`assistant.py`**: Simulated FICO credit assistant featuring deterministic regex/heuristic input guardrails, strict system instructions, automatic model routing/fallback (`gemini-3.8-flash` -> `flash-lite`), and telemetry capture (latency and token counts).
* **`evaluate.py`**: Regression harness using **LLM-as-a-Judge** enforced with **Pydantic Structured Outputs** (`response_schema`). Computes **Precision** and **Recall** against human-labeled ground truth (`scikit-learn`), calculates P95 latency, and exports detailed audit records to `eval_results.json`.
* **`dashboard.py`**: Interactive **Streamlit** internal tool for Conversation Analysts, QA Engineers, and Legal/Compliance teams to inspect transcripts, filter judge/human disagreements, and update ground-truth labels.
* **`.github/workflows/evals.yml`**: CI/CD regression gate running automated evaluations on every push or pull request to `main`.

## Quickstart

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export GEMINI_API_KEY="your_google_ai_studio_api_key"
```

### 3. Run the Evaluation Suite
```bash
python3 evaluate.py
```

### 4. Launch the QA & Transcript Viewer Dashboard
```bash
streamlit run dashboard.py
```

## Production Scalability Notes
While prototyped with Google Gemini (`google-genai`), this evaluation architecture is provider-agnostic and maps directly to enterprise JVM/AWS stacks:
* **AWS Bedrock Integration**: Input guardrails and structured judge outputs translate directly to *Amazon Bedrock Guardrails* and *Converse API Tool Use*.
* **Enterprise Microservices (Java / Spring Boot)**: Low-latency deterministic guardrails run synchronously in Spring Boot microservices, while `evaluate.py` executes asynchronously via SQS/Kafka workers and persists telemetry to S3/DynamoDB/Redshift for SQL-based analyst reporting.
