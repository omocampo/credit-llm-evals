# `credit-llm-evals`: Financial AI Compliance, Guardrails & Evaluation Harness

An end-to-end automated evaluation, guardrail, observability, and human-in-the-loop QA framework built in Python for consumer credit AI assistants. Designed around regulated financial domain standards (**FCRA**, **ECOA**, **UDAAP**), this project treats LLM quality, regulatory safety, and operational cost/latency as measurable software engineering metrics.

🔗 **Live QA & Transcript Viewer Dashboard:** [credit-llm-evals-oof.streamlit.app](https://streamlit.app)

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
└──────────────────────────────────────────┘     SHA-256 Prompt Versioning, Latency (ms) & Model Routing
     │
     ├──────────────────────────────────────────────────────┐
     ▼                                                      ▼
┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ 3a. Custom Judge & DeepEval Pytest Suite │   │ 3b. Langfuse Cloud Observability         │
│   • Pydantic Structured Outputs          │   │   • OpenTelemetry Waterfall Tracing      │
│   • Precision & Recall (scikit-learn)    │   │   • Automated Compliance Score Logging   │
│   • DeepEval BiasMetric & G-Eval Gates   │   │   • Prompt Version Tagging (SHA-256)     │
└──────────────────────────────────────────┘   └──────────────────────────────────────────┘
     │
     ▼ (eval_results.json + CI/CD Quality Gate >= 0.95)
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
* **`assistant.py`**: Simulated FICO credit assistant featuring deterministic regex/heuristic input guardrails, strict system instructions with automatic **SHA-256 prompt versioning** (`v1.0.0-<hash>`), automatic model routing/fallback (`gemini-3.8-flash` -> `flash-lite`), and telemetry capture (latency and token counts).
* **`evaluate.py`**: Regression harness using **LLM-as-a-Judge** enforced with **Pydantic Structured Outputs** (`response_schema`). Computes **Precision** and **Recall** against human-labeled ground truth (`scikit-learn`), calculates P95 latency, exports detailed audit records to `eval_results.json`, and enforces a **CI/CD Quality Gate** (`sys.exit(1)` if Precision or Recall drops below `0.95`).
* **`test_deepeval.py`**: Standardized `pytest` regression suite powered by **DeepEval** (`BiasMetric` and `GEval` backed by Gemini) to gate ECOA fairness neutrality, numerical hallucination prevention, and financial advice boundaries on every code change.
* **`run_with_langfuse.py`**: Observability pipeline instrumented with the **Langfuse SDK** (`@observe` & OpenTelemetry) sending waterfall traces, token usage, prompt hashes, and automated judge compliance scores to **Langfuse Cloud**.
* **`dashboard.py`**: Interactive **Streamlit** internal tool for Conversation Analysts, QA Engineers, and Legal/Compliance teams to inspect transcripts, filter judge/human disagreements, and update ground-truth labels ([Live Demo](https://streamlit.app)).
* **`.github/workflows/`**: Automated CI/CD pipelines (`evals.yml` for regression gating and `langfuse_traces.yml` for observability telemetry).

## Quickstart

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export GEMINI_API_KEY="your_google_ai_studio_api_key"

# Optional: Required only for Langfuse Cloud tracing
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://langfuse.com"
```

### 3. Run the Evaluation Suites
```bash
# Run custom LLM-as-a-Judge harness (Precision/Recall + CI/CD Quality Gate)
python3 evaluate.py

# Run DeepEval pytest suite (BiasMetric & G-Eval)
deepeval test run test_deepeval.py

# Send waterfall traces and judge scores to Langfuse Cloud
python3 run_with_langfuse.py
```

### 4. Launch the QA & Transcript Viewer Dashboard
```bash
streamlit run dashboard.py
```

## Production Scalability Notes
While prototyped with Google Gemini (`google-genai`), this evaluation architecture is provider-agnostic and maps directly to enterprise JVM/AWS stacks:
* **AWS Bedrock Integration**: Input guardrails and structured judge outputs translate directly to *Amazon Bedrock Guardrails* and *Converse API Tool Use*.
* **Enterprise Microservices (Java / Spring Boot)**: Low-latency deterministic guardrails run synchronously in Spring Boot microservices, while `evaluate.py` and observability workers execute asynchronously via SQS/Kafka and persist telemetry to S3/DynamoDB/Redshift for SQL-based analyst reporting.
