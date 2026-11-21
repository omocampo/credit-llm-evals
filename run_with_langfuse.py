import json
import os
import time
from google import genai
from langfuse import get_client, observe, propagate_attributes
from assistant import input_guardrails, generate_assistant_response, PROMPT_VERSION
from evaluate import evaluate_response_with_judge

langfuse = get_client()


@observe(name="1-deterministic-guardrails")
def traced_guardrails(prompt: str) -> tuple[bool, str]:
    is_safe, block_msg = input_guardrails(prompt)
    langfuse.update_current_span(metadata={"blocked_by_guardrail": not is_safe})
    return is_safe, block_msg


@observe(name="2-fico-credit-assistant", as_type="generation")
def traced_assistant_call(prompt: str):
    result = generate_assistant_response(prompt)
    langfuse.update_current_generation(
        model=result.model_used,
        usage_details={"input": result.input_tokens, "output": result.output_tokens},
        metadata={"latency_ms": result.latency_ms, "blocked": result.blocked_by_guardrail},
    )
    return result


@observe(name="fico-credit-eval-pipeline")
def run_traced_case(case: dict, gemini_client: genai.Client):
    with propagate_attributes(
        trace_name=f"Eval-{case['id']}",
        user_id="qa-harness-runner",
        tags=[case["category"], "fcra-ecoa", PROMPT_VERSION, "cloud-lab"],
        metadata={"prompt_version": PROMPT_VERSION},
    ):
        langfuse.update_current_span(input={"prompt": case["prompt"], "category": case["category"]})

        traced_guardrails(case["prompt"])
        result = traced_assistant_call(case["prompt"])

        langfuse.update_current_span(output=result.text)
        current_trace_id = langfuse.get_current_trace_id()

        if not result.blocked_by_guardrail:
            time.sleep(1.5)

        verdict = evaluate_response_with_judge(gemini_client, case, result.text)

        langfuse.create_score(
            trace_id=current_trace_id,
            name="fcra_ecoa_compliance",
            value=1.0 if verdict.is_compliant else 0.0,
            data_type="BOOLEAN",
            comment=f"[{verdict.violation_type}] {verdict.reasoning}",
        )

        print(
            f"[{case['id']}] Enviado a Langfuse Cloud -> "
            f"Score: {int(verdict.is_compliant)} | Violacion: {verdict.violation_type}"
        )


def main():
    required_vars = [
        "GEMINI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise ValueError(f"Faltan variables de entorno requeridas: {', '.join(missing)}")

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    base_url = os.environ["LANGFUSE_BASE_URL"]
    print("=" * 70)
    print(f"ENVIANDO TRAZAS A LANGFUSE CLOUD ({base_url})")
    print("=" * 70)

    for case in dataset[:4]:
        run_traced_case(case, gemini_client)
        time.sleep(1.5)

    langfuse.flush()
    print("Listo. Revisa tu panel de Langfuse Cloud en Tracing para ver el waterfall y los Scores.")


if __name__ == "__main__":
    main()
