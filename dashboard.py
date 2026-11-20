import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="FICO Credit AI - QA & Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
)

RESULTS_FILE = Path("eval_results.json")
GOLDEN_FILE = Path("golden_dataset.json")

st.title("🛡️ FICO Credit AI — Transcript Viewer & Compliance QA")
st.caption(
    "Herramienta interna para Conversation Analysts, QA Engineers y Legal/Compliance (FCRA / ECOA / UDAAP)."
)

if not RESULTS_FILE.exists():
    st.warning(
        "No se encontró `eval_results.json`. Ejecuta primero `python3 evaluate.py` en tu terminal para generar los datos."
    )
    st.stop()

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

metrics = data["metrics"]
records = data["results"]

st.sidebar.header("🔍 Filtros de Auditoría")
categories = ["Todas"] + sorted(list({r["category"] for r in records}))
selected_cat = st.sidebar.selectbox("Categoría de Prueba", categories)

status_filter = st.sidebar.radio(
    "Estado del Veredicto",
    [
        "Todos",
        "Solo Fallos (Non-Compliant)",
        "Desacuerdos (Juez vs Humano)",
        "Bloqueados por Guardrail",
    ],
)

st.sidebar.divider()
st.sidebar.header("📥 Gestión del Golden Dataset")
st.sidebar.write("Descarga el dataset actualizado con las correcciones humanas realizadas en esta sesión.")

if GOLDEN_FILE.exists():
    with open(GOLDEN_FILE, "r", encoding="utf-8") as gf:
        golden_data = json.load(gf)
    st.sidebar.download_button(
        label="⬇️ Exportar golden_dataset.json",
        data=json.dumps(golden_data, indent=2, ensure_ascii=False),
        file_name="golden_dataset_updated.json",
        mime="application/json",
    )

st.subheader("📊 Métricas de Confiabilidad del Evaluador y Rendimiento (SUT)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Judge Precision", f"{metrics['precision']:.2f}")
col2.metric("Judge Recall", f"{metrics['recall']:.2f}")
col3.metric("Latencia P95 (SUT)", f"{metrics['p95_latency_ms']:.0f} ms")
col4.metric(
    "Tokens Totales (SUT)",
    f"{metrics['total_input_tokens']} in / {metrics['total_output_tokens']} out",
)

st.divider()

filtered_records = []
for r in records:
    if selected_cat != "Todas" and r["category"] != selected_cat:
        continue
    if status_filter == "Solo Fallos (Non-Compliant)" and r["judge_is_compliant"]:
        continue
    if status_filter == "Desacuerdos (Juez vs Humano)" and (
        r["judge_is_compliant"] == r["human_label_compliant"]
    ):
        continue
    if status_filter == "Bloqueados por Guardrail" and not r["blocked_by_guardrail"]:
        continue
    filtered_records.append(r)

st.subheader(f"💬 Visor de Transcripciones ({len(filtered_records)} casos mostrados)")

for idx, rec in enumerate(filtered_records):
    badge = "✅ COMPLIANT" if rec["judge_is_compliant"] else "🚨 VIOLATION DETECTED"
    agreement = (
        "🤝 Acuerdo Juez/Humano"
        if rec["judge_is_compliant"] == rec["human_label_compliant"]
        else "⚠️ DESACUERDO Juez/Humano"
    )

    with st.expander(
        f"[{rec['id']}] {rec['category']}  |  {badge}  |  {agreement}",
        expanded=not rec["judge_is_compliant"],
    ):
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("**👤 Prompt del Usuario:**")
            st.info(rec["prompt"])

            st.markdown(
                f"**🤖 Respuesta del Asistente** (`{rec['model_used']}` • `{rec['latency_ms']} ms`):"
            )
            if rec["blocked_by_guardrail"]:
                st.warning(rec["assistant_reply"])
            else:
                st.success(rec["assistant_reply"])

        with c_right:
            st.markdown("**⚖️ Veredicto LLM-as-a-Judge (Structured Output):**")
            st.write(f"- **Compliant:** `{rec['judge_is_compliant']}`")
            st.write(f"- **Violation Type:** `{rec['judge_violation_type']}`")
            st.write(f"- **Reasoning:** {rec['judge_reasoning']}")

            st.divider()
            st.markdown("**🧑‍💻 Revisión Humana (Human-in-the-Loop):**")

            if rec.get("is_synthetic"):
                st.caption("Caso sintético de prueba negativa (no editable en golden_dataset.json).")
            else:
                new_label = st.radio(
                    "Etiqueta Humana (Ground Truth):",
                    options=[True, False],
                    format_func=lambda x: "Cumple (True)" if x else "No Cumple (False)",
                    index=0 if rec["human_label_compliant"] else 1,
                    key=f"radio_{rec['id']}_{idx}",
                    horizontal=True,
                )

                if st.button("💾 Guardar Etiqueta en Golden Dataset", key=f"btn_{rec['id']}_{idx}"):
                    rec["human_label_compliant"] = new_label
                    with open(RESULTS_FILE, "w", encoding="utf-8") as wf:
                        json.dump(data, wf, indent=2, ensure_ascii=False)

                    if GOLDEN_FILE.exists():
                        with open(GOLDEN_FILE, "r", encoding="utf-8") as gf:
                            g_list = json.load(gf)
                        for item in g_list:
                            if item["id"] == rec["id"]:
                                item["human_label_compliant"] = new_label
                        with open(GOLDEN_FILE, "w", encoding="utf-8") as gf:
                            json.dump(g_list, gf, indent=2, ensure_ascii=False)

                    st.toast(f"Etiqueta actualizada para {rec['id']} -> {new_label}", icon="✅")
                    st.rerun()
