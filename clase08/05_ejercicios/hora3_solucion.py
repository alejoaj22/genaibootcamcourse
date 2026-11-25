# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 3 (Solución) — Propuesta Formal del Prototipo Funcional
# MAGIC **Objetivo:** solución de referencia completa de una Propuesta Formal de Prototipo Funcional lista para ejecutar el Sprint de la Clase 9.
# MAGIC **Tiempo estimado:** 10 minutos.
# MAGIC **Concepto que ilustra:** especificación profesional de MVP, arquitectura desacoplada Endpoint ↔ App, contrato JSON y roles en equipo mixto (RA5).
# MAGIC **Requisitos previos:** ninguno.

# COMMAND ----------

import json

propuesta_equipo = {
    "equipo": "Equipo 01 — dPhone Smart Support",
    "titulo_proyecto": "Asistente Inteligente de Triage y Respuesta para Soporte dPhone",
    "problema": {
        "usuario_objetivo": "Analista de soporte al cliente Nivel 1 en el centro de atención de dPhone.",
        "dolor_actual": "Debe abrir 3 sistemas distintos (manuales PDF, tabla de inventario y tickets abiertos) tardando ~8 minutos por caso y cometiendo errores en políticas de garantía.",
        "objetivo_medible": "Reducir el tiempo de diagnóstico y redacción de borrador a menos de 60 segundos con >= 85% de corrección factual y 0 violaciones de cumplimiento.",
    },
    "alcance_mvp": {
        "regla_1_1_1": {
            "usuario": "1 analista de soporte Nivel 1",
            "flujo": "Seleccionar ticket abierto (pre-evaluado en batch o ingreso manual) -> consultar Serving Endpoint dphone_endpoint -> revisar borrador y aprobar respuesta",
            "metrica": "Correctness >= 0.85, RetrievalGroundedness >= 0.85, Safety == 1.0 y policy_compliance == 1.0 en MLflow",
        },
        "in_scope": [
            "Agente dphone_agent v2 (@challenger promovido a @champion) con RAG sobre dphone_vs_index y UC Functions de catálogo e inventario.",
            "Despliegue en Serving Endpoint (dphone_endpoint, Small, scale_to_zero=True) y flujo batch diario con ai_query() sobre customer_support_tickets.",
            "Aplicación web ligera (Databricks App / Streamlit en Clase 9) conectada al endpoint vía SDK/REST con monitoreo online.",
        ],
        "out_of_scope": [
            "Envío automático de respuestas al cliente sin aprobación humana.",
            "Integración bidireccional con CRM externo (Salesforce/Zendesk) o pasarela de pagos real.",
            "Soporte de voz en tiempo real o canales WhatsApp/Instagram en esta fase.",
        ],
    },
    "arquitectura": {
        "capa_datos": "Unity Catalog (<catalog>.genai_course): smartphones, inventory_stock, customer_support_tickets, gold_docs_chunked e índice dphone_vs_index.",
        "capa_ia_serving": "Modelo <catalog>.genai_course.dphone_agent (@champion, v2) expuesto en Serving Endpoint dphone_endpoint (workload_size='Small', scale_to_zero_enabled=True).",
        "capa_aplicacion": "Interfaz web en Databricks Apps / Streamlit (Clase 9) usando la clase DPhoneEndpointClient sobre POST /serving-endpoints/dphone_endpoint/invocations.",
        "contrato_json": {
            "request": {
                "input": [{"role": "user", "content": "Customer asks if dPhone D1 Pro is waterproof and in stock."}],
                "custom_inputs": {"session_id": "ticket-1042"},
            },
            "response": {
                "request_id": "a1b2c3d4",
                "status": "ok",
                "answer_text": "The dPhone D1 Pro is water-resistant... Current stock: 25 units.",
                "latency_ms": 2350,
            },
        },
    },
    "roles_equipo_mixto": [
        {
            "rol": "Data & AI Engineer",
            "responsable": "Laura Gómez",
            "clase_8": "Evaluación Champion/Challenger en MLflow, despliegue de dphone_endpoint y tabla batch ai_query()",
            "clase_9": "Activación de Inference Tables y Online Monitoring (Notebook 4.2)",
        },
        {
            "rol": "App / Backend Developer",
            "responsable": "Carlos Restrepo",
            "clase_8": "Cliente Python DPhoneEndpointClient con manejo de timeouts, cold start y contrato JSON",
            "clase_9": "Construcción de la UI conectada al endpoint y lectura de tickets pre-procesados",
        },
        {
            "rol": "Product / Negocio & QA",
            "responsable": "Mariana Vélez",
            "clase_8": "Definición MVP 1-1-1, curaduría del Golden Set (12+ casos) y redacción de 3 Guidelines de negocio",
            "clase_9": "Pruebas de aceptación con usuarios piloto y guion ejecutivo de Demo Day",
        },
    ],
    "riesgos": [
        {
            "riesgo": "Latencia alta por cold start (>45 s) durante la demostración en vivo.",
            "mitigacion": "Realizar una consulta de calentamiento 5 minutos antes de presentar o desactivar temporalmente scale_to_zero durante la demo.",
        },
        {
            "riesgo": "El agente promete descuentos o fechas de reposición fuera de política.",
            "mitigacion": "Reglas explícitas en SYSTEM_PROMPT y compuerta bloqueante de policy_compliance == 1.0 antes de promover versiones.",
        },
        {
            "riesgo": "Ampliación de alcance (scope creep) intentando conectar APIs externas sin credenciales.",
            "mitigacion": "Alcance congelado en 3 ítems IN-SCOPE; toda integración externa queda documentada en OUT-OF-SCOPE (v2).",
        },
    ],
}

# COMMAND ----------

# Validación y exportación en formato Markdown completo
texto_completo = json.dumps(propuesta_equipo, ensure_ascii=False)
assert "TODO" not in texto_completo
assert 1 <= len(propuesta_equipo["alcance_mvp"]["in_scope"]) <= 3
assert len(propuesta_equipo["alcance_mvp"]["out_of_scope"]) >= 3
assert len(propuesta_equipo["roles_equipo_mixto"]) >= 2
assert len(propuesta_equipo["riesgos"]) >= 2

lineas = [
    f"# {propuesta_equipo['titulo_proyecto']}",
    f"**Equipo:** {propuesta_equipo['equipo']}",
    "",
    "## 1. Problema y Propuesta de Valor",
    f"- **Usuario:** {propuesta_equipo['problema']['usuario_objetivo']}",
    f"- **Dolor:** {propuesta_equipo['problema']['dolor_actual']}",
    f"- **Objetivo:** {propuesta_equipo['problema']['objetivo_medible']}",
    "",
    "## 2. Alcance MVP (Regla 1-1-1)",
    f"- **1 Usuario:** {propuesta_equipo['alcance_mvp']['regla_1_1_1']['usuario']}",
    f"- **1 Flujo:** {propuesta_equipo['alcance_mvp']['regla_1_1_1']['flujo']}",
    f"- **1 Métrica:** {propuesta_equipo['alcance_mvp']['regla_1_1_1']['metrica']}",
    "- **IN-SCOPE:**\n  - " + "\n  - ".join(propuesta_equipo["alcance_mvp"]["in_scope"]),
    "- **OUT-OF-SCOPE (v2):**\n  - " + "\n  - ".join(propuesta_equipo["alcance_mvp"]["out_of_scope"]),
    "",
    "## 3. Arquitectura y Contrato JSON",
    f"- **Datos:** {propuesta_equipo['arquitectura']['capa_datos']}",
    f"- **IA / Serving:** {propuesta_equipo['arquitectura']['capa_ia_serving']}",
    f"- **Aplicación:** {propuesta_equipo['arquitectura']['capa_aplicacion']}",
    "```json",
    json.dumps(propuesta_equipo["arquitectura"]["contrato_json"], indent=2, ensure_ascii=False),
    "```",
    "",
    "## 4. Roles en Equipo Mixto",
    "| Rol | Responsable | Entregable Clase 8 | Entregable Clase 9 |",
    "|---|---|---|---|",
]
for r in propuesta_equipo["roles_equipo_mixto"]:
    lineas.append(f"| {r['rol']} | {r['responsable']} | {r['clase_8']} | {r['clase_9']} |")

lineas.append("\n## 5. Riesgos y Mitigaciones")
for rk in propuesta_equipo["riesgos"]:
    lineas.append(f"- **{rk['riesgo']}** → {rk['mitigacion']}")

print("\n".join(lineas))

# COMMAND ----------

# Salida esperada:
# - Validación superada sin errores (`AssertionError`).
# - Documento completo de Propuesta Formal en Markdown con las 5 secciones (Problema, MVP 1-1-1, Arquitectura + JSON, Roles y Riesgos).
