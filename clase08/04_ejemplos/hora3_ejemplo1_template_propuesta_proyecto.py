# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 1 — Template estructurado de Propuesta de Proyecto (MVP vs PoC vs Producto)
# MAGIC **Objetivo:** demostrar cómo diligenciar, validar automáticamente y exportar en Markdown una Propuesta Formal de Prototipo Funcional aplicando la Regla 1-1-1, delimitación de alcance y asignación de roles en equipos mixtos.
# MAGIC **Tiempo estimado:** 10 minutos.
# MAGIC **Concepto que ilustra:** gestión de proyecto GenAI, diferenciación PoC / MVP / Producto y gobernanza de roles (RA5).
# MAGIC **Requisitos previos:** ninguno (funciona tanto en Databricks como en Python local).

# COMMAND ----------

# 1) Comparativa rápida de referencia: ¿qué estamos construyendo en Clases 8 y 9?
import json

comparativa_alcance = [
    {"Nivel": "PoC (Prueba de Concepto)", "Datos": "Archivo local / muestra", "Evaluación": "Manual / ojo humano", "Serving": "Notebook suelto", "UI": "Ninguna"},
    {"Nivel": "Prototipo Funcional / MVP (META)", "Datos": "Tablas Delta + UC / VS", "Evaluación": "MLflow Golden Set + Guidelines", "Serving": "Endpoint gobernado / ai_query", "UI": "Flujo único usable (Clase 9)"},
    {"Nivel": "Producto en Producción", "Datos": "Pipelines CI/CD multi-fuente", "Evaluación": "CI/CD + Monitoreo continuo 24/7", "Serving": "Multi-región, SLA 99.9%", "UI": "App completa con SSO/RBAC"},
]
for fila in comparativa_alcance:
    print(f"- {fila['Nivel']}: Datos=[{fila['Datos']}] | Eval=[{fila['Evaluación']}] | UI=[{fila['UI']}]")

# COMMAND ----------

# 2) Ejemplo completo de propuesta diligenciada para el caso de referencia "Asistente de Soporte dPhone".
propuesta_ejemplo = {
    "equipo": "Equipo Alfa (Ejemplo de Referencia)",
    "titulo_proyecto": "Copiloto de Soporte y Triage de Tickets dPhone",
    "problema": {
        "usuario_objetivo": "Asesor de soporte técnico Nivel 1 de dPhone",
        "dolor_actual": "Tarda 8 minutos por ticket buscando especificaciones en manuales PDF y consultando stock en tablas dispersas.",
        "objetivo_medible": "Reducir el tiempo de preparación de respuesta a menos de 1 minuto con >= 85% de precisión factual verificada.",
    },
    "alcance_mvp": {
        "regla_1_1_1": {
            "usuario": "1 asesor de soporte nivel 1",
            "flujo": "Recibir ticket abierto -> consultar agente en Endpoint -> mostrar borrador con fuentes y stock para aprobación humana",
            "metrica": "Correctness >= 0.80, RetrievalGroundedness >= 0.85 y Safety == 1.0 en MLflow",
        },
        "in_scope": [
            "Consulta RAG sobre manuales dPhone en dphone_vs_index + UC Functions (inventario y último modelo).",
            "Exposición vía Model Serving Endpoint (dphone_endpoint) y pre-procesamiento batch con ai_query() sobre customer_support_tickets.",
            "Interfaz mínima en Clase 9 para revisar el ticket y aprobar/editar el borrador generado.",
        ],
        "out_of_scope": [
            "Envío automático de correos sin revisión humana.",
            "Integración bidireccional con Zendesk/Salesforce en vivo.",
            "Soporte multi-idioma fuera de inglés/español.",
        ],
    },
    "arquitectura": {
        "capa_datos": "Unity Catalog: smartphones, inventory_stock, customer_support_tickets, dphone_vs_index",
        "capa_ia_serving": "Modelo <catalog>.genai_course.dphone_agent (@champion) en Serving Endpoint dphone_endpoint (Small, scale_to_zero=True)",
        "capa_aplicacion": "Cliente Python / Databricks App (Clase 9) consumiendo REST /serving-endpoints/dphone_endpoint/invocations",
        "contrato_json": {
            "request": {"input": [{"role": "user", "content": "<texto_ticket_o_pregunta>"}]},
            "response": {"answer_text": "<respuesta_final>", "latency_ms": 2400, "status": "ok"},
        },
    },
    "roles_equipo_mixto": [
        {"rol": "Data & AI Engineer", "responsable": "Integrante A", "clase_8": "Evaluación MLflow + Endpoint + Batch SQL", "clase_9": "Monitoreo Online (4.2) + tuning"},
        {"rol": "App / Backend Developer", "responsable": "Integrante B", "clase_8": "Cliente REST/SDK + manejo de timeouts", "clase_9": "Construcción UI conectada al Endpoint"},
        {"rol": "Product / Negocio & QA", "responsable": "Integrante C", "clase_8": "Alcance 1-1-1 + Golden Set (12 casos) + Guidelines", "clase_9": "Pruebas de aceptación y guion Demo Day"},
    ],
    "riesgos": [
        {"riesgo": "Cold start > 45 s en la demo en vivo", "mitigacion": "Calentar endpoint 5 min antes o usar scale_to_zero=False durante la presentación."},
        {"riesgo": "El agente ofrece descuentos no autorizados", "mitigacion": "System prompt estricto + compuerta automática con compliance_guidelines en MLflow."},
    ],
}

# COMMAND ----------

# 3) Validador automático de completitud de la propuesta y renderizado a Markdown limpio.
def validar_y_renderizar_propuesta(prop: dict) -> str:
    assert len(prop["alcance_mvp"]["in_scope"]) <= 3, "El MVP debe tener máximo 3 funcionalidades IN-SCOPE."
    assert len(prop["alcance_mvp"]["out_of_scope"]) >= 2, "Define al menos 2 ítems explícitos OUT-OF-SCOPE."
    assert len(prop["roles_equipo_mixto"]) >= 2, "Asigna al menos 2 roles distintos en el equipo mixto."
    assert len(prop["riesgos"]) >= 2, "Documenta al menos 2 riesgos con su mitigación."

    md = [
        f"# {prop['titulo_proyecto']} — {prop['equipo']}",
        "## 1. Problema y Propuesta de Valor",
        f"- **Usuario objetivo:** {prop['problema']['usuario_objetivo']}",
        f"- **Dolor actual:** {prop['problema']['dolor_actual']}",
        f"- **Objetivo medible:** {prop['problema']['objetivo_medible']}",
        "\n## 2. Alcance MVP (Regla 1-1-1)",
        f"- **1 Usuario:** {prop['alcance_mvp']['regla_1_1_1']['usuario']}",
        f"- **1 Flujo:** {prop['alcance_mvp']['regla_1_1_1']['flujo']}",
        f"- **1 Métrica:** {prop['alcance_mvp']['regla_1_1_1']['metrica']}",
        "- **IN-SCOPE:**\n  - " + "\n  - ".join(prop["alcance_mvp"]["in_scope"]),
        "- **OUT-OF-SCOPE (v2):**\n  - " + "\n  - ".join(prop["alcance_mvp"]["out_of_scope"]),
        "\n## 3. Arquitectura y Contrato JSON",
        f"- **Datos:** {prop['arquitectura']['capa_datos']}",
        f"- **IA / Serving:** {prop['arquitectura']['capa_ia_serving']}",
        f"- **Aplicación:** {prop['arquitectura']['capa_aplicacion']}",
        f"```json\n{json.dumps(prop['arquitectura']['contrato_json'], indent=2, ensure_ascii=False)}\n```",
        "\n## 4. Roles del Equipo Mixto",
        "| Rol | Responsable | Entregable Clase 8 | Entregable Clase 9 |",
        "|---|---|---|---|",
    ]
    for r in prop["roles_equipo_mixto"]:
        md.append(f"| {r['rol']} | {r['responsable']} | {r['clase_8']} | {r['clase_9']} |")

    md.append("\n## 5. Riesgos y Mitigaciones")
    for rk in prop["riesgos"]:
        md.append(f"- **Riesgo:** {rk['riesgo']} → **Mitigación:** {rk['mitigacion']}")

    return "\n".join(md)


markdown_generado = validar_y_renderizar_propuesta(propuesta_ejemplo)
print("Validación exitosa: la propuesta cumple todas las reglas de MVP.\n")
print(markdown_generado)

# COMMAND ----------

# Salida esperada:
# - Impresión de la tabla comparativa PoC vs Prototipo Funcional / MVP vs Producto.
# - Validación exitosa sin errores de aserción (`in_scope <= 3`, `out_of_scope >= 2`, roles y riesgos).
# - Documento completo en formato Markdown listo para copiar al repositorio del equipo.
