# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 3 (Starter) — Propuesta Formal del Prototipo Funcional
# MAGIC **Objetivo:** completar los `# TODO` en equipo mixto para generar y validar automáticamente la propuesta oficial del prototipo hacia el Sprint de la Clase 9.
# MAGIC **Tiempo estimado:** 15 minutos.
# MAGIC **Concepto que ilustra:** definición de alcance MVP (Regla 1-1-1), arquitectura de integración y distribución de roles en equipos mixtos (RA5).
# MAGIC **Requisitos previos:** trabajo colaborativo entre perfiles de datos, desarrollo y negocio.

# COMMAND ----------

import json

# TODO 1..5: Completa TODOS los campos marcados con "TODO" con la información real de tu equipo.
propuesta_equipo = {
    "equipo": "TODO: Nombre o número del equipo (ej. Equipo 03)",
    "titulo_proyecto": "TODO: Nombre corto del prototipo funcional",
    "problema": {
        "usuario_objetivo": "TODO: ¿Quién usará el prototipo en su trabajo diario?",
        "dolor_actual": "TODO: ¿Qué problema concreto tiene hoy (tiempo, errores, costo)?",
        "objetivo_medible": "TODO: Resultado esperado medible (ej. reducir tiempo de respuesta de 8 min a < 1 min).",
    },
    "alcance_mvp": {
        "regla_1_1_1": {
            "usuario": "TODO: 1 usuario principal",
            "flujo": "TODO: 1 flujo completo de extremo a extremo",
            "metrica": "TODO: 1 criterio cuantitativo en MLflow (ej. Correctness >= 0.80, Safety == 1.0)",
        },
        # Máximo 3 funcionalidades IN-SCOPE para Clases 8-9
        "in_scope": [
            "TODO 1: Capa de datos / RAG / UC Functions",
            "TODO 2: Evaluación MLflow + Serving Endpoint o ai_query()",
            "TODO 3: Interfaz mínima usable (Clase 9)",
        ],
        # Mínimo 3 elementos explícitos OUT-OF-SCOPE (v2)
        "out_of_scope": [
            "TODO out 1",
            "TODO out 2",
            "TODO out 3",
        ],
    },
    "arquitectura": {
        "capa_datos": "TODO: Catálogo, esquema, tablas Delta y/o Vector Search Index",
        "capa_ia_serving": "TODO: Modelo UC, alias (@champion), Serving Endpoint (workload_size, scale_to_zero)",
        "capa_aplicacion": "TODO: Cómo consumirá el endpoint la app o proceso batch en Clase 9",
        "contrato_json": {
            "request": {"input": [{"role": "user", "content": "<TODO: ejemplo_consulta>"}]},
            "response": {"status": "ok", "answer_text": "<TODO: respuesta>", "latency_ms": 0},
        },
    },
    "roles_equipo_mixto": [
        {
            "rol": "Data & AI Engineer",
            "responsable": "TODO: Nombre",
            "clase_8": "TODO: Entregable hoy",
            "clase_9": "TODO: Entregable Sprint",
        },
        {
            "rol": "App / Backend Developer",
            "responsable": "TODO: Nombre",
            "clase_8": "TODO: Entregable hoy",
            "clase_9": "TODO: Entregable Sprint",
        },
        {
            "rol": "Product / Negocio & QA",
            "responsable": "TODO: Nombre",
            "clase_8": "TODO: Entregable hoy",
            "clase_9": "TODO: Entregable Sprint",
        },
    ],
    "riesgos": [
        {"riesgo": "TODO: Riesgo técnico o de negocio 1", "mitigacion": "TODO: Cómo lo mitigan"},
        {"riesgo": "TODO: Riesgo técnico o de negocio 2", "mitigacion": "TODO: Cómo lo mitigan"},
    ],
}

# COMMAND ----------

# Validador automático: verifica que no queden campos "TODO" sin diligenciar y que el MVP esté acotado.
def validar_propuesta(prop: dict) -> None:
    texto_completo = json.dumps(prop, ensure_ascii=False)
    assert "TODO" not in texto_completo, "Aún quedan campos con 'TODO'. Complétalos todos en equipo."
    assert 1 <= len(prop["alcance_mvp"]["in_scope"]) <= 3, "El MVP debe tener entre 1 y 3 funcionalidades IN-SCOPE."
    assert len(prop["alcance_mvp"]["out_of_scope"]) >= 3, "Incluye al menos 3 elementos explícitos OUT-OF-SCOPE."
    assert len(prop["roles_equipo_mixto"]) >= 2, "Debe haber al menos 2 roles asignados."
    assert len(prop["riesgos"]) >= 2, "Debe haber al menos 2 riesgos con su mitigación."

validar_propuesta(propuesta_equipo)

print(f"# Propuesta Formal: {propuesta_equipo['titulo_proyecto']} ({propuesta_equipo['equipo']})")
print("## Problema:", propuesta_equipo["problema"]["objetivo_medible"])
print("## Regla 1-1-1:", propuesta_equipo["alcance_mvp"]["regla_1_1_1"])
print("## Contrato JSON:\n", json.dumps(propuesta_equipo["arquitectura"]["contrato_json"], indent=2, ensure_ascii=False))
print("\nPropuesta validada con éxito. ¡Listos para el Sprint de la Clase 9!")

# COMMAND ----------

# Salida esperada:
# - Validación exitosa sin `AssertionError` (ningún campo contiene `TODO`).
# - Resumen estructurado del proyecto, Regla 1-1-1 y contrato JSON impreso correctamente.
