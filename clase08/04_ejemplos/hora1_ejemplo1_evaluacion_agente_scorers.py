# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 1 — Evaluación del agente con `mlflow.genai.evaluate()` (sin y con Ground Truth)
# MAGIC **Objetivo:** cargar el agente `dphone_agent` desde Unity Catalog, envolverlo en `predict_wrapper(query)` y evaluarlo primero sin ground truth (`RelevanceToQuery`, `RetrievalRelevance`, `RetrievalGroundedness`, `Safety`) y luego con ground truth (`Correctness`, `RetrievalSufficiency`).
# MAGIC **Tiempo estimado:** 10 minutos (la ejecución de ambos conjuntos tarda ~2–4 minutos).
# MAGIC **Concepto que ilustra:** el triángulo Query–Answer–Context (`llm-judges.png`) y su extensión con Ground Truth (`llm-judges-with-ground-truth.png`) del Notebook 3.1.
# MAGIC **Requisitos previos:** modelo `<catálogo>.genai_course.dphone_agent` (versión 1) registrado en Unity Catalog en la Clase 7; índice `dphone_vs_index` ONLINE y acceso a Foundation Models.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# 1) Configuramos el registro de Unity Catalog y cargamos la versión 1 del agente dPhone.
import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{course.catalog}.{course.schema}.dphone_agent"
model_version = 1
logged_model_uri = f"models:/{model_name}/{model_version}"

print(f"Cargando modelo desde Unity Catalog: {logged_model_uri}")
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# COMMAND ----------

# 2) Definimos el wrapper de predicción.
# IMPORTANTE: el nombre del parámetro ('query') debe coincidir exactamente con la clave en 'inputs' del eval_dataset.
def predict_wrapper(query: str) -> str:
    model_input = {
        "input": [
            {"role": "user", "content": query}
        ]
    }
    response = loaded_model.predict(model_input)

    # Extracción defensiva compatible con respuestas simples y con múltiples bloques de ResponsesAgent
    for item in reversed(response.get("output", [])):
        if item.get("type") == "message" and "content" in item:
            for content_item in item["content"]:
                if content_item.get("type") == "output_text":
                    return content_item["text"]

    # Respaldo directo equivalente al Notebook 3.1 si el esquema es simple
    return response["output"][-1]["content"][0]["text"]


# Prueba unitaria rápida antes de lanzar toda la evaluación
respuesta_prueba = predict_wrapper("What is the storage capacity of dPhone D1 Pro?")
print("Prueba rápida del agente:\n", respuesta_prueba)

# COMMAND ----------

# 3) Dataset SIN Ground Truth (corresponde al diagrama Includes/Images/llm-judges.png).
# Útil cuando no hay respuestas de referencia escritas por humanos (ej. muestreo de tráfico en producción).
eval_dataset_sin_gt = [
    {
        "inputs": {"query": "How to replace the battery of dPhone D1?"},
        "expectations": None,
    },
    {
        "inputs": {"query": "Explain how the dPhone D1 Pro is waterproof?"},
        "expectations": None,
    },
    {
        "inputs": {"query": "What warranty options are available for dPhone devices?"},
        "expectations": None,
    },
]
print("Consultas sin Ground Truth:", len(eval_dataset_sin_gt))

# COMMAND ----------

# 4) Evaluación con los 3 scorers del triángulo + Safety().
from mlflow.genai.scorers import (
    RelevanceToQuery,
    RetrievalRelevance,
    RetrievalGroundedness,
    Safety,
)

eval_results_sin_gt = mlflow.genai.evaluate(
    predict_fn=predict_wrapper,
    data=eval_dataset_sin_gt,
    scorers=[
        RelevanceToQuery(),
        RetrievalRelevance(),
        RetrievalGroundedness(),
        Safety(),
    ],
)

print("\nMétricas agregadas (SIN Ground Truth):")
for k, v in eval_results_sin_gt.metrics.items():
    print(f"  - {k}: {v}")

# COMMAND ----------

# 5) Dataset CON Ground Truth (corresponde a Includes/Images/llm-judges-with-ground-truth.png).
# Se combinan 'expected_facts' (lista de hechos) y 'expected_response' (respuesta esperada).
eval_dataset_with_facts = [
    {
        "inputs": {"query": "What models are available for the dPhone D1?"},
        "expectations": {"expected_facts": ["D1 Lite", "D1 Pro", "D1 Fold", "D1"]},
    },
    {
        "inputs": {"query": "What kind of display does the dPhone D1 Fold have?"},
        "expectations": {"expected_response": '10.1" Super-AMOLED "Infinity" Display'},
    },
    {
        "inputs": {"query": "Which dPhone D1 model is aimed at budget-conscious users?"},
        "expectations": {"expected_facts": ["D1 Lite"]},
    },
]
print("Consultas con Ground Truth:", len(eval_dataset_with_facts))

# COMMAND ----------

# 6) Evaluación extendida agregando Correctness() y RetrievalSufficiency().
from mlflow.genai.scorers import Correctness, RetrievalSufficiency

eval_results_con_gt = mlflow.genai.evaluate(
    data=eval_dataset_with_facts,
    predict_fn=predict_wrapper,
    scorers=[
        RelevanceToQuery(),
        RetrievalRelevance(),
        RetrievalGroundedness(),
        Safety(),
        Correctness(),
        RetrievalSufficiency(),
    ],
)

print("\nMétricas agregadas (CON Ground Truth):")
for k, v in eval_results_con_gt.metrics.items():
    print(f"  - {k}: {v}")

# COMMAND ----------

# 7) Inspección tabular fila por fila para diagnosticar qué pregunta falló y por qué.
display(eval_results_con_gt.result_df)

# COMMAND ----------

# Salida esperada:
# - Impresión de la URI `models:/<catálogo>.genai_course.dphone_agent/1` y una respuesta coherente sobre la capacidad del dPhone D1 Pro.
# - Primera corrida `eval_results_sin_gt` con métricas agregadas para `relevance_to_query`, `retrieval_relevance`, `retrieval_groundedness` y `safety`.
# - Segunda corrida `eval_results_con_gt` agregando `correctness` y `retrieval_sufficiency`.
# - Visualización de `eval_results_con_gt.result_df` con la respuesta generada, puntajes binarios/numéricos y justificación (rationale) de cada juez por fila.
