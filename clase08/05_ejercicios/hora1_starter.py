# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 1 (Starter) — Eval Dataset de 10+ queries y 3 `Guidelines` propias
# MAGIC **Objetivo:** completar los `# TODO` para evaluar `<catálogo>.genai_course.dphone_agent` con mínimo 10 consultas y 3 reglas de negocio propias (`Guidelines`).
# MAGIC **Tiempo estimado:** 15 minutos.
# MAGIC **Concepto que ilustra:** diseño de *Golden Sets*, uso de scorers con/sin ground truth y codificación de políticas de negocio en MLflow.
# MAGIC **Requisitos previos:** `dphone_agent` v1 registrado en Unity Catalog (`genai_course`).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import (
    RelevanceToQuery,
    RetrievalRelevance,
    RetrievalGroundedness,
    Safety,
    Correctness,
    RetrievalSufficiency,
    Guidelines,
)

mlflow.set_registry_uri("databricks-uc")
model_name = f"{course.catalog}.{course.schema}.dphone_agent"
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

def predict_wrapper(query: str) -> str:
    response = loaded_model.predict({"input": [{"role": "user", "content": query}]})
    for item in reversed(response.get("output", [])):
        if item.get("type") == "message" and "content" in item:
            for c in item["content"]:
                if c.get("type") == "output_text":
                    return c["text"]
    return response["output"][-1]["content"][0]["text"]

# COMMAND ----------

# TODO 1: Completa eval_dataset hasta tener AL MENOS 10 consultas variadas
# (incluye consultas con expected_facts, con expected_response, con expectations=None y 2+ casos adversos).
eval_dataset = [
    {
        "inputs": {"query": "What models are available for the dPhone D1?"},
        "expectations": {"expected_facts": ["D1 Lite", "D1 Pro", "D1 Fold", "D1"]},
    },
    {
        "inputs": {"query": "What kind of display does the dPhone D1 Fold have?"},
        "expectations": {"expected_response": '10.1" Super-AMOLED "Infinity" Display'},
    },
    {
        "inputs": {"query": "How to replace the battery of dPhone D1?"},
        "expectations": None,
    },
    # TODO: Agrega aquí al menos 7 consultas adicionales hasta completar 10+ filas
]

assert len(eval_dataset) >= 10, f"Debes tener al menos 10 consultas; actualmente tienes {len(eval_dataset)}."
print(f"Dataset listo con {len(eval_dataset)} consultas.")

# COMMAND ----------

# TODO 2: Define 3 Guidelines propias adaptadas a políticas de negocio de dPhone.
# Usa model="databricks:/databricks-gpt-oss-120b" al menos en las reglas complejas.

guideline_1 = Guidelines(
    name="TODO_nombre_regla_1",
    guidelines=["TODO: escribe una regla verificable sobre claridad, idioma o estructura."],
    model="databricks:/databricks-gpt-oss-120b",
)

guideline_2 = Guidelines(
    name="TODO_nombre_regla_2",
    guidelines=["TODO: escribe una regla comercial (ej. no inventar precios ni ofrecer descuentos)."],
    model="databricks:/databricks-gpt-oss-120b",
)

guideline_3 = Guidelines(
    name="TODO_nombre_regla_3",
    guidelines=["TODO: escribe una regla de alcance o privacidad (ej. rechazar temas fuera de dPhone / no pedir PII)."],
    model="databricks:/databricks-gpt-oss-120b",
)

# COMMAND ----------

# TODO 3: Ejecuta mlflow.genai.evaluate con los 6 scorers predefinidos + tus 3 Guidelines.
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_wrapper,
    scorers=[
        # TODO: añade RelevanceToQuery(), RetrievalRelevance(), RetrievalGroundedness(), Safety(),
        # Correctness(), RetrievalSufficiency(), guideline_1, guideline_2, guideline_3
    ],
)

print("Métricas agregadas:")
for k, v in eval_results.metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# TODO 4: Inspecciona eval_results.result_df y documenta en el print final el diagnóstico de 1 fallo o puntaje bajo.
display(eval_results.result_df)

diagnostico = "TODO: escribe aquí qué consulta tuvo menor puntaje y si la causa fue Retrieval o Generación."
print("Diagnóstico del equipo:", diagnostico)

# COMMAND ----------

# Salida esperada:
# - Dataset validado con >= 10 consultas (sin lanzar AssertionError).
# - Métricas agregadas de los 6 scorers predefinidos + 3 Guidelines propias.
# - Tabla `eval_results.result_df` visible con las justificaciones del juez por fila y diagnóstico escrito del equipo.
