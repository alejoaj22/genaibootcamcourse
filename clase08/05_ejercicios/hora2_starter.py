# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (Starter) — Deployment y Batch Inference de 5 tickets con `ai_query()`
# MAGIC **Objetivo:** completar los `# TODO` para configurar `EndpointCoreConfigInput`, validar o desplegar el Serving Endpoint `dphone_endpoint` y procesar 5 tickets abiertos con `ai_query()`.
# MAGIC **Tiempo estimado:** 15 minutos.
# MAGIC **Concepto que ilustra:** despliegue en Databricks Model Serving y consumo SQL masivo mediante `ai_query()` (Notebook 4.1).
# MAGIC **Requisitos previos:** `dphone_agent` registrado en Unity Catalog; permisos de Serving Endpoint (o acceso al endpoint compartido `dphone_endpoint`).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_04

# COMMAND ----------

import time
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

model_fullname = f"{course.catalog}.{course.schema}.dphone_agent"
versiones = mlflow_client.search_model_versions(f"name='{model_fullname}'")
latest_model_version = max(int(v.version) for v in versiones)
serving_endpoint_name = "dphone_endpoint"

print(f"Modelo encontrado: {model_fullname} (última versión: {latest_model_version})")

# COMMAND ----------

# TODO 1: Completa endpoint_config_dict con model_name, model_version, scale_to_zero_enabled y workload_size.
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": None,           # TODO: asigna model_fullname
            "model_version": None,        # TODO: asigna latest_model_version
            "scale_to_zero_enabled": None, # TODO: True para ahorro de costos en laboratorio
            "workload_size": None,        # TODO: "Small"
            "environment_vars": {},
        }
    ]
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
w = WorkspaceClient()

# COMMAND ----------

# TODO 2: Verifica si el endpoint 'dphone_endpoint' ya existe. Si no existe, créalo con create_and_wait().
# Tip: revisa hora2_ejemplo1_deployment_serving_endpoint.py para manejar ambos casos sin error.
endpoint_estado = None

# TODO: escribe aquí el bloque try/except para obtener el endpoint existente o crearlo

print("Estado del endpoint:", endpoint_estado)

# COMMAND ----------

# TODO 3: Envía una consulta de prueba al endpoint vía SDK (w.api_client.do) y mide su latencia en segundos.
t0 = time.time()
respuesta_sdk = None  # TODO: llama POST /serving-endpoints/{serving_endpoint_name}/invocations
latencia_s = time.time() - t0

print(f"Latencia SDK: {latencia_s:.2f} s")
print("Respuesta:", str(respuesta_sdk)[:400])

# COMMAND ----------

# TODO 4: Ejecuta ai_query() en SQL sobre 5 tickets abiertos de customer_support_tickets
# y guarda el resultado en la tabla <catálogo>.genai_course.mi_lote_tickets_eval.
tabla_tickets = f"{course.catalog}.{course.schema}.customer_support_tickets"
tabla_salida = f"{course.catalog}.{course.schema}.mi_lote_tickets_eval"

df_batch = spark.sql(f"""
    SELECT
        description,
        -- TODO: reemplaza NULL por la llamada a ai_query(endpoint => ..., request => ...)
        NULL AS agent_response
    FROM {tabla_tickets}
    WHERE status = 'open'
    LIMIT 5
""")

df_batch.write.mode("overwrite").saveAsTable(tabla_salida)
print(f"Filas guardadas en {tabla_salida}: {spark.read.table(tabla_salida).count()}")
display(spark.read.table(tabla_salida))

# COMMAND ----------

# TODO 5: Revisa las 5 filas generadas y documenta qué ticket requeriría revisión humana obligatoria.
observacion_qa = "TODO: indica cuál de los 5 tickets necesita revisión humana antes de responder al cliente y por qué."
print("Observación QA:", observacion_qa)

# COMMAND ----------

# Salida esperada:
# - `EndpointCoreConfigInput` construido sin errores (`scale_to_zero_enabled=True`, `workload_size="Small"`).
# - Endpoint `dphone_endpoint` en estado `READY` y prueba SDK exitosa con latencia impresa.
# - Tabla `<catálogo>.genai_course.mi_lote_tickets_eval` con exactamente 5 filas y respuestas generadas por `ai_query()`.
