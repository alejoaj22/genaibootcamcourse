# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (Solución) — Deployment y Batch Inference de 5 tickets con `ai_query()`
# MAGIC **Objetivo:** solución de referencia completa para configurar `EndpointCoreConfigInput`, verificar o desplegar `dphone_endpoint`, probarlo por SDK y ejecutar batch inference sobre 5 tickets abiertos con `ai_query()`.
# MAGIC **Tiempo estimado:** 15 minutos (si el endpoint ya está `READY`, ~2 minutos).
# MAGIC **Concepto que ilustra:** Model Serving productivo + SQL `ai_query()` en Unity Catalog (Notebook 4.1).
# MAGIC **Requisitos previos:** `dphone_agent` registrado en Unity Catalog (`genai_course`).

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
if not versiones:
    raise RuntimeError(f"No hay versiones registradas para {model_fullname}.")

latest_model_version = max(int(v.version) for v in versiones)
serving_endpoint_name = "dphone_endpoint"

print(f"Modelo: {model_fullname} v{latest_model_version} | Endpoint: {serving_endpoint_name}")

# COMMAND ----------

# 1) Configuración oficial del Serving Endpoint.
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_fullname,
            "model_version": latest_model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
            "environment_vars": {},
        }
    ]
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
w = WorkspaceClient()

# COMMAND ----------

# 2) Verificación idempotente o creación del endpoint.
try:
    ep = w.serving_endpoints.get(name=serving_endpoint_name)
    endpoint_estado = ep.state
    print(f"Endpoint '{serving_endpoint_name}' reutilizado. Estado: {endpoint_estado}")
except Exception:
    print(f"Creando '{serving_endpoint_name}' (~8-15 min)...")
    ep = w.serving_endpoints.create_and_wait(
        name=serving_endpoint_name,
        config=endpoint_config,
    )
    endpoint_estado = ep.state
    print("Endpoint creado:", endpoint_estado)

# Esperar en caso de que el endpoint compartido siguiera en actualización
while "READY" not in str(w.serving_endpoints.get(name=serving_endpoint_name).state.ready):
    print("Esperando estado READY...")
    time.sleep(20)

# COMMAND ----------

# 3) Invocación de prueba por SDK (calienta el endpoint antes del lote SQL y mide latencia).
t0 = time.time()
respuesta_sdk = w.api_client.do(
    "POST",
    f"/serving-endpoints/{serving_endpoint_name}/invocations",
    body={"input": [{"role": "user", "content": "What warranty covers the dPhone D1?"}]},
)
latencia_s = time.time() - t0

print(f"Latencia SDK: {latencia_s:.2f} s")
print("Respuesta resumida:", str(respuesta_sdk)[:400])

# COMMAND ----------

# 4) Batch inference sobre exactamente 5 tickets abiertos usando ai_query() y persistencia en Delta.
tabla_tickets = f"{course.catalog}.{course.schema}.customer_support_tickets"
tabla_salida = f"{course.catalog}.{course.schema}.mi_lote_tickets_eval"

try:
    df_batch = spark.sql(f"""
        SELECT
            description,
            ai_query(endpoint => '{serving_endpoint_name}', request => description) AS agent_response,
            current_timestamp() AS processed_at
        FROM {tabla_tickets}
        WHERE status = 'open'
        LIMIT 5
    """)
    df_batch.write.mode("overwrite").saveAsTable(tabla_salida)
except Exception as err:
    print("Usando variante estructurada ResponsesAgent por esquema del endpoint:", err)
    df_batch = spark.sql(f"""
        SELECT
            description,
            ai_query(
                endpoint => '{serving_endpoint_name}',
                request => named_struct(
                    'input',
                    array(named_struct('role', 'user', 'content', description))
                )
            ) AS agent_response,
            current_timestamp() AS processed_at
        FROM {tabla_tickets}
        WHERE status = 'open'
        LIMIT 5
    """)
    df_batch.write.mode("overwrite").saveAsTable(tabla_salida)

df_verificacion = spark.read.table(tabla_salida)
assert df_verificacion.count() == 5, "Se esperaban exactamente 5 filas procesadas."
print(f"Batch completado: {df_verificacion.count()} filas guardadas en {tabla_salida}")
display(df_verificacion)

# COMMAND ----------

# 5) Análisis crítico QA.
observacion_qa = (
    "Los tickets que involucran reembolsos, garantías fuera de plazo, reclamos de facturación o quejas de tono emocional "
    "requieren revisión humana (Human-in-the-Loop) antes de enviarse al cliente, usando el borrador de ai_query() como punto de partida."
)
print("Observación QA:", observacion_qa)

# COMMAND ----------

# Salida esperada:
# - Endpoint `dphone_endpoint` verificado o desplegado en estado `READY`.
# - Invocación SDK completada con latencia medida en segundos.
# - Tabla `<catálogo>.genai_course.mi_lote_tickets_eval` creada con exactamente 5 filas y columna `agent_response` generada por `ai_query()`.
