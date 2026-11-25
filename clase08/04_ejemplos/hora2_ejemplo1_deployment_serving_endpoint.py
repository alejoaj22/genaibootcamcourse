# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 1 — Deployment productivo del agente en Databricks Model Serving
# MAGIC **Objetivo:** configurar `EndpointCoreConfigInput` (`scale_to_zero_enabled=True`, `workload_size="Small"`) y desplegar o verificar el Serving Endpoint `dphone_endpoint` con `WorkspaceClient().serving_endpoints.create_and_wait()`, realizando luego una invocación de prueba.
# MAGIC **Tiempo estimado:** 10–15 minutos si el endpoint se crea desde cero; 1 minuto si ya existe `READY`.
# MAGIC **Concepto que ilustra:** exposición gobernada de un agente registrado en Unity Catalog como API REST / Model Serving (Notebook 4.1).
# MAGIC **Requisitos previos:** modelo `<catálogo>.genai_course.dphone_agent` registrado en Unity Catalog y permisos para crear/consultar Serving Endpoints.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_04

# COMMAND ----------

# 1) Verificamos qué versiones existen en Unity Catalog para elegir la más reciente (ej. v2 o v1).
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

model_fullname = f"{course.catalog}.{course.schema}.dphone_agent"
versiones = mlflow_client.search_model_versions(f"name='{model_fullname}'")
if not versiones:
    raise RuntimeError(f"No se encontró el modelo {model_fullname} en Unity Catalog.")

latest_model_version = max(int(v.version) for v in versiones)
serving_endpoint_name = "dphone_endpoint"

print(f"Modelo a desplegar: {model_fullname} (versión {latest_model_version})")
print(f"Endpoint destino:   {serving_endpoint_name}")

# COMMAND ----------

# 2) Definimos la configuración oficial del Serving Endpoint (Notebook 4.1).
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_fullname,
            "model_version": latest_model_version,
            "scale_to_zero_enabled": True,   # Ahorra DBUs cuando no hay tráfico (introduce cold start)
            "workload_size": "Small",        # Adecuado para prototipo / baja concurrencia
            "environment_vars": {},
        }
    ]
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
w = WorkspaceClient()

# COMMAND ----------

# 3) Despliegue idempotente: si el endpoint ya existe, verificamos su estado en lugar de fallar.
# NOTA: La creación inicial toma ~8 a 15 minutos mientras compila el contenedor del agente.
endpoint_existente = None
try:
    endpoint_existente = w.serving_endpoints.get(name=serving_endpoint_name)
    print(f"El endpoint '{serving_endpoint_name}' ya existe. Estado actual: {endpoint_existente.state}")
except Exception:
    print(f"Creando endpoint '{serving_endpoint_name}' y esperando a que quede READY (~8-15 min)...")
    endpoint_creado = w.serving_endpoints.create_and_wait(
        name=serving_endpoint_name,
        config=endpoint_config,
    )
    print("Endpoint creado con éxito:", endpoint_creado.state)

# COMMAND ----------

# 4) Si ya existía pero seguía en construcción, esperamos de forma segura hasta que esté READY.
import time

estado = w.serving_endpoints.get(name=serving_endpoint_name).state
if getattr(estado, "ready", None) and str(estado.ready) != "EndpointStateReady.READY":
    print("Esperando a que el endpoint pase a READY...")
    for _ in range(30):
        estado = w.serving_endpoints.get(name=serving_endpoint_name).state
        print("  Estado:", estado.ready, "| Config update:", estado.config_update)
        if "READY" in str(estado.ready) and "IN_PROGRESS" not in str(estado.config_update):
            break
        time.sleep(20)
else:
    print("Endpoint READY para recibir tráfico:", estado)

# COMMAND ----------

# 5) Prueba funcional directa contra el Serving Endpoint vía Databricks SDK.
# Advertencia: si scale_to_zero_enabled=True y estuvo inactivo, la primera llamada tendrá cold start (~30-90s).
import time

t0 = time.time()
respuesta = w.api_client.do(
    "POST",
    f"/serving-endpoints/{serving_endpoint_name}/invocations",
    body={
        "input": [
            {"role": "user", "content": "What models are available for the dPhone D1?"}
        ]
    },
)
latencia = time.time() - t0

print(f"Latencia de respuesta: {latencia:.2f} s")
print("Respuesta JSON cruda (resumen):", str(respuesta)[:500])

# COMMAND ----------

# Salida esperada:
# - Identificación automática de la última versión registrada de `<catálogo>.genai_course.dphone_agent`.
# - Creación o verificación idempotente del Serving Endpoint `dphone_endpoint` con `scale_to_zero_enabled=True` y `workload_size="Small"`.
# - Estado final `READY` y respuesta JSON exitosa ante la consulta de prueba, mostrando la latencia medida.
