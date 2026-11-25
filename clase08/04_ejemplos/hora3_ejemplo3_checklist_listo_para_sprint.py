# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 3 — Auditoría técnica automática: Checklist "Listo para el Sprint (Clase 9)"
# MAGIC **Objetivo:** ejecutar una verificación automatizada contra Unity Catalog, MLflow y Model Serving para comprobar qué requisitos de la Clase 8 ya están verdes y qué brechas debe cerrar el equipo antes de la Clase 9.
# MAGIC **Tiempo estimado:** 5 minutos.
# MAGIC **Concepto que ilustra:** control de calidad (*Definition of Ready*) para pasar de evaluación y deployment al sprint de construcción y monitoreo (RA4 y RA5).
# MAGIC **Requisitos previos:** haber ejecutado `Lab_Setup` y los ejemplos previos de la Clase 8.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()
w = WorkspaceClient()

model_name = f"{course.catalog}.{course.schema}.dphone_agent"
tabla_tickets = f"{course.catalog}.{course.schema}.customer_support_tickets"
tabla_batch = f"{course.catalog}.{course.schema}.tickets_respuestas_batch"
endpoint_name = "dphone_endpoint"

checks = []

# Check 1: Modelo registrado en UC
try:
    versiones = mlflow_client.search_model_versions(f"name='{model_name}'")
    ok = len(versiones) >= 1
    detalle = f"{len(versiones)} versión(es) encontrada(s)"
except Exception as e:
    ok, detalle = False, str(e)
checks.append({"requisito": "1. Agente registrado en Unity Catalog", "estado": "OK" if ok else "PENDIENTE", "detalle": detalle})

# Check 2: Alias @champion o @challenger configurado
try:
    aliases = mlflow_client.get_registered_model(model_name).aliases
    ok = len(aliases) >= 1
    detalle = f"Alias activos: {aliases}"
except Exception as e:
    ok, detalle = False, str(e)
checks.append({"requisito": "2. Alias de modelo (@champion/@challenger)", "estado": "OK" if ok else "PENDIENTE", "detalle": detalle})

# Check 3: Tabla fuente batch customer_support_tickets
try:
    cnt = spark.read.table(tabla_tickets).count()
    ok = cnt > 0
    detalle = f"{cnt} filas en {tabla_tickets}"
except Exception as e:
    ok, detalle = False, "Ejecuta Lab_Setup_04 para crear customer_support_tickets"
checks.append({"requisito": "3. Tabla batch customer_support_tickets", "estado": "OK" if ok else "PENDIENTE", "detalle": detalle})

# Check 4: Serving Endpoint creado y READY
try:
    ep = w.serving_endpoints.get(name=endpoint_name)
    ready = "READY" in str(ep.state.ready)
    ok = ready
    detalle = f"Estado={ep.state.ready}"
except Exception:
    ok, detalle = False, f"Endpoint '{endpoint_name}' no encontrado (o en otro nombre de equipo)"
checks.append({"requisito": "4. Model Serving Endpoint READY", "estado": "OK" if ok else "PENDIENTE", "detalle": detalle})

# Check 5: Evidencia de salida batch persistida
try:
    cnt_b = spark.read.table(tabla_batch).count()
    ok = cnt_b > 0
    detalle = f"{cnt_b} respuestas guardadas en {tabla_batch}"
except Exception:
    ok, detalle = False, "Ejecuta hora2_ejemplo2 para generar tickets_respuestas_batch"
checks.append({"requisito": "5. Tabla de salidas batch ai_query()", "estado": "OK" if ok else "PENDIENTE", "detalle": detalle})

df_checks = pd.DataFrame(checks)
display(df_checks)

total_ok = sum(1 for c in checks if c["estado"] == "OK")
print(f"\nResultado Definition of Ready: {total_ok}/{len(checks)} requisitos listos para el Sprint de Clase 9.")

# COMMAND ----------

# Salida esperada:
# - Tabla semáforo con los 5 requisitos técnicos verificados en tiempo real sobre Unity Catalog y Model Serving.
# - Resumen numérico `X/5 requisitos listos para el Sprint de Clase 9` indicando exactamente qué falta completar si algún punto quedó en `PENDIENTE`.
