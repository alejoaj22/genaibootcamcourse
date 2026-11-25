# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 2 — Batch Inference en SQL con `ai_query()` sobre tickets de soporte
# MAGIC **Objetivo:** inicializar la tabla `customer_support_tickets` con `Lab_Setup_04`, inspeccionar tickets con `status = 'open'` y ejecutar inferencia por lotes directamente en SQL mediante `ai_query()`, persistiendo los resultados en una tabla Delta.
# MAGIC **Tiempo estimado:** 10 minutos.
# MAGIC **Concepto que ilustra:** consumo masivo y gobernado de un Serving Endpoint desde SQL sin salir de Unity Catalog (Notebook 4.1).
# MAGIC **Requisitos previos:** `dphone_endpoint` en estado `READY` (Ejemplo 1) y un SQL Warehouse o cómputo compatible con `ai_query()`.

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_04

# COMMAND ----------

# 1) Inspeccionamos la tabla customer_support_tickets creada por Lab_Setup_04.
customer_requests_table = f"{course.catalog}.{course.schema}.customer_support_tickets"
df_tickets = spark.read.table(customer_requests_table)

print(f"Tabla cargada: {customer_requests_table} | Total filas: {df_tickets.count()}")
display(df_tickets.groupBy("status").count())

# COMMAND ----------

# 2) Mostramos una muestra de los tickets abiertos antes de invocar el endpoint (buena práctica FinOps).
customer_requests_df = df_tickets.filter("status = 'open'").limit(3)
display(customer_requests_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3) Batch Inference oficial del Notebook 4.1 (limitado a 3 filas en demostración para controlar costo/tiempo).
# MAGIC -- Nota: si el endpoint tiene scale_to_zero=True y estaba dormido, la primera invocación tomará ~30-90s en despertar.
# MAGIC SELECT
# MAGIC   description,
# MAGIC   ai_query(endpoint => "dphone_endpoint", request => description) AS response
# MAGIC FROM customer_support_tickets
# MAGIC WHERE status = 'open'
# MAGIC LIMIT 3

# COMMAND ----------

# 4) Patrón productivo: si el endpoint exige estrictamente el esquema ResponsesAgent (input=[{role, content}])
# o queremos persistir el lote en una tabla Delta idempotente, lo ejecutamos y guardamos con trazabilidad.
tabla_salida_batch = f"{course.catalog}.{course.schema}.tickets_respuestas_batch"

try:
    df_batch = spark.sql(f"""
        SELECT
            description,
            ai_query(endpoint => 'dphone_endpoint', request => description) AS agent_response,
            current_timestamp() AS processed_at
        FROM {customer_requests_table}
        WHERE status = 'open'
        LIMIT 3
    """)
    df_batch.write.mode("overwrite").saveAsTable(tabla_salida_batch)
    print(f"Resultados batch guardados correctamente en: {tabla_salida_batch}")
    display(spark.read.table(tabla_salida_batch))
except Exception as err:
    print("Aviso: el endpoint requirió formato estructurado ResponsesAgent o no estuvo listo:", err)
    print("Ejecutando variante estructurada con named_struct('input', array(...))...")
    df_batch = spark.sql(f"""
        SELECT
            description,
            ai_query(
                endpoint => 'dphone_endpoint',
                request => named_struct(
                    'input',
                    array(named_struct('role', 'user', 'content', description))
                )
            ) AS agent_response,
            current_timestamp() AS processed_at
        FROM {customer_requests_table}
        WHERE status = 'open'
        LIMIT 3
    """)
    df_batch.write.mode("overwrite").saveAsTable(tabla_salida_batch)
    display(spark.read.table(tabla_salida_batch))

# COMMAND ----------

# Salida esperada:
# - Conteo por estado de la tabla `<catálogo>.genai_course.customer_support_tickets`.
# - Muestra de 3 tickets con `status = 'open'` y sus descripciones.
# - Resultado SQL de `ai_query(endpoint => "dphone_endpoint", request => description)` mostrando lado a lado la descripción del ticket y la respuesta del agente.
# - Tabla Delta `<catálogo>.genai_course.tickets_respuestas_batch` persistida con marca de tiempo `processed_at`.
