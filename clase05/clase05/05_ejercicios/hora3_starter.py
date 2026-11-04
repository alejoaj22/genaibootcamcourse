# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 3 (Hora 3) — Starter: Parsear y analizar calidad
# MAGIC **Objetivo:** parsear documentos con `ai_parse_document` y clasificar su calidad con reglas.
# MAGIC **Tiempo estimado:** 15 minutos
# MAGIC **Concepto que ilustra:** calidad de datos en GenAI y estrategias de fallback.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` y una tabla Bronze (`bronze_reto` del ejercicio 2 o `bronze_docs_raw` del notebook 1.1A).

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

TABLA_BRONZE = "bronze_docs_raw"   # TODO: cambiar a "bronze_reto" si lo creaste
N_DOCS = 3                         # Mantener pequeño para controlar costos

# COMMAND ----------

# TODO 1: lee la tabla Bronze, ordénala por length y limita a N_DOCS documentos
df_bronze = None  # TODO

# COMMAND ----------

# TODO 2: parsea con ai_parse_document usando expr(); conserva path
df_parsed = None  # TODO: df_bronze.withColumn("parsed_content", F.expr(...)).select("path","parsed_content")
# df_parsed.createOrReplaceTempView("parsed_tmp")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO 3: calcula error_status, n_paginas, n_elementos, n_caracteres y chars_por_pagina
# MAGIC -- Pista: array_join(transform(cast(parsed_content:document:elements AS ARRAY<VARIANT>), e -> e:content::string), '\n')
# MAGIC SELECT path FROM parsed_tmp

# COMMAND ----------

# TODO 4: clasifica estado_calidad (ERROR, VACIO, SOSPECHOSO, OK) y accion_fallback con when/otherwise
# df_calidad = ...

# COMMAND ----------

# TODO 5: guarda df_calidad en la tabla "silver_reto_calidad" (overwrite) y muéstrala
# df_calidad.write...
# display(spark.table("silver_reto_calidad"))

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO 6 — Conclusiones:** (3-5 líneas) ¿qué documento es el más problemático y qué harías?

# COMMAND ----------

# Salida esperada:
# - Con TODOs sin completar, el notebook falla en la primera celda con None (comportamiento esperado).
# - Completado: tabla silver_reto_calidad con una fila por documento y columnas
#   path, error_status, n_paginas, n_elementos, n_caracteres, chars_por_pagina, estado_calidad, accion_fallback.
