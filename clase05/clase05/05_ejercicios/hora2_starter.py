# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 2 (Hora 2) — Starter: Bronze con documentos propios
# MAGIC **Objetivo:** ingestar documentos de tu reto en una tabla Bronze con metadatos de trazabilidad.
# MAGIC **Tiempo estimado:** 15 minutos
# MAGIC **Concepto que ilustra:** capa Bronze: datos crudos + metadatos de ingestión.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado; documentos subidos a `<volumen>/docs_reto` (o usa `docs`).

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

# Carpeta de origen: cámbiala por la de tus documentos ("docs_reto") si ya los subiste.
CARPETA = "docs"           # TODO: cambiar a "docs_reto" si usas documentos propios
datasource = f"{course.dataset_volume}/{CARPETA}"
print(datasource)

# COMMAND ----------

# TODO 1: lee los archivos con el formato binaryfile y agrega la columna ingestion_time
df_raw = None  # TODO: spark.read.format(...).load(...).withColumn(...)

# COMMAND ----------

# TODO 2: agrega las columnas file_name, extension, size_kb y content_sha256
df_bronze = df_raw  # TODO: encadena withColumn con regexp_extract, lower, round y sha2

# COMMAND ----------

# TODO 3: muestra archivos vacíos (length = 0) y duplicados (mismo content_sha256)
# vacios = ...
# duplicados = ...

# COMMAND ----------

# TODO 4: guarda la tabla Bronze con el nombre "bronze_reto" en modo overwrite
# df_bronze.write....

# COMMAND ----------

# Verificación (no editar): cuenta de filas y detalle Delta.
display(spark.sql("SELECT COUNT(*) AS n_archivos FROM bronze_reto"))
display(spark.sql("DESCRIBE DETAIL bronze_reto"))

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO 5 — Reflexión:** ¿qué metadato adicional agregarías para tu reto y por qué? (escribe aquí tu respuesta)

# COMMAND ----------

# Salida esperada:
# - Con los TODO sin completar fallará la verificación (la tabla bronze_reto no existe).
# - Completado: n_archivos > 0 y DESCRIBE DETAIL con format = delta.
