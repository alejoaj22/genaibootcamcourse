# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 2 — Inspeccionar metadata para trazabilidad
# MAGIC **Objetivo:** enriquecer la lectura binaria con nombre de archivo, extensión, tamaño en KB y hash para detectar duplicados.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** metadatos de ingestión (ruta, tamaño, timestamps, hash) como base de la trazabilidad en RAG.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado. No modifica `bronze_docs_raw`.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

datasource = f"{course.dataset_volume}/docs"

df = (spark.read.format("binaryfile")
      .load(datasource)
      .withColumn("ingestion_time", F.current_timestamp()))

# COMMAND ----------

# Metadatos derivados:
# - file_name: último segmento de la ruta
# - extension: texto tras el último punto, en minúsculas
# - size_kb: tamaño legible
# - content_sha256: huella del contenido (detecta duplicados y cambios)
df_meta = (df
    .withColumn("file_name", F.regexp_extract("path", r"([^/]+)$", 1))
    .withColumn("extension", F.lower(F.regexp_extract("file_name", r"\.([^.]+)$", 1)))
    .withColumn("size_kb", F.round(F.col("length") / 1024, 1))
    .withColumn("content_sha256", F.sha2(F.col("content"), 256))
    .drop("content")   # soltamos el binario solo para visualizar metadatos livianos
)
display(df_meta.orderBy("size_kb"))

# COMMAND ----------

# Control de calidad 1: archivos vacíos (length = 0) o sospechosamente pequeños (< 1 KB).
display(df_meta.filter("length = 0 OR size_kb < 1"))

# COMMAND ----------

# Control de calidad 2: duplicados exactos (mismo hash, distinto nombre).
display(df_meta.groupBy("content_sha256")
               .agg(F.count("*").alias("n"), F.collect_list("file_name").alias("archivos"))
               .filter("n > 1"))

# COMMAND ----------

# Estadísticas de tamaño para decidir recursos de cómputo en Silver.
display(df_meta.agg(F.count("*").alias("n_archivos"),
                    F.round(F.min("size_kb"), 1).alias("min_kb"),
                    F.round(F.avg("size_kb"), 1).alias("promedio_kb"),
                    F.round(F.max("size_kb"), 1).alias("max_kb"),
                    F.round(F.sum("size_kb") / 1024, 2).alias("total_mb")))

# COMMAND ----------

# Salida esperada:
# - Tabla de metadatos con columnas: path, modificationTime, length, ingestion_time, file_name, extension, size_kb, content_sha256.
# - Filtro de vacíos: probablemente 0 filas con el dataset estándar.
# - Duplicados: normalmente 0 filas (si hay, aparecen agrupados por hash).
# - Una fila con estadísticas de tamaño (mínimo, promedio, máximo, total en MB).
