# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 2 (Hora 2) — Solución: Bronze con documentos propios
# MAGIC **Objetivo:** ingestar documentos en una tabla Bronze con metadatos de trazabilidad.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** capa Bronze: datos crudos + metadatos de ingestión.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado. Por defecto usa los documentos dPhone (`docs`).

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

CARPETA = "docs"           # Cambiar a "docs_reto" para documentos propios
datasource = f"{course.dataset_volume}/{CARPETA}"
print(datasource)

# COMMAND ----------

# 1) Lectura binaria + marca de tiempo de ingestión.
df_raw = (spark.read.format("binaryfile")
          .load(datasource)
          .withColumn("ingestion_time", F.current_timestamp()))

# COMMAND ----------

# 2) Metadatos de trazabilidad.
df_bronze = (df_raw
    .withColumn("file_name", F.regexp_extract("path", r"([^/]+)$", 1))
    .withColumn("extension", F.lower(F.regexp_extract("file_name", r"\.([^.]+)$", 1)))
    .withColumn("size_kb", F.round(F.col("length") / 1024, 1))
    .withColumn("content_sha256", F.sha2(F.col("content"), 256)))

# COMMAND ----------

# 3) Controles de calidad: vacíos y duplicados (se muestran sin el binario).
vacios = df_bronze.filter("length = 0").drop("content")
duplicados = (df_bronze.groupBy("content_sha256")
              .agg(F.count("*").alias("n"), F.collect_list("file_name").alias("archivos"))
              .filter("n > 1"))
print("Archivos vacíos:", vacios.count())
print("Grupos duplicados:", duplicados.count())
display(vacios)
display(duplicados)

# COMMAND ----------

# 4) Persistencia Delta (overwrite = idempotente para la práctica).
df_bronze.write.mode("overwrite").saveAsTable("bronze_reto")

# COMMAND ----------

# Verificación: filas y detalle Delta.
display(spark.sql("SELECT COUNT(*) AS n_archivos FROM bronze_reto"))
display(spark.sql("DESCRIBE DETAIL bronze_reto"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Reflexión (ejemplo):** agregaría `fuente_origen` (área propietaria del documento) y `version_documento`, porque en el reto es necesario saber qué versión de una política respondió a la consulta y quién es responsable de actualizarla.

# COMMAND ----------

# Salida esperada:
# - Rutas impresas, conteo de vacíos (0 esperado) y de duplicados (0 esperado) con el dataset dPhone.
# - n_archivos igual al número de archivos en la carpeta y DESCRIBE DETAIL con format = delta.
