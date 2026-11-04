# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 3 — Crear Bronze con particionamiento
# MAGIC **Objetivo:** crear una tabla Bronze alternativa particionada por extensión y revisar su detalle e historial Delta.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** diseño físico de tablas Delta (particionamiento, `DESCRIBE DETAIL`, `DESCRIBE HISTORY`, time travel) y cuándo conviene o no particionar.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado. Crea la tabla `bronze_docs_partitioned` (no toca `bronze_docs_raw`).

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

datasource = f"{course.dataset_volume}/docs"

df = (spark.read.format("binaryfile")
      .load(datasource)
      .withColumn("ingestion_time", F.current_timestamp())
      .withColumn("file_name", F.regexp_extract("path", r"([^/]+)$", 1))
      # Columna de partición: la extensión (en minúsculas)
      .withColumn("extension", F.lower(F.regexp_extract("file_name", r"\.([^.]+)$", 1))))

# COMMAND ----------

# Escribimos la tabla Delta particionada por extensión.
(df.write
   .mode("overwrite")
   .partitionBy("extension")
   .saveAsTable("bronze_docs_partitioned"))
print("Tabla bronze_docs_partitioned creada.")

# COMMAND ----------

# Detalle físico de la tabla: formato, particiones, número de archivos y tamaño.
display(spark.sql("DESCRIBE DETAIL bronze_docs_partitioned"))

# COMMAND ----------

# Historial de transacciones Delta (versión 0 = creación).
display(spark.sql("DESCRIBE HISTORY bronze_docs_partitioned"))

# COMMAND ----------

# Consulta con poda de particiones: solo lee la partición pdf.
display(spark.sql("""
    SELECT file_name, length, ingestion_time
    FROM bronze_docs_partitioned
    WHERE extension = 'pdf'
"""))

# COMMAND ----------

# Time travel: reescribimos (versión 1) y leemos la versión 0 para demostrar que Delta conserva historial.
df.write.mode("overwrite").partitionBy("extension").saveAsTable("bronze_docs_partitioned")
display(spark.sql("SELECT COUNT(*) AS filas_version_0 FROM bronze_docs_partitioned VERSION AS OF 0"))
display(spark.sql("SELECT COUNT(*) AS filas_actual FROM bronze_docs_partitioned"))

# COMMAND ----------

# Reflexión: con decenas de documentos, particionar genera muchos archivos pequeños y NO mejora el rendimiento.
# El particionamiento solo se justifica con volúmenes grandes (≥ cientos de GB) y columnas de baja cardinalidad.
print("Regla práctica: partición por columnas de baja cardinalidad y tablas > ~1 TB; en caso contrario, no particionar.")

# COMMAND ----------

# Salida esperada:
# - "Tabla bronze_docs_partitioned creada."
# - DESCRIBE DETAIL: format=delta, partitionColumns=["extension"], numFiles ≥ número de extensiones.
# - DESCRIBE HISTORY: 2 versiones (0 y 1) tras la segunda escritura, operación CREATE OR REPLACE TABLE AS SELECT / WRITE.
# - Consulta filtrada con solo los PDFs.
# - Conteos iguales en versión 0 y versión actual.
