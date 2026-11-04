# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 1 — Ingestar diferentes formatos con `binaryfile`
# MAGIC **Objetivo:** leer documentos por tipo de archivo usando `pathGlobFilter` y `recursiveFileLookup`.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** Bronze es agnóstico al formato: solo guarda bytes + metadatos; el filtrado por tipo se hace en la lectura.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

datasource = f"{course.dataset_volume}/docs"
print("Fuente:", datasource)

# COMMAND ----------

# 1) Lectura de TODOS los formatos (equivalente a la celda del notebook 1.1A).
df_todo = (spark.read.format("binaryfile")
           .load(datasource)
           .withColumn("ingestion_time", F.current_timestamp()))
print("Total de archivos:", df_todo.count())
df_todo.printSchema()

# COMMAND ----------

# 2) Descubrimos las extensiones presentes para iterar sobre ellas.
extensiones = sorted({f.name.rsplit(".", 1)[-1].lower()
                      for f in dbutils.fs.ls(datasource) if "." in f.name})
print("Extensiones encontradas:", extensiones)

# COMMAND ----------

# 3) Leemos cada formato por separado con pathGlobFilter y contamos.
resultados = []
for ext in extensiones:
    df_ext = (spark.read.format("binaryfile")
              .option("pathGlobFilter", f"*.{ext}")      # solo archivos con esa extensión
              .option("recursiveFileLookup", "true")      # incluye subcarpetas
              .load(datasource))
    resultados.append((ext, df_ext.count()))

display(spark.createDataFrame(resultados, ["extension", "n_archivos"]))

# COMMAND ----------

# 4) Caso práctico: ingerir SOLO PDFs en un DataFrame (no se guarda ninguna tabla).
df_pdf = (spark.read.format("binaryfile")
          .option("pathGlobFilter", "*.pdf")
          .load(datasource)
          .withColumn("ingestion_time", F.current_timestamp()))
display(df_pdf.select("path", "length", "modificationTime", "ingestion_time"))

# COMMAND ----------

# 5) Múltiples patrones: PDF o DOCX usando llaves en el glob.
df_multi = (spark.read.format("binaryfile")
            .option("pathGlobFilter", "*.{pdf,docx}")
            .load(datasource))
print("Archivos PDF o DOCX:", df_multi.count())

# COMMAND ----------

# Salida esperada:
# - Esquema: path (string), modificationTime (timestamp), length (long), content (binary), ingestion_time (timestamp).
# - Lista de extensiones (por ejemplo ['pdf', ...] según el dataset).
# - Tabla con el conteo por extensión; la suma coincide con el total de archivos.
# - Un DataFrame con solo PDFs y un conteo para PDF o DOCX.
