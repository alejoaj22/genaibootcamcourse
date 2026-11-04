# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 3 — Explorar el dataset dPhone
# MAGIC **Objetivo:** conocer qué archivos y tablas tiene el dataset del curso antes de ingestarlos.
# MAGIC **Tiempo estimado:** 8 minutos
# MAGIC **Concepto que ilustra:** inspección previa de fuentes (primer paso de calidad de datos) y diferencia entre datos no estructurados (docs) y estructurados (CSV/tablas).
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado (crea el volumen `dataset`).

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

# Raíz del dataset y carpeta de documentos.
raiz = course.dataset_volume
print("Raíz del dataset:", raiz)

# Listamos las carpetas de primer nivel.
for f in dbutils.fs.ls(raiz):
    print(f"{f.name:<25} tamaño={f.size}")

# COMMAND ----------

# Listamos los documentos de la carpeta docs con su tamaño en KB.
ruta_docs = f"{raiz}/docs"
archivos = dbutils.fs.ls(ruta_docs)
print(f"Total de archivos en docs: {len(archivos)}")

filas = []
for f in archivos:
    extension = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else "sin_extension"
    filas.append((f.name, extension, round(f.size / 1024, 1)))

df_docs = spark.createDataFrame(filas, ["archivo", "extension", "tamano_kb"])
display(df_docs.orderBy("tamano_kb"))

# COMMAND ----------

# Resumen por extensión: ¿qué formatos tenemos y cuánto pesan?
from pyspark.sql import functions as F

display(df_docs.groupBy("extension")
              .agg(F.count("*").alias("n_archivos"),
                   F.round(F.sum("tamano_kb"), 1).alias("total_kb"),
                   F.round(F.max("tamano_kb"), 1).alias("max_kb"))
              .orderBy(F.desc("n_archivos")))

# COMMAND ----------

# Contraste: datos estructurados del caso (tablas creadas por Lab_Setup para los agentes).
# Creamos las tablas solo si no existen (create_agent_tables usa CREATE TABLE IF NOT EXISTS).
course.create_agent_tables()
display(spark.sql("SELECT * FROM smartphones LIMIT 5"))

# COMMAND ----------

# Pregunta de reflexión para el grupo (sin código):
# 1) ¿Qué preguntas se responden mejor con tablas (SQL) y cuáles con documentos (RAG)?
# 2) ¿Hay documentos de tamaño extremo (muy pequeños o muy grandes)?
print("Tablas → cifras exactas y filtros. Documentos → explicaciones, políticas, procedimientos.")

# COMMAND ----------

# Salida esperada:
# - Raíz: /Volumes/<catálogo>/genai_course/dataset con carpetas como docs/ y raw/.
# - Un listado de N documentos con extensión y tamaño en KB (la lista exacta depende del dataset v1).
# - Un resumen agrupado por extensión.
# - 5 filas de la tabla smartphones.
