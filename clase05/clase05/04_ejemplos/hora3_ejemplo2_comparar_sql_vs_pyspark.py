# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 2 — Comparar SQL vs PySpark para el mismo parseo
# MAGIC **Objetivo:** demostrar que SQL y PySpark (`expr`) producen el mismo resultado y comparar ergonomía y tiempo.
# MAGIC **Tiempo estimado:** 12 minutos
# MAGIC **Concepto que ilustra:** las AI Functions son expresiones SQL: se pueden usar desde SQL o desde `expr()` en PySpark.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado; AI Functions y endpoint `databricks-gpt-oss-20b` disponibles. Procesa 1 documento.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

import time
from pyspark.sql import functions as F

# Un solo documento pequeño, para controlar costos y latencia.
df_doc = (spark.read.format("binaryfile")
          .load(f"{course.dataset_volume}/docs")
          .orderBy("length")
          .limit(1))
df_doc.createOrReplaceTempView("doc_unico")

# COMMAND ----------

# Versión SQL (como en la celda del notebook 1.1B).
t0 = time.time()
df_sql = spark.sql("""
    SELECT path,
           ai_parse_document(content, map('version', '2.0')) AS parsed_content
    FROM doc_unico
""")
texto_sql = df_sql.select(F.col("parsed_content").cast("string").alias("s")).collect()[0]["s"]
t_sql = time.time() - t0
print(f"SQL    → {t_sql:.1f} s, {len(texto_sql)} caracteres de JSON")

# COMMAND ----------

# Versión PySpark con expr() (como en la celda programática del notebook 1.1B).
t0 = time.time()
df_py = df_doc.withColumn("parsed_content",
                          F.expr("""ai_parse_document(content, map("version", "2.0"))""")) \
              .select("path", "parsed_content")
texto_py = df_py.select(F.col("parsed_content").cast("string").alias("s")).collect()[0]["s"]
t_py = time.time() - t0
print(f"PySpark → {t_py:.1f} s, {len(texto_py)} caracteres de JSON")

# COMMAND ----------

# Comparación de resultados. El parseo suele ser determinista, pero se verifica en lugar de suponerlo.
print("¿Resultados idénticos?:", texto_sql == texto_py)
print("Diferencia de longitud (caracteres):", abs(len(texto_sql) - len(texto_py)))

# COMMAND ----------

# Ergonomía: el prompt de ai_query con comillas.
# En PySpark dentro de expr() hay varios niveles de comillas; por eso usamos variables y f-strings con cuidado.
endpoint = "databricks-gpt-oss-20b"
prefijo = "Convierte el siguiente JSON de un documento parseado en markdown limpio. Separa páginas con == page ==. No incluyas JSON ni bloques de código.\nJSON:\n"

# Escapamos comillas simples del prompt para incrustarlo en la expresión SQL.
prefijo_sql = prefijo.replace("'", "\\'")

df_md = df_py.withColumn(
    "plain_text",
    F.expr(f"""ai_query('{endpoint}',
                        CONCAT('{prefijo_sql}', CAST(parsed_content AS STRING)),
                        responseFormat => '{{"type":"text"}}')"""))
fila = df_md.select("path", "plain_text").collect()[0]
print(fila["plain_text"][:800])

# COMMAND ----------

# Resumen para discusión en clase.
display(spark.createDataFrame([
    ("SQL",     "Más legible, ideal para exploración y analistas", round(t_sql, 1)),
    ("PySpark", "Integra lógica Python, funciones y control de flujo", round(t_py, 1)),
], ["enfoque", "cuando_usarlo", "segundos"]))

# COMMAND ----------

# Salida esperada:
# - Tiempos de SQL y PySpark similares (segundos a decenas de segundos según el documento).
# - "¿Resultados idénticos?: True" (si fuera False, comentar posibles causas: versión del parser o variabilidad).
# - Primeros 800 caracteres del markdown generado por el LLM, con páginas separadas por "== page ==".
# - Tabla de 2 filas con la comparación de enfoques.
