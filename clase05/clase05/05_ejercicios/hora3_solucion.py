# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 3 (Hora 3) — Solución: Parsear y analizar calidad
# MAGIC **Objetivo:** parsear documentos con `ai_parse_document` y clasificar su calidad con reglas.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** calidad de datos en GenAI y estrategias de fallback.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` y la tabla `bronze_docs_raw` (notebook 1.1A) o `bronze_reto`.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

TABLA_BRONZE = "bronze_docs_raw"
N_DOCS = 3

# COMMAND ----------

# 1) Lectura de Bronze limitada para controlar costos.
df_bronze = spark.read.table(TABLA_BRONZE).orderBy("length").limit(N_DOCS)

# COMMAND ----------

# 2) Parseo con AI Function usando expr(); se conserva path para trazabilidad.
df_parsed = (df_bronze
             .withColumn("parsed_content", F.expr("""ai_parse_document(content, map("version", "2.0"))"""))
             .select("path", "parsed_content"))
# Materializamos el parseo UNA vez (evita pagar el costo varias veces al consultar la vista).
df_parsed.write.mode("overwrite").saveAsTable("tmp_parsed_calidad")

# COMMAND ----------

# 3) Métricas de calidad con SQL sobre el VARIANT.
df_metricas = spark.sql("""
    SELECT path,
           cast(parsed_content:error_status AS STRING) AS error_status,
           size(cast(parsed_content:document:pages AS ARRAY<VARIANT>)) AS n_paginas,
           size(cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS n_elementos,
           length(array_join(
               transform(cast(parsed_content:document:elements AS ARRAY<VARIANT>),
                         e -> coalesce(e:content::string, '')), '\\n')) AS n_caracteres
    FROM tmp_parsed_calidad
""").withColumn("chars_por_pagina",
                F.round(F.col("n_caracteres") / F.greatest(F.col("n_paginas"), F.lit(1)), 1))

# COMMAND ----------

# 4) Reglas de calidad y acción de fallback.
# error_status "vacío" puede venir como NULL, 'null', '[]' o ''; los tratamos como "sin error".
hay_error = (F.col("error_status").isNotNull() &
             ~F.col("error_status").isin("null", "[]", "", "{}"))

df_calidad = (df_metricas
    .withColumn("estado_calidad",
        F.when(hay_error, "ERROR")
         .when(F.col("n_caracteres") < 50, "VACIO")
         .when(F.col("chars_por_pagina") < 200, "SOSPECHOSO")
         .otherwise("OK"))
    .withColumn("accion_fallback",
        F.when(F.col("estado_calidad") == "ERROR", "Cuarentena y revisión humana")
         .when(F.col("estado_calidad") == "VACIO", "Reintentar o usar OCR alternativo")
         .when(F.col("estado_calidad") == "SOSPECHOSO", "Revisión visual con document_renderer")
         .otherwise("Continuar a Silver")))

# COMMAND ----------

# 5) Persistencia y visualización del reporte de calidad.
df_calidad.write.mode("overwrite").saveAsTable("silver_reto_calidad")
display(spark.table("silver_reto_calidad"))
display(spark.table("silver_reto_calidad").groupBy("estado_calidad").count())

# COMMAND ----------

# Limpieza de la tabla temporal.
spark.sql("DROP TABLE IF EXISTS tmp_parsed_calidad")

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusiones (ejemplo):** el documento con menor `chars_por_pagina` es el candidato a ser escaneado; se revisa visualmente con `document_renderer` y, si el texto es ilegible, se envía a cuarentena hasta contar con un parser/OCR alternativo. Los documentos `OK` avanzan a Silver para su conversión a markdown.

# COMMAND ----------

# Salida esperada:
# - Tabla silver_reto_calidad con ≤ 3 filas y las columnas
#   path, error_status, n_paginas, n_elementos, n_caracteres, chars_por_pagina, estado_calidad, accion_fallback.
# - Un conteo por estado_calidad (la mayoría OK con el dataset dPhone; los valores exactos dependen de los documentos).
