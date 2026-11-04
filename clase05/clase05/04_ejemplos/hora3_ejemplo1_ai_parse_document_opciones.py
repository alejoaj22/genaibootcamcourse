# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 1 — Opciones de `ai_parse_document()`
# MAGIC **Objetivo:** explorar la salida del parser (v2.0), navegar el VARIANT y usar `imageOutputPath`.
# MAGIC **Tiempo estimado:** 12 minutos
# MAGIC **Concepto que ilustra:** parsing inteligente de documentos: páginas, elementos, tipos, coordenadas y estado de error.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado; AI Functions habilitadas. Procesa solo 1 documento (el más pequeño) para limitar costo.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

from pyspark.sql import functions as F

# Tomamos el documento más pequeño (barato y rápido) directamente del volumen.
df_uno = (spark.read.format("binaryfile")
          .load(f"{course.dataset_volume}/docs")
          .orderBy("length")
          .limit(1))
df_uno.createOrReplaceTempView("doc_muestra")
display(df_uno.select("path", "length"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Opción 1: parseo con la versión 2.0 (la usada en el notebook 1.1B).
# MAGIC SELECT path,
# MAGIC        ai_parse_document(content, map('version', '2.0')) AS parsed_content
# MAGIC FROM doc_muestra

# COMMAND ----------

# Guardamos el resultado en un DataFrame y una vista temporal para explorarlo con SQL.
parsed = spark.sql("""
    SELECT path,
           ai_parse_document(content, map('version', '2.0')) AS parsed_content
    FROM doc_muestra
""")
parsed.createOrReplaceTempView("parsed_muestra")   # se evalúa cada vez que se consulta; es un documento pequeño

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Navegación del VARIANT con la notación de dos puntos.
# MAGIC -- Los nombres de campo siguen la salida v2.0; confirma con la documentación si cambian.
# MAGIC SELECT path,
# MAGIC        parsed_content:error_status AS error_status,
# MAGIC        size(cast(parsed_content:document:pages AS ARRAY<VARIANT>)) AS n_paginas,
# MAGIC        size(cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS n_elementos
# MAGIC FROM parsed_muestra

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Un elemento por fila: tipo, página y fragmento de contenido.
# MAGIC SELECT e:type::string AS tipo,
# MAGIC        e:content::string AS contenido
# MAGIC FROM parsed_muestra,
# MAGIC      LATERAL variant_explode(parsed_content:document:elements) AS t(pos, e)
# MAGIC LIMIT 20

# COMMAND ----------

# Opción 2: parseo con imageOutputPath para guardar imágenes de las páginas/figuras en un volumen.
ruta_imgs = f"{course.dataset_volume}/imgs_ejemplo"
parsed_img = spark.sql(f"""
    SELECT path,
           ai_parse_document(content, map('version', '2.0',
                                          'imageOutputPath', '{ruta_imgs}')) AS parsed_content
    FROM doc_muestra
""")
parsed_img.write.mode("overwrite").saveAsTable("tmp_parsed_con_imagenes")

# COMMAND ----------

# Verificamos las imágenes generadas en el volumen (si el documento contiene páginas/figuras).
try:
    for f in dbutils.fs.ls(ruta_imgs)[:10]:
        print(f.path)
except Exception as e:
    print("No se encontraron imágenes (revisar permisos del volumen):", str(e)[:200])

# COMMAND ----------

# Limpieza opcional de la tabla temporal.
spark.sql("DROP TABLE IF EXISTS tmp_parsed_con_imagenes")

# COMMAND ----------

# Salida esperada:
# - Un documento (el de menor tamaño) con su ruta.
# - parsed_content como VARIANT con campos document (pages, elements), metadata y error_status.
# - Una fila con error_status (nulo/vacío si todo salió bien), número de páginas y de elementos.
# - Hasta 20 elementos con su tipo (title, text, table, figure…) y contenido.
# - Rutas de imágenes bajo <volumen>/imgs_ejemplo/ (o un mensaje si el documento no genera imágenes).
