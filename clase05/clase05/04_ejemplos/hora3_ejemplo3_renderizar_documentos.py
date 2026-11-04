# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 3 — Renderizar documentos parseados (bounding boxes)
# MAGIC **Objetivo:** inspeccionar visualmente lo que el parser detectó usando `document_renderer.py`.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** inspección visual como control de calidad del parsing (cajas delimitadoras, tipo de elemento, contenido).
# MAGIC **Requisitos previos:** `Includes/Lab_Setup` ejecutado; `Includes/document_renderer.py` accesible; este notebook ubicado en una carpeta hermana de `Includes/`.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

import sys, os
from pyspark.sql import functions as F

# Agregamos la carpeta Includes al path de Python para importar el renderizador.
sys.path.append(os.path.abspath('../Includes'))
from document_renderer import render_ai_parse_output, render_ai_parse_output_interactive

# COMMAND ----------

# Elegimos 2 documentos pequeños para limitar costo; guardamos la ruta para identificarlos.
df_docs = (spark.read.format("binaryfile")
           .load(f"{course.dataset_volume}/docs")
           .orderBy("length")
           .limit(2))

# Parseo programático con imágenes de salida (necesarias para dibujar las cajas sobre la página).
parsed_df = (df_docs.withColumn(
                "parsed_content",
                F.expr(f"""ai_parse_document(content, map("version", "2.0",
                                                          "imageOutputPath", "{course.dataset_volume}/imgs")
                                            )"""))
             .select("path", "parsed_content"))

filas = parsed_df.collect()      # 2 filas; el parseo ocurre aquí
for f in filas:
    print("Documento:", f["path"])

# COMMAND ----------

# Renderizado estático del primer documento: páginas con cajas y contenido por elemento.
print("Renderizando:", filas[0]["path"])
render_ai_parse_output(filas[0]["parsed_content"])

# COMMAND ----------

# Renderizado del segundo documento (si existe).
if len(filas) > 1:
    print("Renderizando:", filas[1]["path"])
    render_ai_parse_output(filas[1]["parsed_content"])

# COMMAND ----------

# Versión interactiva (permite explorar elementos con el ratón).
render_ai_parse_output_interactive(filas[0]["parsed_content"])

# COMMAND ----------

# Lista de verificación visual para discutir en clase:
checklist = [
    "¿Las cajas rodean títulos, párrafos y tablas sin solaparse?",
    "¿Se detectaron todas las tablas como tablas (no como texto suelto)?",
    "¿El orden de lectura de los elementos tiene sentido?",
    "¿Hay texto ilegible o con caracteres extraños (señal de escaneo/OCR pobre)?",
    "¿Se perdieron encabezados, pies de página o figuras importantes?",
]
for i, c in enumerate(checklist, 1):
    print(f"{i}. {c}")

# COMMAND ----------

# Salida esperada:
# - Rutas de los 2 documentos más pequeños.
# - Para cada documento, una visualización de página con cajas de colores y el contenido parseado por elemento.
# - Una vista interactiva del primer documento.
# - La lista de 5 preguntas de verificación visual.
