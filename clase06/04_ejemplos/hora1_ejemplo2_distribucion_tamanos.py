# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 2 — Visualizar la distribución de tamaños de los chunks
# MAGIC **Objetivo:** auditar la tabla Gold: cuántos chunks hay por documento, cómo se distribuyen sus tamaños y cuáles son sospechosos de ser ruido.
# MAGIC **Tiempo estimado:** 7 minutos
# MAGIC **Concepto que ilustra:** control de calidad de la capa Gold antes de indexar (percentiles, colas, chunks diminutos).
# MAGIC **Requisitos previos:** haber ejecutado `1- RAG/1.1C- Data Prep (Gold)` (tabla `gold_docs_chunked`).

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# Cargamos la tabla Gold y calculamos longitud en caracteres y tokens aproximados.
from pyspark.sql import functions as F

gold = (spark.read.table("gold_docs_chunked")
        .withColumn("doc_name", F.element_at(F.split("path", "/"), -1))
        .withColumn("chunk_len", F.length("chunk"))
        .withColumn("tokens_aprox", F.ceil(F.col("chunk_len") / 4)))

print("Total de chunks:", gold.count())
print("Documentos distintos:", gold.select("doc_name").distinct().count())
display(gold.select("id", "doc_name", "chunk_len", "tokens_aprox", "chunk").limit(10))

# COMMAND ----------

# Estadísticos y percentiles de la longitud.
display(gold.select(
    F.count("*").alias("n"),
    F.min("chunk_len").alias("min"),
    F.expr("percentile_approx(chunk_len, 0.10)").alias("p10"),
    F.expr("percentile_approx(chunk_len, 0.50)").alias("mediana"),
    F.round(F.avg("chunk_len"), 1).alias("promedio"),
    F.expr("percentile_approx(chunk_len, 0.90)").alias("p90"),
    F.max("chunk_len").alias("max"),
))

# COMMAND ----------

# Histograma de longitudes. Con chunk_size=1500 esperamos una masa cerca de 1500 y una cola a la izquierda.
import matplotlib.pyplot as plt

largos = [r["chunk_len"] for r in gold.select("chunk_len").collect()]

plt.figure(figsize=(9, 4))
plt.hist(largos, bins=20, color="#4C78A8", edgecolor="white")
plt.axvline(1500, color="red", linestyle="--", label="chunk_size = 1500")
plt.axvline(100, color="orange", linestyle=":", label="umbral de ruido = 100")
plt.xlabel("Longitud del chunk (caracteres)")
plt.ylabel("Número de chunks")
plt.title("Distribución de tamaños en gold_docs_chunked")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# Chunks por documento: documentos con muchos chunks dominarán los resultados de búsqueda.
por_doc = (gold.groupBy("doc_name")
               .agg(F.count("*").alias("n_chunks"),
                    F.round(F.avg("chunk_len"), 0).alias("len_promedio"))
               .orderBy(F.desc("n_chunks")))
display(por_doc)

pdf = por_doc.toPandas()
plt.figure(figsize=(9, 4))
plt.barh(pdf.doc_name, pdf.n_chunks, color="#54A24B")
plt.gca().invert_yaxis()
plt.xlabel("Número de chunks")
plt.title("Chunks por documento")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Chunks sospechosos: muy cortos (posible ruido: pies de página, restos de tabla, "- End of Document -").
UMBRAL_RUIDO = 100
ruido = gold.where(F.col("chunk_len") < UMBRAL_RUIDO)
print(f"Chunks con menos de {UMBRAL_RUIDO} caracteres: {ruido.count()}")
display(ruido.select("doc_name", "chunk_len", "chunk"))

# COMMAND ----------

# Chunks duplicados exactos (pueden aparecer si un documento se ingestó dos veces en Bronze).
duplicados = (gold.groupBy("chunk").agg(F.count("*").alias("veces"))
                  .where("veces > 1"))
print("Textos de chunk repetidos:", duplicados.count())
display(duplicados.limit(5))

# COMMAND ----------

# Salida esperada:
# - Total de chunks (decenas) y número de documentos distintos del dataset dPhone.
# - Percentiles: mediana y p90 cercanos a 1500 (sin superarlo); mínimo pequeño.
# - Histograma con un pico junto a la línea roja y una cola de chunks cortos (finales de página).
# - Barra por documento: los manuales largos aportan más chunks.
# - Lista de chunks < 100 caracteres (candidatos a filtrar en Gold) y conteo de duplicados (idealmente 0).
