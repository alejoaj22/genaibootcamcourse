# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 1 (STARTER) — Tres configuraciones de chunking
# MAGIC **Objetivo:** comparar tres configuraciones de `chunk_size`/`chunk_overlap` sobre toda la tabla Silver y recomendar una.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** trade-offs del chunking medidos con métricas objetivas, chunking distribuido con `pandas_udf`.
# MAGIC **Requisitos previos:** tabla `silver_docs_parsed`. Completa las marcas `# TODO`. El notebook se puede ejecutar de arriba hacia abajo desde el inicio: las celdas con TODO traen un valor por defecto mínimo.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade langchain_text_splitters

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

from typing import Iterator
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARADORES = ["\n== page ==\n", "== page ==", "\n\n", "\n", " ", ""]
silver_df = spark.read.table("silver_docs_parsed").where(F.col("plain_text").isNotNull())

# TODO 1: define tres configuraciones (chunk_size, chunk_overlap) distintas.
CONFIGURACIONES = [
    (1500, 150),
    # TODO: agrega dos configuraciones más, por ejemplo (500, 50) y (3000, 300)
]

# COMMAND ----------

def crear_udf(chunk_size, chunk_overlap):
    """Fábrica: devuelve una pandas_udf que trocea con los parámetros dados."""
    @pandas_udf("array<string>")
    def get_chunks(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # El splitter se crea una vez por ejecutor/lote, no por fila.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARADORES)
        def trocear(texto):
            if not isinstance(texto, str) or not texto.strip():
                return []
            return [c for c in splitter.split_text(texto) if c.strip()]
        for s in batch_iter:
            yield s.apply(trocear)
    return get_chunks

# COMMAND ----------

total_caracteres_silver = silver_df.select(F.sum(F.length("plain_text"))).first()[0]
metricas = []
chunks_por_config = {}

for size, overlap in CONFIGURACIONES:
    udf_chunks = crear_udf(size, overlap)
    df = (silver_df
          .withColumn("chunk", F.explode(udf_chunks("plain_text")))
          .select("path", "chunk")
          .withColumn("len", F.length("chunk")))
    df.cache()
    chunks_por_config[(size, overlap)] = df

    # TODO 2: calcula las métricas. Reemplaza los None por expresiones de Spark.
    fila = df.agg(
        F.count("*").alias("n_chunks"),
        F.round(F.avg("len"), 1).alias("avg_len"),
        F.lit(None).alias("p90_len"),             # TODO: percentil 90 con percentile_approx
        F.lit(None).alias("pct_cortos"),          # TODO: % de chunks con len < 100
        F.lit(None).alias("suma_len"),            # TODO: suma de len
        F.lit(None).alias("pct_fin_frase"),       # TODO: % de chunks que terminan en . ! ? o |
        F.max("len").alias("max_len"),
    ).first()

    metricas.append({
        "chunk_size": size, "chunk_overlap": overlap,
        "n_chunks": fila["n_chunks"], "avg_len": fila["avg_len"], "p90_len": fila["p90_len"],
        "pct_cortos": fila["pct_cortos"],
        # TODO 3: factor_duplicacion = suma_len / total_caracteres_silver
        "factor_duplicacion": None,
        "pct_fin_frase": fila["pct_fin_frase"],
        "tokens_max_aprox": fila["max_len"] // 4,
    })

display(pd.DataFrame(metricas))

# COMMAND ----------

# TODO 4: prueba de integridad. Para cada hecho, ¿existe un chunk que contenga el hecho Y "dPhone"?
HECHOS = ["IP69"]   # TODO: agrega al menos dos hechos más, por ejemplo "5,500" y "Helios"

integridad = []
for config, df in chunks_por_config.items():
    fila = {"config": f"{config[0]}/{config[1]}"}
    for hecho in HECHOS:
        # TODO: reemplaza False por un conteo con F.col("chunk").contains(hecho) & F.col("chunk").contains("dPhone")
        fila[hecho] = False
    integridad.append(fila)

display(pd.DataFrame(integridad))

# COMMAND ----------

# TODO 5: escribe tu recomendación (3-5 líneas) citando al menos dos métricas.
RECOMENDACION = """
TODO: Recomendamos la configuración ... porque ...
"""
print(RECOMENDACION)

# COMMAND ----------

# Salida esperada (una vez completado):
# - Tabla con 3 filas y las métricas n_chunks, avg_len, p90_len, pct_cortos, factor_duplicacion,
#   pct_fin_frase y tokens_max_aprox.
# - Tabla de integridad con True/False por hecho y configuración.
# - Recomendación argumentada.
