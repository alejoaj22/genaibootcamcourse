# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 1 (SOLUCIÓN) — Tres configuraciones de chunking
# MAGIC **Objetivo:** comparar tres configuraciones de `chunk_size`/`chunk_overlap` sobre toda la tabla Silver y recomendar una.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** trade-offs del chunking medidos con métricas objetivas, chunking distribuido con `pandas_udf`.
# MAGIC **Requisitos previos:** tabla `silver_docs_parsed`.

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

# Tres configuraciones: pequeña, la del notebook oficial y grande (overlap del 10 % en todas).
CONFIGURACIONES = [
    (500, 50),
    (1500, 150),
    (3000, 300),
]

# COMMAND ----------

def crear_udf(chunk_size, chunk_overlap):
    """Fábrica: devuelve una pandas_udf que trocea con los parámetros dados."""
    @pandas_udf("array<string>")
    def get_chunks(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # El splitter se crea una vez por lote, no por fila (ventaja del pandas_udf de iterador).
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

    fila = df.agg(
        F.count("*").alias("n_chunks"),
        F.round(F.avg("len"), 1).alias("avg_len"),
        F.expr("percentile_approx(len, 0.9)").alias("p90_len"),
        F.round(100 * F.avg((F.col("len") < 100).cast("int")), 1).alias("pct_cortos"),
        F.sum("len").alias("suma_len"),
        F.round(100 * F.avg(F.col("chunk").rlike(r"[.!?|]\s*$").cast("int")), 1).alias("pct_fin_frase"),
        F.max("len").alias("max_len"),
    ).first()

    metricas.append({
        "chunk_size": size, "chunk_overlap": overlap,
        "n_chunks": fila["n_chunks"], "avg_len": fila["avg_len"], "p90_len": fila["p90_len"],
        "pct_cortos": fila["pct_cortos"],
        "factor_duplicacion": round(fila["suma_len"] / total_caracteres_silver, 3),
        "pct_fin_frase": fila["pct_fin_frase"],
        "tokens_max_aprox": fila["max_len"] // 4,
    })

tabla_metricas = pd.DataFrame(metricas)
display(tabla_metricas)

# COMMAND ----------

# Prueba de integridad: el hecho y el nombre del producto deben convivir en algún chunk.
HECHOS = ["IP69", "5,500", "Helios"]

integridad = []
for config, df in chunks_por_config.items():
    fila = {"config": f"{config[0]}/{config[1]}"}
    for hecho in HECHOS:
        n = df.where(F.col("chunk").contains(hecho) & F.col("chunk").contains("dPhone")).count()
        fila[hecho] = n > 0
    integridad.append(fila)

display(pd.DataFrame(integridad))

# COMMAND ----------

# Visualización: distribución de longitudes por configuración.
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 4))
for config, df in chunks_por_config.items():
    largos = [r["len"] for r in df.select("len").collect()]
    plt.hist(largos, bins=20, alpha=0.5, label=f"{config[0]}/{config[1]}")
plt.xlabel("Longitud del chunk (caracteres)")
plt.ylabel("Frecuencia")
plt.title("Distribución de longitudes por configuración")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# Recomendación (se construye leyendo las métricas calculadas).
m = tabla_metricas.set_index(["chunk_size", "chunk_overlap"])
RECOMENDACION = f"""
Recomendamos 1500/150. Frente a 500/50 reduce los chunks de {m.loc[(500,50),'n_chunks']} a {m.loc[(1500,150),'n_chunks']}
y el % de chunks cortos de {m.loc[(500,50),'pct_cortos']}% a {m.loc[(1500,150),'pct_cortos']}%, manteniendo los hechos clave
junto al nombre del producto. Frente a 3000/300, sus chunks (p90 = {m.loc[(1500,150),'p90_len']} caracteres,
~{m.loc[(1500,150),'tokens_max_aprox']} tokens máx.) producen embeddings más específicos y prompts más baratos.
El factor de duplicación ({m.loc[(1500,150),'factor_duplicacion']}) es un costo de almacenamiento aceptable.
"""
print(RECOMENDACION)

# COMMAND ----------

# Liberamos la caché.
for df in chunks_por_config.values():
    df.unpersist()

# COMMAND ----------

# Salida esperada:
# - Tabla con 3 filas: n_chunks decrece (500/50 > 1500/150 > 3000/300); avg_len y p90_len crecen sin superar chunk_size;
#   pct_cortos es mayor en 500/50; factor_duplicacion entre ~1.0 y ~1.15; tokens_max_aprox ≈ chunk_size / 4.
# - Tabla de integridad: 1500/150 y 3000/300 mantienen IP69, 5,500 y Helios junto a "dPhone" (True);
#   500/50 puede separar alguno (False).
# - Histograma superpuesto de las tres distribuciones.
# - Recomendación impresa con valores reales de las métricas.
