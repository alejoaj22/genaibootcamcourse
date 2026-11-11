# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 2 — Reranking con DatabricksReranker y FlashRank (rank-T5-flan)
# MAGIC **Objetivo:** recuperar 10 candidatos y reordenarlos con dos rerankers, comparando posiciones y latencia.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** patrón de dos etapas (bi-encoder para recall + cross-encoder para precisión) y su costo en latencia.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE; acceso a internet la primera vez para descargar `rank-T5-flan` (se guarda en el volumen `_cache`).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch flashrank

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import time
import pandas as pd
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index("vs_endpoint", f"{course.catalog}.{course.schema}.dphone_vs_index")

def a_df(res):
    columnas = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
    df = pd.DataFrame(res.get("result", {}).get("data_array", []) or [], columns=columnas)
    if not df.empty:
        df.insert(0, "rank", range(1, len(df) + 1))
        df["id"] = df["id"].astype(int)
        df["documento"] = df["path"].str.split("/").str[-1]
    return df

CONSULTA = "Explain the Halo camera in dPhone?"
K = 10

# COMMAND ----------

# Etapa 1: búsqueda semántica sin reranking (línea base).
t0 = time.perf_counter()
base = a_df(vs_index.similarity_search(query_text=CONSULTA, columns=["id", "path", "chunk"], num_results=K))
lat_base = time.perf_counter() - t0
display(base[["rank", "id", "documento", "score"]])

# COMMAND ----------

# Etapa 2a: DatabricksReranker dentro de similarity_search (reranking gestionado).
t0 = time.perf_counter()
dbx = a_df(vs_index.similarity_search(
    query_text=CONSULTA, columns=["id", "path", "chunk"], num_results=K,
    reranker=DatabricksReranker(columns_to_rerank=["chunk"]),
))
lat_dbx = time.perf_counter() - t0
display(dbx[["rank", "id", "documento", "score"]])

# COMMAND ----------

# Etapa 2b: FlashRank sobre los mismos candidatos de la línea base (corre en el notebook).
from flashrank import Ranker, RerankRequest

cache_dir = f"{course.cache_volume}/flashrank"
t0 = time.perf_counter()
ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)
lat_carga = time.perf_counter() - t0

pasajes = [{"id": int(r.id), "text": r.chunk, "meta": {"documento": r.documento}} for r in base.itertuples()]
t0 = time.perf_counter()
salida = ranker.rerank(RerankRequest(query=CONSULTA, passages=pasajes))
lat_flash = time.perf_counter() - t0

flash = pd.DataFrame([{"rank": i + 1, "id": int(p["id"]), "documento": p["meta"]["documento"],
                       "score": float(p["score"]), "chunk": p["text"]} for i, p in enumerate(salida)])
display(flash[["rank", "id", "documento", "score"]])

# COMMAND ----------

# Comparación de posiciones: ¿cómo se movió cada chunk de la línea base?
pos_base = dict(zip(base["id"], base["rank"]))
pos_dbx = dict(zip(dbx["id"], dbx["rank"]))
pos_flash = dict(zip(flash["id"], flash["rank"]))

comparacion = pd.DataFrame([{
    "id": i,
    "documento": base.loc[base["id"] == i, "documento"].iloc[0],
    "rank_base": pos_base[i],
    "rank_databricks": pos_dbx.get(i),
    "rank_flashrank": pos_flash.get(i),
    "menciona_halo": "halo" in base.loc[base["id"] == i, "chunk"].iloc[0].lower(),
} for i in base["id"]])
display(comparacion.sort_values("rank_base"))

# COMMAND ----------

# Top-3 de cada método: ¿cuántos mencionan "Halo"?
for nombre, df in [("Base", base), ("DatabricksReranker", dbx), ("FlashRank", flash)]:
    n = df.head(3)["chunk"].str.contains("halo", case=False).sum()
    print(f"{nombre:<20} → {n}/3 chunks del top-3 mencionan 'Halo'")

# COMMAND ----------

# Latencias.
lat = pd.DataFrame([
    ("Búsqueda base (k=10)", round(lat_base * 1000)),
    ("Búsqueda + DatabricksReranker", round(lat_dbx * 1000)),
    ("FlashRank: carga del modelo (una vez)", round(lat_carga * 1000)),
    ("FlashRank: rerank de 10 pasajes", round(lat_flash * 1000)),
], columns=["etapa", "milisegundos"])
display(lat)

# COMMAND ----------

# Salida esperada:
# - Tres tablas top-10 (base, DatabricksReranker, FlashRank) con los mismos o casi los mismos ids en distinto orden.
# - Tabla de comparación: los chunks que mencionan "Halo" suben de posición tras el reranking.
# - El número de chunks con "Halo" en el top-3 es igual o mayor con reranking que en la base.
# - Latencias: búsqueda base en decenas/cientos de ms; el reranking añade latencia; la carga de FlashRank
#   es la más costosa la primera vez (descarga) y casi nula después (caché en el volumen).
