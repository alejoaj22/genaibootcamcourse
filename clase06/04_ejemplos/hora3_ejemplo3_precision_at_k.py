# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 3 — Métricas de retrieval: precision@k, recall@k, MRR y nDCG@k
# MAGIC **Objetivo:** implementar las cuatro métricas y comparar objetivamente semántica, full-text e híbrida.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** evaluación offline del retriever con un conjunto de consultas etiquetadas (relevancia aproximada por palabras clave).
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE y tabla `gold_docs_chunked`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import math
import pandas as pd
from pyspark.sql import functions as F
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index("vs_endpoint", f"{course.catalog}.{course.schema}.dphone_vs_index")
gold = spark.read.table("gold_docs_chunked")

# COMMAND ----------

# Conjunto de evaluación: consulta + palabras clave. Un chunk es "relevante" si contiene alguna palabra clave.
# Es una aproximación didáctica; en producción la relevancia se etiqueta a mano o con un LLM juez.
EVAL = [
    ("Which phone has the highest ingress protection rating?", ["IP69"]),
    ("What processor powers the dPhone D1?", ["Helios"]),
    ("Which operating system runs apps from Android and iOS?", ["Fluid"]),
    ("Tell me about the Halo camera", ["Halo"]),
    ("Does the phone support emergency satellite messaging?", ["Satellite"]),
]

def es_relevante(chunk, claves):
    texto = (chunk or "").lower()
    return any(c.lower() in texto for c in claves)

def total_relevantes(claves):
    """Número de chunks relevantes en todo el corpus Gold (denominador del recall)."""
    cond = None
    for c in claves:
        e = F.lower("chunk").contains(c.lower())
        cond = e if cond is None else (cond | e)
    return gold.where(cond).count()

# COMMAND ----------

# Implementación de las métricas con relevancia binaria.
def precision_at_k(rels, k):
    return sum(rels[:k]) / k

def recall_at_k(rels, k, n_relevantes):
    return sum(rels[:k]) / n_relevantes if n_relevantes else 0.0

def reciprocal_rank(rels):
    for i, r in enumerate(rels, start=1):
        if r:
            return 1 / i
    return 0.0

def ndcg_at_k(rels, k, n_relevantes):
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))
    ideal = [1] * min(n_relevantes, k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, len(ideal) + 1))
    return dcg / idcg if idcg else 0.0

# Verificación con el ejemplo de la diapositiva: top-5 con relevantes en posiciones 2, 3 y 5; 4 relevantes en total.
rels_demo = [0, 1, 1, 0, 1]
print("P@5  =", precision_at_k(rels_demo, 5))           # 0.6
print("R@5  =", recall_at_k(rels_demo, 5, 4))           # 0.75
print("RR   =", reciprocal_rank(rels_demo))             # 0.5
print("nDCG =", round(ndcg_at_k(rels_demo, 5, 4), 3))

# COMMAND ----------

# Evaluación de los tres tipos de búsqueda con k = 5.
K = 5
TIPOS = ["semantic", "FULL_TEXT", "hybrid"]

def recuperar(consulta, tipo, k):
    params = dict(query_text=consulta, columns=["id", "chunk"], num_results=k)
    if tipo != "semantic":
        params["query_type"] = tipo
    res = vs_index.similarity_search(**params)
    return [fila[1] for fila in (res.get("result", {}).get("data_array", []) or [])]

filas = []
for consulta, claves in EVAL:
    n_rel = total_relevantes(claves)
    for tipo in TIPOS:
        chunks = recuperar(consulta, tipo, K)
        rels = [1 if es_relevante(c, claves) else 0 for c in chunks]
        rels += [0] * (K - len(rels))   # si el índice devolvió menos de K resultados
        filas.append({
            "consulta": consulta, "tipo": tipo, "n_relevantes_corpus": n_rel,
            f"P@{K}": precision_at_k(rels, K),
            f"R@{K}": recall_at_k(rels, K, n_rel),
            "RR": reciprocal_rank(rels),
            f"nDCG@{K}": ndcg_at_k(rels, K, n_rel),
        })

detalle = pd.DataFrame(filas)
display(detalle)

# COMMAND ----------

# Resumen por tipo (el promedio de RR es el MRR).
resumen = (detalle.groupby("tipo")[[f"P@{K}", f"R@{K}", "RR", f"nDCG@{K}"]].mean()
                  .rename(columns={"RR": "MRR"}).round(3).reset_index())
display(resumen)

# COMMAND ----------

# Gráfico de barras agrupadas por métrica.
import matplotlib.pyplot as plt
import numpy as np

metricas = [f"P@{K}", f"R@{K}", "MRR", f"nDCG@{K}"]
x = np.arange(len(metricas))
ancho = 0.25
plt.figure(figsize=(9, 4))
for i, tipo in enumerate(resumen["tipo"]):
    valores = resumen.loc[resumen["tipo"] == tipo, metricas].values[0]
    plt.bar(x + i * ancho, valores, ancho, label=tipo)
plt.xticks(x + ancho, metricas)
plt.ylim(0, 1.05)
plt.ylabel("Valor")
plt.title(f"Métricas de retrieval por tipo de búsqueda (k={K})")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# Salida esperada:
# - Verificación: P@5 = 0.6, R@5 = 0.75, RR = 0.5 y nDCG ≈ 0.592.
# - Tabla de detalle con 15 filas (5 consultas × 3 tipos) y el número de chunks relevantes del corpus.
# - Resumen por tipo: hybrid suele tener MRR y nDCG iguales o superiores a los otros dos;
#   FULL_TEXT destaca en consultas con nombres propios (Helios, Halo) y puede fallar en las parafraseadas.
# - Gráfico de barras con 4 métricas × 3 tipos de búsqueda (valores entre 0 y 1).
