# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 2 — Búsqueda semántica: scores, paráfrasis, idioma y `num_results`
# MAGIC **Objetivo:** consultar el índice con `similarity_search` y aprender a leer e interpretar los resultados.
# MAGIC **Tiempo estimado:** 8 minutos
# MAGIC **Concepto que ilustra:** la búsqueda semántica encuentra significado y no palabras; sensibilidad al idioma del modelo (`gte-large-en`) y al tamaño del top-k.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE (notebook 1.2 o `hora2_ejemplo1`).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# Conexión al índice existente.
from databricks.vector_search.client import VectorSearchClient
import pandas as pd

vs_client = VectorSearchClient(disable_notice=True)
vs_endpoint_name = "vs_endpoint"
vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"
vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)

def buscar(consulta, k=3, **kwargs):
    """Ejecuta similarity_search y devuelve un DataFrame legible: documento, score y vista previa."""
    res = vs_index.similarity_search(query_text=consulta, columns=["id", "path", "chunk"],
                                     num_results=k, **kwargs)
    columnas = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
    df = pd.DataFrame(res.get("result", {}).get("data_array", []) or [], columns=columnas)
    if df.empty:
        return df
    df.insert(0, "rank", range(1, len(df) + 1))
    df["documento"] = df["path"].str.split("/").str[-1]
    df["vista_previa"] = df["chunk"].str.slice(0, 160)
    return df[["rank", "documento", "score", "vista_previa", "id"]]

# COMMAND ----------

# 1) Consulta del notebook oficial.
display(buscar("Explain the features of the dPhone D1 Lite"))

# COMMAND ----------

# 2) Paráfrasis: misma intención, palabras distintas. Los resultados deberían coincidir en gran medida.
parafrasis = [
    "Is the dPhone resistant to water?",
    "Can I take my dPhone swimming?",
    "waterproof and dust protection rating",
]
for q in parafrasis:
    df = buscar(q, k=3)
    print(f"\n▶ {q}")
    print(df[["rank", "documento", "score"]].to_string(index=False))
    print("   top-1:", df.iloc[0]["vista_previa"][:120] if not df.empty else "(sin resultados)")

# COMMAND ----------

# 3) Idioma: el modelo gte-large-en está entrenado en inglés; comparamos la misma pregunta en dos idiomas.
for q in ["How long does the battery last?", "¿Cuánto dura la batería?"]:
    df = buscar(q, k=3)
    print(f"\n▶ {q}  | score top-1 = {df.iloc[0]['score']:.4f}")
    print(df[["rank", "documento", "score"]].to_string(index=False))

# COMMAND ----------

# 4) Efecto de num_results: más resultados = más recall, pero también más ruido para el LLM.
df10 = buscar("Explain the Halo camera in dPhone", k=10)
display(df10)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 3.5))
plt.plot(df10["rank"], df10["score"], marker="o")
plt.xlabel("Posición (rank)")
plt.ylabel("Score de similitud")
plt.title("Caída del score a lo largo del top-10")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# 5) Consulta "trampa": algo que no está en los documentos. La búsqueda vectorial SIEMPRE devuelve algo.
df_trampa = buscar("What is the recipe for bandeja paisa?", k=3)
display(df_trampa)
print("Lección: el retriever no sabe decir 'no sé'. Un umbral de score o el propio LLM deben filtrar.")

# COMMAND ----------

# Salida esperada:
# - Consulta 1: chunks de documentos sobre el dPhone D1 / D1 Lite con scores en un rango estrecho.
# - Paráfrasis: los tres top-1 provienen del mismo documento o sección (IP69 / resistencia al agua).
# - Idioma: el score top-1 de la consulta en español es menor y el ranking puede cambiar.
# - Gráfico: el score cae con la posición; los primeros 3-5 resultados concentran la relevancia.
# - Consulta trampa: devuelve chunks (irrelevantes) con scores claramente menores a los de consultas válidas.
