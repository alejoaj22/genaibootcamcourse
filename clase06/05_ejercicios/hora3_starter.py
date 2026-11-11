# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 3 (STARTER) — Reranking Antes y Después (P@3 y MRR)
# MAGIC **Objetivo:** medir el impacto de un reranker (cross-encoder) sobre la precisión (P@3), la posición del primer relevante (MRR) y la latencia.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** pipeline de 2 etapas (bi-encoder para recall + cross-encoder para ordenar fino) y su costo en latencia.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE (notebook 1.2). Completa las marcas `# TODO`. El notebook corre de arriba hacia abajo desde el inicio: los TODO traen valores por defecto que devuelven 0 hasta que los completes.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

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
vs_endpoint_name = "vs_endpoint"
vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"
vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# 5 consultas donde el orden de los candidatos es crítico, con sus palabras clave de relevancia.
# Un chunk se considera relevante si contiene alguna de las palabras clave (aproximación didáctica).
QUERIES_EVAL = [
    {"query": "Explain the Halo camera in dPhone?", "keywords": ["Halo"]},
    {"query": "What are the titanium materials used in the chassis?", "keywords": ["titanium"]},
    {"query": "How does the Vein-Scan biometric security work?", "keywords": ["Vein"]},
    {"query": "Tell me about Satellite SOS emergency connectivity", "keywords": ["Satellite"]},
    {"query": "What microkernel architecture is used by Fluid OS?", "keywords": ["microkernel", "Fluid"]},
]

def es_relevante(chunk, claves):
    texto = (chunk or "").lower()
    return 1 if any(c.lower() in texto for c in claves) else 0

# COMMAND ----------

# TODO 1: implementa las métricas.
def precision_at_k(relevancias, k=3):
    """relevancias: lista de 1/0 en orden de ranking. Devuelve (relevantes en el top-k) / k."""
    # TODO: reemplaza el valor por defecto por el cálculo correcto
    return 0.0

def reciprocal_rank(relevancias):
    """Devuelve 1 / posición (empezando en 1) del primer relevante; 0.0 si no hay ninguno."""
    # TODO: reemplaza el valor por defecto por el cálculo correcto
    return 0.0

# Autoverificación (no detiene el notebook; indica si tus funciones ya son correctas).
print("precision_at_k OK:", abs(precision_at_k([1, 0, 1, 0], k=3) - 2 / 3) < 1e-9)
print("reciprocal_rank OK:", reciprocal_rank([0, 1, 0]) == 0.5)

# COMMAND ----------

K_CANDIDATOS = 10
K_EVAL = 3

def buscar(consulta, usar_reranker=False):
    """Devuelve (lista de chunks en orden, latencia en ms)."""
    params = dict(query_text=consulta, columns=["id", "path", "chunk"], num_results=K_CANDIDATOS)
    if usar_reranker:
        # TODO 2: agrega el reranker gestionado: params["reranker"] = DatabricksReranker(columns_to_rerank=["chunk"])
        pass
    t0 = time.perf_counter()
    res = vs_index.similarity_search(**params)
    latencia = (time.perf_counter() - t0) * 1000
    filas = res.get("result", {}).get("data_array", []) or []
    # Con columns=["id", "path", "chunk"], cada fila es [id, path, chunk, score].
    return [f[2] for f in filas], latencia

# COMMAND ----------

resultados = []
for item in QUERIES_EVAL:
    q, claves = item["query"], item["keywords"]

    chunks_base, lat_base = buscar(q, usar_reranker=False)
    chunks_rr, lat_rr = buscar(q, usar_reranker=True)

    rels_base = [es_relevante(c, claves) for c in chunks_base]
    rels_rr = [es_relevante(c, claves) for c in chunks_rr]

    resultados.append({
        "query": q,
        "P@3_base": precision_at_k(rels_base, K_EVAL),
        "P@3_rerank": precision_at_k(rels_rr, K_EVAL),
        "RR_base": reciprocal_rank(rels_base),
        "RR_rerank": reciprocal_rank(rels_rr),
        "lat_base_ms": round(lat_base, 1),
        "lat_rerank_ms": round(lat_rr, 1),
    })

df_eval = pd.DataFrame(resultados)
display(df_eval)

# COMMAND ----------

# TODO 3: completa el resumen. MRR = promedio de RR; agrega también P@3 promedio y latencia promedio.
resumen = pd.DataFrame({
    "configuracion": ["Base (sin rerank)", "Con DatabricksReranker"],
    "MRR": [df_eval["RR_base"].mean(), df_eval["RR_rerank"].mean()],
    "P@3_promedio": [None, None],          # TODO
    "latencia_promedio_ms": [None, None],  # TODO
})
display(resumen)

# COMMAND ----------

# TODO 4: análisis y decisión (3-5 líneas).
ANALISIS = """
TODO:
1. ¿El MRR y el P@3 subieron, se mantuvieron o bajaron? ¿Qué significa para el usuario?
2. ¿Cuántos milisegundos extra costó el reranker?
3. ¿Para qué tipo de asistente (chat interactivo, procesamiento batch, autocompletado) lo recomiendas?
"""
print(ANALISIS)

# COMMAND ----------

# Salida esperada (una vez completado):
# - Tabla df_eval con 5 filas: P@3 y RR antes/después y latencias.
# - Resumen con MRR, P@3 promedio y latencia promedio por configuración.
# - En la mayoría de consultas el RR con reranker es igual o mayor; la latencia aumenta.
# - Un análisis escrito con la decisión de arquitectura.
