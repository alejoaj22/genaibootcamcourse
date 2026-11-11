# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 3 (SOLUCIÓN) — Reranking Antes y Después (P@3 y MRR)
# MAGIC **Objetivo:** medir el impacto de un reranker (cross-encoder) sobre la precisión (P@3), la posición del primer relevante (MRR) y la latencia.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** pipeline de 2 etapas (bi-encoder para recall + cross-encoder para ordenar fino) y su costo en latencia.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE (notebook 1.2).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import time
import pandas as pd
import matplotlib.pyplot as plt
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

vs_client = VectorSearchClient(disable_notice=True)
vs_endpoint_name = "vs_endpoint"
vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"
vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# 5 consultas donde el orden de los candidatos es crítico, con sus palabras clave de relevancia.
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

# 1) Métricas.
def precision_at_k(relevancias, k=3):
    """(relevantes en el top-k) / k. Si llegaron menos de k resultados, los faltantes cuentan como no relevantes."""
    return sum(relevancias[:k]) / k

def reciprocal_rank(relevancias):
    """1 / posición del primer relevante (posiciones desde 1); 0.0 si no hay relevantes."""
    for i, r in enumerate(relevancias, start=1):
        if r:
            return 1 / i
    return 0.0

print("precision_at_k OK:", abs(precision_at_k([1, 0, 1, 0], k=3) - 2 / 3) < 1e-9)
print("reciprocal_rank OK:", reciprocal_rank([0, 1, 0]) == 0.5)

# COMMAND ----------

K_CANDIDATOS = 10
K_EVAL = 3

def buscar(consulta, usar_reranker=False):
    """Devuelve (lista de chunks en orden, latencia en ms)."""
    params = dict(query_text=consulta, columns=["id", "path", "chunk"], num_results=K_CANDIDATOS)
    if usar_reranker:
        # El reranker gestionado reordena los candidatos dentro del propio servicio.
        params["reranker"] = DatabricksReranker(columns_to_rerank=["chunk"])
    t0 = time.perf_counter()
    res = vs_index.similarity_search(**params)
    latencia = (time.perf_counter() - t0) * 1000
    filas = res.get("result", {}).get("data_array", []) or []
    # Con columns=["id", "path", "chunk"], cada fila es [id, path, chunk, score] → el chunk está en la posición 2.
    return [f[2] for f in filas], latencia

# Calentamiento: la primera llamada suele ser más lenta; la descartamos para no sesgar latencias.
_ = buscar("warm up"), buscar("warm up", usar_reranker=True)

# COMMAND ----------

# 2) Evaluación antes/después.
resultados = []
for item in QUERIES_EVAL:
    q, claves = item["query"], item["keywords"]

    chunks_base, lat_base = buscar(q, usar_reranker=False)
    chunks_rr, lat_rr = buscar(q, usar_reranker=True)

    rels_base = [es_relevante(c, claves) for c in chunks_base]
    rels_rr = [es_relevante(c, claves) for c in chunks_rr]

    resultados.append({
        "query": q,
        "rels_top5_base": str(rels_base[:5]),
        "rels_top5_rerank": str(rels_rr[:5]),
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

# 3) Resumen.
resumen = pd.DataFrame({
    "configuracion": ["Base (sin rerank)", "Con DatabricksReranker"],
    "MRR": [df_eval["RR_base"].mean(), df_eval["RR_rerank"].mean()],
    "P@3_promedio": [df_eval["P@3_base"].mean(), df_eval["P@3_rerank"].mean()],
    "latencia_promedio_ms": [df_eval["lat_base_ms"].mean(), df_eval["lat_rerank_ms"].mean()],
}).round(3)
display(resumen)

# COMMAND ----------

# 4) Visualización: calidad (izquierda) y latencia (derecha).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

metricas = ["MRR", "P@3_promedio"]
x = range(len(metricas))
ancho = 0.35
ax1.bar([p - ancho / 2 for p in x], resumen.loc[0, metricas], width=ancho, label="Base", color="#4C78A8")
ax1.bar([p + ancho / 2 for p in x], resumen.loc[1, metricas], width=ancho, label="Rerank", color="#F58518")
ax1.set_xticks(list(x))
ax1.set_xticklabels(metricas)
ax1.set_ylim(0, 1.05)
ax1.set_title("Calidad del retrieval")
ax1.legend()

ax2.bar(resumen["configuracion"], resumen["latencia_promedio_ms"], color=["#4C78A8", "#F58518"], width=0.5)
ax2.set_ylabel("ms por consulta")
ax2.set_title("Latencia promedio")

plt.tight_layout()
plt.show()

# COMMAND ----------

# 5) Análisis construido a partir de los resultados reales (no se asume que el reranker siempre gana).
mrr_b, mrr_r = resumen.loc[0, "MRR"], resumen.loc[1, "MRR"]
p3_b, p3_r = resumen.loc[0, "P@3_promedio"], resumen.loc[1, "P@3_promedio"]
lat_extra = resumen.loc[1, "latencia_promedio_ms"] - resumen.loc[0, "latencia_promedio_ms"]

if mrr_r > mrr_b or p3_r > p3_b:
    veredicto = ("El reranker mejora la calidad: conviene usarlo en el chat de soporte, donde unos cientos de ms "
                 "extra son aceptables y permiten enviar menos chunks (y más precisos) al LLM.")
elif mrr_r == mrr_b and p3_r == p3_b:
    veredicto = ("El reranker no cambia la calidad en este conjunto (el bi-encoder ya ordenaba bien). "
                 "Solo agrega latencia: no se justifica hasta tener consultas más difíciles o un corpus mayor.")
else:
    veredicto = ("El reranker empeora el orden en este conjunto: revisar las consultas afectadas antes de adoptarlo.")

print(f"""
MRR: {mrr_b:.3f} → {mrr_r:.3f} | P@3: {p3_b:.3f} → {p3_r:.3f} | Latencia extra: {lat_extra:.0f} ms por consulta
Veredicto: {veredicto}
Nota: con solo 5 consultas y relevancia por palabras clave, la conclusión es orientativa; en la Clase 9 se evaluará con LLM-judge.
""")

# COMMAND ----------

# Salida esperada:
# - Verificación de las métricas: "precision_at_k OK: True" y "reciprocal_rank OK: True".
# - df_eval con 5 filas: vectores de relevancia top-5, P@3, RR y latencias antes/después.
# - Resumen: MRR y P@3 con reranker iguales o mayores en la mayoría de los casos; latencia mayor.
# - Dos gráficos (calidad y latencia) y un veredicto coherente con los números obtenidos.
