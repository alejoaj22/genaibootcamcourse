# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 1 — Búsqueda híbrida vs semántica vs full-text (y filtros por metadata)
# MAGIC **Objetivo:** comparar lado a lado los tres modos de búsqueda en consultas conceptuales y de términos exactos, y aplicar filtros por `path`.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** BM25 (palabras), embeddings (significado), fusión híbrida (RRF) y filtrado por metadata.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` ONLINE.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import pandas as pd

vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index("vs_endpoint", f"{course.catalog}.{course.schema}.dphone_vs_index")

def buscar(consulta, tipo="semantic", k=5, filtros=None):
    """tipo: 'semantic' (por defecto del índice), 'FULL_TEXT' o 'hybrid'."""
    params = dict(query_text=consulta, columns=["id", "path", "chunk"], num_results=k)
    if tipo != "semantic":
        params["query_type"] = tipo
    if filtros:
        params["filters"] = filtros
    res = vs_index.similarity_search(**params)
    columnas = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
    df = pd.DataFrame(res.get("result", {}).get("data_array", []) or [], columns=columnas)
    if not df.empty:
        df.insert(0, "rank", range(1, len(df) + 1))
        df["documento"] = df["path"].str.split("/").str[-1]
    return df

# COMMAND ----------

# Consultas de dos familias: conceptuales (significado) y de términos exactos (siglas, modelos, códigos).
consultas = {
    "conceptual_1": "Which phone survives being underwater?",
    "conceptual_2": "How do I keep my personal data safe on the device?",
    "exacta_1": "6G broadband",
    "exacta_2": "Helios G1M",
    "mixta_1": "Which products have IP69?",
}
TIPOS = ["semantic", "FULL_TEXT", "hybrid"]

filas = []
resultados = {}
for nombre, q in consultas.items():
    for tipo in TIPOS:
        df = buscar(q, tipo, k=5)
        resultados[(nombre, tipo)] = df
        top1 = df.iloc[0] if not df.empty else None
        filas.append((nombre, q, tipo,
                      top1["documento"] if top1 is not None else None,
                      round(float(top1["score"]), 4) if top1 is not None else None,
                      top1["chunk"][:100] if top1 is not None else None))

display(spark.createDataFrame(filas, ["familia", "consulta", "tipo", "doc_top1", "score_top1", "vista_previa_top1"]))

# COMMAND ----------

# Solapamiento de ids en el top-5 entre modos: ¿qué tan distintos son sus resultados?
solap = []
for nombre in consultas:
    ids = {t: set(resultados[(nombre, t)]["id"].astype(int)) for t in TIPOS}
    solap.append((nombre,
                  len(ids["semantic"] & ids["FULL_TEXT"]),
                  len(ids["semantic"] & ids["hybrid"]),
                  len(ids["FULL_TEXT"] & ids["hybrid"])))
display(spark.createDataFrame(solap, ["consulta", "sem∩fulltext", "sem∩hybrid", "fulltext∩hybrid"]))

# COMMAND ----------

# ¿El top-3 contiene el término exacto de la consulta? (proxy rápido de acierto para consultas exactas)
for nombre, termino in [("exacta_1", "6G"), ("exacta_2", "Helios"), ("mixta_1", "IP69")]:
    print(f"\n▶ {consultas[nombre]}  (término: {termino})")
    for tipo in TIPOS:
        top3 = resultados[(nombre, tipo)].head(3)
        aciertos = top3["chunk"].str.contains(termino, case=False).sum()
        print(f"   {tipo:<10} → {aciertos}/3 chunks contienen '{termino}'")

# COMMAND ----------

# Ilustración de Reciprocal Rank Fusion (RRF) con las listas semantic y FULL_TEXT de una consulta.
# Nota: es una simulación didáctica; la fusión interna de Vector Search puede diferir en detalles.
def rrf(listas, k=60):
    puntajes = {}
    for lista in listas:
        for pos, doc_id in enumerate(lista, start=1):
            puntajes[doc_id] = puntajes.get(doc_id, 0) + 1 / (k + pos)
    return sorted(puntajes.items(), key=lambda x: x[1], reverse=True)

sem = resultados[("mixta_1", "semantic")]["id"].astype(int).tolist()
ft = resultados[("mixta_1", "FULL_TEXT")]["id"].astype(int).tolist()
hyb = resultados[("mixta_1", "hybrid")]["id"].astype(int).tolist()
print("Semántica :", sem)
print("Full-text :", ft)
print("RRF manual:", [d for d, _ in rrf([sem, ft])][:5])
print("Hybrid VS :", hyb)

# COMMAND ----------

# Filtros por metadata: excluimos un documento construyendo la ruta dinámicamente (sin catálogo fijo).
paths = [r["path"] for r in spark.read.table("gold_docs_chunked").select("path").distinct().collect()]
manual = next((p for p in paths if "User_Manual" in p), paths[0])
print("Documento excluido:", manual)

q = "How to replace the battery of dPhone D1?"
display(buscar(q, k=3)[["rank", "documento", "score"]])
display(buscar(q, k=3, filtros={"path NOT": manual})[["rank", "documento", "score"]])

# Filtro de inclusión: solo buscar dentro de ese documento.
display(buscar(q, k=3, filtros={"path": manual})[["rank", "documento", "score"]])

# COMMAND ----------

# Salida esperada:
# - Tabla de 15 filas (5 consultas × 3 tipos) con el documento y score del top-1.
#   Los scores NO son comparables entre tipos (escalas distintas).
# - Conceptuales: semantic e hybrid aciertan; FULL_TEXT puede fallar por falta de palabras en común.
# - Exactas ("6G", "Helios G1M"): FULL_TEXT e hybrid tienen más chunks con el término en el top-3.
# - Solapamientos: hybrid comparte resultados con ambos modos.
# - RRF manual: un orden parecido (no necesariamente idéntico) al hybrid del índice.
# - Filtros: con "path NOT" el documento excluido desaparece; con "path" solo aparece ese documento.
