# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (STARTER) — 5 Queries × 3 Modos de Búsqueda
# MAGIC **Objetivo:** comparar cualitativa y cuantitativamente `semantic`, `FULL_TEXT` e `hybrid` en 5 consultas típicas de usuario.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** fortalezas y debilidades de la búsqueda vectorial pura, léxica (BM25) e híbrida.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` creado y ONLINE. Completa los `# TODO`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import pandas as pd
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)
vs_endpoint_name = "vs_endpoint"
vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"
vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# TODO 1: define 5 queries representativas según las categorías requeridas
QUERIES = [
    {
        "id": "Q1",
        "categoria": "conceptual_paraphrase",
        "query": "Can the smartphone survive immersion in water?"
    },
    {
        "id": "Q2",
        "categoria": "conceptual_os",
        "query": "Is it possible to execute both iPhone and Android applications?"
    },
    {
        "id": "Q3",
        "categoria": "termino_exacto",
        "query": "Helios G1M"  # TODO: o prueba otra sigla como "6G Ready"
    },
    {
        "id": "Q4",
        "categoria": "especificacion_tecnica",
        "query": "What is the battery capacity and charging speed?"
    },
    {
        "id": "Q5",
        "categoria": "fuera_de_dominio",
        "query": "Where can I buy tickets for the Taylor Swift concert in Bogota?"
    }
]

# COMMAND ----------

# TODO 2: completa la función de búsqueda para los 3 modos
def ejecutar_busqueda(query_text, query_type="semantic", k=3):
    """
    Ejecuta similarity_search en el índice vectorial.
    query_type puede ser 'semantic', 'FULL_TEXT', o 'hybrid'
    """
    params = {
        "query_text": query_text,
        "columns": ["id", "path", "chunk"],
        "num_results": k
    }
    
    # TODO: Si query_type NO es 'semantic', agrega params["query_type"] = query_type
    # (mientras no lo completes, los tres modos devolverán resultados de búsqueda semántica).
    pass

    try:
        resp = vs_index.similarity_search(**params)
        cols = [c["name"] for c in resp.get("manifest", {}).get("columns", [])]
        filas = resp.get("result", {}).get("data_array", []) or []
        df = pd.DataFrame(filas, columns=cols)
        return df
    except Exception as e:
        print(f"Error ejecutando {query_type} para '{query_text}': {e}")
        return pd.DataFrame()

# COMMAND ----------

# TODO 3: itera sobre las consultas y los 3 modos ('semantic', 'FULL_TEXT', 'hybrid')
MODOS = ["semantic", "FULL_TEXT", "hybrid"]
matriz_resultados = []

for item in QUERIES:
    q_id = item["id"]
    q_cat = item["categoria"]
    q_text = item["query"]
    
    for modo in MODOS:
        df_res = ejecutar_busqueda(q_text, query_type=modo, k=3)
        
        if not df_res.empty:
            top1 = df_res.iloc[0]
            doc_name = top1["path"].split("/")[-1] if "path" in top1 else "N/A"
            score = float(top1["score"]) if "score" in top1 else 0.0
            preview = str(top1.get("chunk", ""))[:120].replace("\n", " ")
        else:
            doc_name, score, preview = "SIN_RESULTADOS", 0.0, ""
            
        matriz_resultados.append({
            "query_id": q_id,
            "categoria": q_cat,
            "query_text": q_text,
            "modo": modo,
            "top1_doc": doc_name,
            "top1_score": round(score, 4),
            "top1_preview": preview
        })

df_comparativo = pd.DataFrame(matriz_resultados)
display(df_comparativo)

# COMMAND ----------

# TODO 4: Evalúa si el top1 es relevante para cada query y calcula top1_accuracy por modo
# Agrega una columna indicando si el resultado realmente responde o aborda la query
# Pista: para Q5 (fuera de dominio), ningún resultado debería considerarse relevante (False).

# df_comparativo["es_relevante"] = [...]
# accuracy_por_modo = df_comparativo.groupby("modo")["es_relevante"].mean()
# print(accuracy_por_modo)

# COMMAND ----------

# TODO 5: Veredicto y recomendación de negocio
VEREDICTO = """
TODO: Escribe aquí 4-6 líneas recomendando qué modo de búsqueda debe consumir el Agente en la Clase 7.
Menciona cómo se comporta en términos técnicos (siglas/códigos) vs preguntas abiertas y qué pasa con consultas fuera de dominio.
"""
print(VEREDICTO)

# COMMAND ----------

# Salida esperada:
# - Matriz con 15 filas (5 queries x 3 modos).
# - Diferencias claras: FULL_TEXT brilla en Q3 (Helios G1M), semantic brilla en Q1/Q2 (paráfrasis).
# - hybrid combina ambas ventajas.
# - En Q5, los scores son significativamente más bajos o irrelevantes, demostrando la necesidad de umbrales o LLM judge.
