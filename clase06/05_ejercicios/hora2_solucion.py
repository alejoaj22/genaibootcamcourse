# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (SOLUCIÓN) — 5 Queries × 3 Modos de Búsqueda
# MAGIC **Objetivo:** comparar cualitativa y cuantitativamente `semantic`, `FULL_TEXT` e `hybrid` en 5 consultas típicas de usuario.
# MAGIC **Tiempo estimado:** 25 minutos
# MAGIC **Concepto que ilustra:** fortalezas y debilidades de la búsqueda vectorial pura, léxica (BM25) e híbrida.
# MAGIC **Requisitos previos:** índice `dphone_vs_index` creado y ONLINE.

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

# 1) Definición de las 5 consultas de evaluación
QUERIES = [
    {
        "id": "Q1",
        "categoria": "conceptual_paraphrase",
        "query": "Can the smartphone survive immersion in water?",
        "keywords": ["IP69", "waterproof", "water"]
    },
    {
        "id": "Q2",
        "categoria": "conceptual_os",
        "query": "Is it possible to execute both iPhone and Android applications?",
        "keywords": ["Fluid", "iOS", "Android", "compatibility"]
    },
    {
        "id": "Q3",
        "categoria": "termino_exacto",
        "query": "Helios G1M",
        "keywords": ["Helios", "G1M", "Neural Chip"]
    },
    {
        "id": "Q4",
        "categoria": "especificacion_tecnica",
        "query": "What is the battery capacity and charging speed?",
        "keywords": ["5,500", "battery", "60W", "charging"]
    },
    {
        "id": "Q5",
        "categoria": "fuera_de_dominio",
        "query": "Where can I buy tickets for the Taylor Swift concert in Bogota?",
        "keywords": ["Taylor Swift", "concert", "Bogota"]
    }
]

# COMMAND ----------

# 2) Función de búsqueda parametrizada
def ejecutar_busqueda(query_text, query_type="semantic", k=3):
    params = {
        "query_text": query_text,
        "columns": ["id", "path", "chunk"],
        "num_results": k
    }
    if query_type != "semantic":
        params["query_type"] = query_type

    try:
        resp = vs_index.similarity_search(**params)
        cols = [c["name"] for c in resp.get("manifest", {}).get("columns", [])]
        filas = resp.get("result", {}).get("data_array", []) or []
        return pd.DataFrame(filas, columns=cols)
    except Exception as e:
        print(f"Error ejecutando {query_type} para '{query_text}': {e}")
        return pd.DataFrame()

# COMMAND ----------

# 3) Ejecución de la matriz comparativa
MODOS = ["semantic", "FULL_TEXT", "hybrid"]
matriz_resultados = []

for item in QUERIES:
    q_id = item["id"]
    q_cat = item["categoria"]
    q_text = item["query"]
    claves = item["keywords"]
    
    for modo in MODOS:
        df_res = ejecutar_busqueda(q_text, query_type=modo, k=3)
        
        if not df_res.empty:
            top1 = df_res.iloc[0]
            doc_name = top1["path"].split("/")[-1] if "path" in top1 else "N/A"
            score = float(top1["score"]) if "score" in top1 else 0.0
            chunk_text = str(top1.get("chunk", ""))
            preview = chunk_text[:120].replace("\n", " ")
            
            # Criterio objetivo de relevancia:
            # Si es Q5 (fuera de dominio), nunca es relevante
            if q_id == "Q5":
                es_relevante = False
            else:
                es_relevante = any(k.lower() in chunk_text.lower() for k in claves)
        else:
            doc_name, score, preview, es_relevante = "SIN_RESULTADOS", 0.0, "", False
            
        matriz_resultados.append({
            "query_id": q_id,
            "categoria": q_cat,
            "query_text": q_text,
            "modo": modo,
            "top1_doc": doc_name,
            "top1_score": round(score, 4),
            "top1_preview": preview,
            "es_relevante": es_relevante
        })

df_comparativo = pd.DataFrame(matriz_resultados)
display(df_comparativo)

# COMMAND ----------

# 4) Tasa de acierto cualitativa (top-1 accuracy)
accuracy_por_modo = df_comparativo.groupby("modo")["es_relevante"].agg(["count", "sum", "mean"]).reset_index()
accuracy_por_modo.columns = ["modo", "total_queries", "aciertos_top1", "top1_accuracy"]
display(accuracy_por_modo)

# COMMAND ----------

# 5) Análisis de comportamiento por categoría
pivot_res = df_comparativo.pivot(index="query_id", columns="modo", values="es_relevante")
display(pivot_res)

# COMMAND ----------

# 6) Veredicto técnico y de negocio
veredicto_txt = """
RECOMENDACIÓN FINAL:
Para el agente de la Clase 7, se debe configurar por defecto el modo 'hybrid'.
Justificación técnica:
1. En preguntas conceptuales y paráfrasis (Q1 y Q2), el modo 'semantic' e 'hybrid' aciertan al 100%, mientras que FULL_TEXT 
   puede fallar si el usuario no usa los términos léxicos exactos del documento.
2. En términos técnicos precisos y códigos (Q3: 'Helios G1M'), FULL_TEXT e hybrid garantizan que el documento con la coincidencia 
   exacta quede rankeado en primera posición.
3. Para la consulta fuera de dominio (Q5), los 3 métodos devuelven resultados (el retriever no tiene noción nativa de abstención), 
   pero el score híbrido y semántico cae drásticamente, lo cual permite filtrar por umbral o delegar al LLM la detección de 'información insuficiente'.
Conclusión: 'hybrid' entrega la mayor robustez frente a la diversidad de perfiles de usuario.
"""
print(veredicto_txt)

# COMMAND ----------

# Salida esperada:
# - df_comparativo con 15 registros evaluados.
# - Matriz pivotada mostrando que 'hybrid' logra el rendimiento más consistente (4/5 o 80%, dado que Q5 es no-relevante por diseño).
# - Claridad sobre por qué FULL_TEXT pierde en paráfrasis pero triunfa en siglas.
