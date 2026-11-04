# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 1 — Esquema conceptual del ciclo de vida de RAG
# MAGIC **Objetivo:** visualizar las 6 etapas de RAG y ubicar qué cubre cada clase de TRANSFORMAR.
# MAGIC **Tiempo estimado:** 8 minutos
# MAGIC **Concepto que ilustra:** ciclo de vida RAG (Ingestión → Extracción → Chunking → Vectorización → Búsqueda → Síntesis) y capas Medallion.
# MAGIC **Requisitos previos:** clúster encendido. No requiere Lab_Setup ni datos (ejemplo 100 % conceptual).

# COMMAND ----------

# Definimos las etapas del ciclo de vida RAG como datos (una fila por etapa).
etapas = [
    (1, "Ingestión",     "Bronze", "Clase 5 (hoy)", "Leer archivos binarios y guardarlos con metadatos", "spark.read.format('binaryfile')"),
    (2, "Extracción",    "Silver", "Clase 5 (hoy)", "Parsear documentos a texto/estructura", "ai_parse_document(), ai_query()"),
    (3, "Chunking",      "Gold",   "Clase 6",       "Dividir en fragmentos con sentido", "Notebook 1.1C"),
    (4, "Vectorización", "Gold",   "Clase 6",       "Convertir fragmentos en embeddings", "Modelo de embeddings"),
    (5, "Búsqueda",      "Índice", "Clase 6-7",     "Recuperar los fragmentos más similares", "Vector Search"),
    (6, "Síntesis",      "Respuesta", "Clase 7",    "El LLM responde con el contexto recuperado", "ai_query() / agente"),
]
columnas = ["orden", "etapa", "capa", "clase", "descripcion", "herramienta"]

# Creamos un DataFrame de Spark para verlo como tabla.
df_ciclo = spark.createDataFrame(etapas, columnas)
display(df_ciclo)

# COMMAND ----------

# Resaltamos el alcance de HOY con un filtro simple.
df_hoy = df_ciclo.filter(df_ciclo.clase.contains("hoy"))
display(df_hoy)

# COMMAND ----------

# Generamos el diagrama Mermaid automáticamente a partir de los datos.
nodos = []
for orden, etapa, capa, clase, desc, herr in etapas:
    nodos.append(f'E{orden}["{orden}. {etapa}<br/>({capa})"]')
flechas = " --> ".join([f"E{o}" for o, *_ in etapas])
mermaid = "flowchart LR\n  " + "\n  ".join(nodos) + "\n  " + flechas

# Mostramos el diagrama como HTML con la librería Mermaid cargada desde CDN.
html = f"""
<pre class="mermaid">{mermaid}</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true }});
</script>
"""
print(mermaid)  # Siempre se imprime el texto por si el HTML no carga (red restringida)
displayHTML(html)

# COMMAND ----------

# Resumen textual: cuántas etapas cubre cada clase.
display(df_ciclo.groupBy("clase").count().orderBy("clase"))

# COMMAND ----------

# Salida esperada:
# - Una tabla con 6 filas (etapas) y 6 columnas.
# - Un filtro con 2 filas (Ingestión y Extracción) marcadas como "Clase 5 (hoy)".
# - El texto Mermaid impreso y, si hay acceso a internet, el diagrama renderizado.
# - Un conteo: Clase 5 (hoy)=2, Clase 6=3 (Chunking, Vectorización), Clase 6-7=1, Clase 7=1 (el orden puede variar).
