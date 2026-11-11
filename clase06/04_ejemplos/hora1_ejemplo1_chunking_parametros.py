# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 1 — Chunking con parámetros diferentes
# MAGIC **Objetivo:** observar cómo cambian la cantidad, el tamaño y el solapamiento de los chunks al variar `chunk_size` y `chunk_overlap`.
# MAGIC **Tiempo estimado:** 8 minutos
# MAGIC **Concepto que ilustra:** trade-offs de `chunk_size`/`chunk_overlap` en `RecursiveCharacterTextSplitter` y el "factor de duplicación" que introduce el overlap.
# MAGIC **Requisitos previos:** tabla `silver_docs_parsed` (Clase 5) en el esquema `genai_course`; carpeta `clase06/` importada junto a `Includes/`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade langchain_text_splitters

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# Leemos la capa Silver y escogemos el documento con más texto: es el que mejor muestra las diferencias.
from pyspark.sql import functions as F

silver_df = spark.read.table("silver_docs_parsed")

fila = (silver_df
        .where(F.col("plain_text").isNotNull())
        .withColumn("n_caracteres", F.length("plain_text"))
        .orderBy(F.desc("n_caracteres"))
        .first())

texto = fila["plain_text"]
print("Documento elegido:", fila["path"].split("/")[-1])
print("Longitud del texto (caracteres):", fila["n_caracteres"])
print("Tokens aproximados (caracteres / 4):", fila["n_caracteres"] // 4)

# COMMAND ----------

# Mismos separadores jerárquicos que el notebook 1.1C.
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARADORES = ["\n== page ==\n", "== page ==", "\n\n", "\n", " ", ""]

# Cinco configuraciones: de muy pequeña a muy grande.
CONFIGURACIONES = [
    (300, 0),
    (500, 50),
    (1000, 100),
    (1500, 150),   # la configuración del notebook oficial
    (3000, 300),
]

def trocear(texto, chunk_size, chunk_overlap):
    """Devuelve la lista de chunks no vacíos para una configuración dada."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES,
    )
    return [c for c in splitter.split_text(texto) if c.strip()]

resultados = []
chunks_por_config = {}
for size, overlap in CONFIGURACIONES:
    chunks = trocear(texto, size, overlap)
    chunks_por_config[(size, overlap)] = chunks
    largos = [len(c) for c in chunks]
    resultados.append((
        size, overlap, len(chunks),
        min(largos), max(largos), round(sum(largos) / len(largos), 1),
        # Factor de duplicación: cuánto texto total almacenamos respecto al original.
        round(sum(largos) / len(texto), 3),
    ))

df_resultados = spark.createDataFrame(
    resultados,
    ["chunk_size", "chunk_overlap", "n_chunks", "min_len", "max_len", "avg_len", "factor_duplicacion"],
)
display(df_resultados)

# COMMAND ----------

# Comparamos visualmente el primer chunk de la configuración más pequeña y de la oficial.
for config in [(300, 0), (1500, 150)]:
    primer = chunks_por_config[config][0]
    print(f"===== Configuración {config} · primer chunk ({len(primer)} caracteres) =====")
    print(primer[:800])
    print()

# COMMAND ----------

# Demostración del overlap: buscamos el texto compartido entre chunks consecutivos.
def solapamiento(a, b, maximo=400):
    """Longitud del sufijo más largo de 'a' que también es prefijo de 'b'."""
    for n in range(min(len(a), len(b), maximo), 0, -1):
        if a[-n:] == b[:n]:
            return a[-n:]
    return ""

chunks = chunks_por_config[(500, 50)]
print(f"Configuración (500, 50): {len(chunks)} chunks\n")
for i in range(min(3, len(chunks) - 1)):
    comun = solapamiento(chunks[i], chunks[i + 1])
    print(f"Chunk {i} → chunk {i+1}: {len(comun)} caracteres compartidos")
    print(f"   Texto compartido: «{comun.strip()[:120]}»\n")

# COMMAND ----------

# Gráfico rápido: número de chunks vs tamaño promedio por configuración.
import matplotlib.pyplot as plt

pdf = df_resultados.toPandas()
etiquetas = [f"{s}/{o}" for s, o in zip(pdf.chunk_size, pdf.chunk_overlap)]

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(etiquetas, pdf.n_chunks, color="#4C78A8")
ax1.set_xlabel("chunk_size / chunk_overlap")
ax1.set_ylabel("Número de chunks", color="#4C78A8")
ax2 = ax1.twinx()
ax2.plot(etiquetas, pdf.avg_len, color="#F58518", marker="o")
ax2.set_ylabel("Longitud promedio (caracteres)", color="#F58518")
plt.title("Efecto de chunk_size y chunk_overlap")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Preguntas para discutir:
# 1) ¿Qué configuración deja hechos clave (p. ej. "IP69") junto al nombre del producto?
# 2) ¿Cuánto almacenamiento extra introduce el overlap (factor_duplicacion)?
# 3) Si el modelo de embeddings admitiera solo 512 tokens (~2000 caracteres), ¿qué configuraciones serían peligrosas?
print("Regla práctica: chunk_size mediano (800-1500) y overlap del 10-20 % como punto de partida; luego medir.")

# COMMAND ----------

# Salida esperada:
# - Nombre del documento más largo de Silver y su longitud en caracteres y tokens aproximados.
# - Tabla con 5 filas: n_chunks disminuye al crecer chunk_size; factor_duplicacion > 1 cuando hay overlap
#   (≈1.0 con overlap 0, ≈1.05-1.15 con overlap del 10 %).
# - El primer chunk de (300, 0) es un fragmento corto; el de (1500, 150) contiene una página o sección completa.
# - Para (500, 50): pares de chunks consecutivos con hasta ~50 caracteres compartidos.
# - Un gráfico de barras (n_chunks) con línea (longitud promedio).
