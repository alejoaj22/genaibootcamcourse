# Databricks notebook source
# MAGIC %md
# MAGIC # 🌉 Hora 2 — Ejemplo 3: Puente en Vivo con la Clase 1 (Tokens con `tiktoken` y Embeddings con `databricks-gte-large-en`)
# MAGIC
# MAGIC - **Objetivo:** Materializar con código real en Databricks los conceptos matemáticos de la Clase 1: (1) contar y descomponer tokens en español, inglés y SQL usando `tiktoken`, y (2) generar vectores de embeddings reales de **1.024 dimensiones** con el endpoint `databricks-gte-large-en` para calcular y visualizar la matriz de **Similitud Coseno** sobre preguntas del caso dPhone.
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Tokenización BPE, sobrecosto multilingüe del token, representación vectorial densa (embeddings) y similitud coseno ($\cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}$) como motor de búsqueda semántica (RAG).
# MAGIC - **Requisitos previos:** Cómputo conectado y endpoint `databricks-gte-large-en` disponible en Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalación de `tiktoken`, `databricks-sdk` y `openai`
# MAGIC
# MAGIC Instalamos `tiktoken` (el tokenizador BPE rápido en Rust/Python) junto con el SDK de Databricks y reiniciamos el intérprete de Python.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade tiktoken databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parte A — Tokenización en vivo: Español vs Inglés vs Código SQL
# MAGIC
# MAGIC En la Clase 1 vimos que el modelo no lee letras ni palabras completas, sino **tokens** (sub-palabras con un ID numérico), y que el español suele requerir más tokens que el inglés para expresar la misma idea.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `enc = tiktoken.get_encoding("o200k_base")`: Cargamos la codificación BPE moderna (también comparamos con `cl100k_base`).
# MAGIC - `token_ids = enc.encode(texto)`: Convierte el string en la lista de enteros (IDs de tokens).
# MAGIC - `piezas = [enc.decode([tid]) for tid in token_ids]`: Reconstruye el fragmento de texto exacto que representa cada ID para ver cómo se parten palabras como `"garantía"`, `"Medellín"` o `"dPhone"`.

# COMMAND ----------

import tiktoken
import pandas as pd

# Cargamos el tokenizador moderno o200k_base
enc = tiktoken.get_encoding("o200k_base")

pares_frases = [
    ("1. Pregunta completa (ES)", "ES", "¿Cuál es el dPhone más reciente, hay stock en Medellín y qué cubre la garantía si se moja?"),
    ("1. Pregunta completa (EN)", "EN", "What is the latest dPhone, is it in stock in Medellin and what does the warranty cover if it gets wet?"),
    ("2. Ticket Sarah (ES)",      "ES", "Hola, estoy interesada en comprar 3 unidades del último modelo de dPhone."),
    ("2. Ticket Sarah (EN)",      "EN", "Hello, I am interested in buying 3 of latest dPhone model."),
    ("3. Consulta SQL",           "SQL", "SELECT model, price, quantity FROM smartphones JOIN inventory_stock ON phone_id = item_id WHERE quantity >= 3;")
]

filas_tokens = []
for etiqueta, idioma, texto in pares_frases:
    ids = enc.encode(texto)
    piezas = [enc.decode([i]) for i in ids]
    filas_tokens.append({
        "Caso": etiqueta,
        "Idioma": idioma,
        "Caracteres": len(texto),
        "Palabras": len(texto.split()),
        "Num_Tokens": len(ids),
        "Tokens_por_Palabra": round(len(ids) / len(texto.split()), 2),
        "Primeros_8_Subtokens": " | ".join(repr(p) for p in piezas[:8])
    })

df_tokens = pd.DataFrame(filas_tokens)
display(df_tokens)

print("\n=== DESCOMPOSICIÓN SUB-PALABRA DE TÉRMINOS CLAVE DE DERARTEL ===")
for palabra in ["dPhone", "garantía", "warranty", "Medellín", "smartphones", "sobrecalentamiento"]:
    ids = enc.encode(palabra)
    piezas = [enc.decode([i]) for i in ids]
    print(f"  Palabra: {palabra:<20} -> {len(ids)} tokens -> IDs: {str(ids):<24} -> Piezas: {piezas}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parte B — Generación de Embeddings reales con `databricks-gte-large-en`
# MAGIC
# MAGIC Ahora invocamos el endpoint **`databricks-gte-large-en`** de Databricks Foundation Model APIs. Este es exactamente el mismo modelo de embeddings que usaremos en la Semana 6 cuando creemos el índice de **Databricks Vector Search** en el notebook `1.2- Creating Vector Search.ipynb`.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `resp = client.embeddings.create(model="databricks-gte-large-en", input=frases_dphone)`: Envía una lista de 5 textos al endpoint y recibe 5 vectores numéricos.
# MAGIC - `vectores = np.array([item.embedding for item in resp.data])`: Convierte la respuesta en una matriz `numpy` de tamaño `(5, 1024)` (5 frases $\times$ 1.024 dimensiones).

# COMMAND ----------

import numpy as np
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

ENDPOINT_EMBEDDINGS = "databricks-gte-large-en"

# Definimos 5 textos del dominio DerarTel / dPhone para comparar su cercanía semántica
casos_embeddings = [
    ("F1_Compra_A",   "How can I buy 3 units of the newest dPhone model?"),
    ("F2_Compra_B",   "I want to purchase the latest smartphone release from DerarTel."),
    ("F3_Specs_Fold", "Can you share the technical specifications and screen size of the dPhone D1 Fold?"),
    ("F4_Garantia",   "What does the user manual say about water damage and warranty repairs?"),
    ("F5_Pollo_Arroz","Traditional recipe for cooking roasted chicken with rice and vegetables.")
]

etiquetas = [c[0] for c in casos_embeddings]
textos = [c[1] for c in casos_embeddings]

# Llamamos en vivo al endpoint de embeddings en Databricks
respuesta_emb = client.embeddings.create(
    model=ENDPOINT_EMBEDDINGS,
    input=textos
)

# Extraemos los vectores en una matriz NumPy
matriz_vectores = np.array([item.embedding for item in respuesta_emb.data], dtype=np.float64)

print(f"✅ Embeddings generados exitosamente con '{ENDPOINT_EMBEDDINGS}'")
print(f"   Forma de la matriz (Frases x Dimensiones) : {matriz_vectores.shape}")
print(f"   Norma L2 del primer vector                : {np.linalg.norm(matriz_vectores[0]):.4f}")
print(f"   Primeros 8 números del vector F1_Compra_A : {np.round(matriz_vectores[0][:8], 4).tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cálculo y análisis de la Matriz de Similitud Coseno ($\cos(\theta)$)
# MAGIC
# MAGIC Recordemos la fórmula vista en la Clase 1:
# MAGIC $$\text{Similitud Coseno}(A, B) = \frac{A \cdot B}{\|A\| \, \|B\|}$$
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `normas = np.linalg.norm(matriz_vectores, axis=1, keepdims=True)`: Calcula la longitud ($\|v\|$) de cada uno de los 5 vectores de 1.024 dimensiones.
# MAGIC - `matriz_normalizada = matriz_vectores / normas`: Convierte cada vector en un vector unitario (longitud 1).
# MAGIC - `matriz_similitud = np.dot(matriz_normalizada, matriz_normalizada.T)`: El producto punto entre todos los pares de vectores unitarios nos da directamente la matriz $5 \times 5$ de similitud coseno.

# COMMAND ----------

# Normalizamos cada fila para que tenga norma unitaria y calculamos el producto punto cruzado
normas = np.linalg.norm(matriz_vectores, axis=1, keepdims=True)
matriz_normalizada = matriz_vectores / normas
matriz_similitud = np.round(np.dot(matriz_normalizada, matriz_normalizada.T), 4)

# Construimos un DataFrame legible con la matriz 5x5 de similitud coseno
df_similitud = pd.DataFrame(matriz_similitud, index=etiquetas, columns=etiquetas)
df_similitud.insert(0, "Texto_Original", textos)

print("=== MATRIZ DE SIMILITUD COSENO (1.024 DIMENSIONES) ===")
display(df_similitud)

# Destacamos los pares más reveladores para explicar por qué funciona la búsqueda semántica (RAG)
sim_compra = matriz_similitud[0, 1]
sim_compra_garantia = matriz_similitud[0, 3]
sim_compra_receta = matriz_similitud[0, 4]

print(f"🔎 Par Semántico Equivalente (F1_Compra_A vs F2_Compra_B)  : {sim_compra:.4f}  <-- ¡Alta similitud aunque usan palabras distintas (buy/newest vs purchase/latest)!")
print(f"🔎 Par Mismo Dominio Distinto Tema (F1_Compra vs F4_Garantía): {sim_compra_garantia:.4f}")
print(f"🔎 Par Fuera de Dominio (F1_Compra vs F5_Pollo_Arroz)      : {sim_compra_receta:.4f}  <-- ¡Baja similitud, el buscador vectorial lo descarta!")

# Salida esperada:
# - Tabla de tokens mostrando que la pregunta en español tiene más tokens (~24) que en inglés (~22), y cómo 'garantía' o 'sobrecalentamiento' se dividen en múltiples sub-tokens.
# - Forma de la matriz (Frases x Dimensiones): (5, 1024).
# - Matriz de similitud coseno 5x5 con diagonal principal en 1.0000:
#   * F1_Compra_A vs F2_Compra_B > 0.82 (muy cercanas semánticamente).
#   * F1_Compra_A vs F4_Garantia ~ 0.55 - 0.68 (ambas hablan de teléfonos, pero distinta intención).
#   * F1_Compra_A vs F5_Pollo_Arroz < 0.45 (temas completamente inconexos).
