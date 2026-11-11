# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 3 — Explorar embeddings: dimensiones, coseno, mapa de calor y PCA
# MAGIC **Objetivo:** ver "por dentro" qué es un embedding de `databricks-gte-large-en` y cómo la similitud coseno ordena textos por significado.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** espacio vectorial, normalización, similitud coseno y bi-encoder (consulta y documentos se codifican por separado).
# MAGIC **Requisitos previos:** acceso al endpoint de Foundation Model APIs `databricks-gte-large-en`. No necesita el índice (útil mientras éste se crea). La última sección usa `gold_docs_chunked`.

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# Función para obtener embeddings del endpoint gestionado (databricks-sdk viene preinstalado).
import numpy as np
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
MODELO = "databricks-gte-large-en"

def embeber(textos, tamano_lote=16):
    """Devuelve una matriz numpy (n_textos, dimensiones). Envía lotes pequeños para respetar los límites del endpoint."""
    vectores = []
    for i in range(0, len(textos), tamano_lote):
        resp = w.serving_endpoints.query(name=MODELO, input=textos[i:i + tamano_lote])
        vectores.extend(d.embedding for d in resp.data)
    return np.array(vectores)

v = embeber(["The dPhone D1 has an IP69 rating."])
print("Dimensiones del embedding:", v.shape[1])
print("Primeros 8 valores:", np.round(v[0][:8], 4))
print("Norma L2:", round(float(np.linalg.norm(v[0])), 4))

# COMMAND ----------

# Frases de prueba agrupadas por tema (más una frase intrusa).
frases = [
    "The phone is completely waterproof and dust-tight.",          # agua
    "IP69 certification protects against high-pressure water jets.",  # agua
    "Can I use my smartphone in the rain?",                          # agua
    "The battery has 5,500 mAh with 60W rapid charging.",            # batería
    "How many hours does the battery last on a single charge?",      # batería
    "Fast charging fills the cell in under an hour.",                # batería
    "The Helios neural chip accelerates AI workloads.",              # procesador
    "On-device machine learning runs on a dedicated NPU.",           # procesador
    "Mix flour, eggs and sugar to bake a cake.",                     # intrusa
]
temas = ["agua"] * 3 + ["bateria"] * 3 + ["chip"] * 2 + ["cocina"]

E = embeber(frases)
E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
S = E_norm @ E_norm.T          # matriz de similitud coseno

# COMMAND ----------

# Mapa de calor de similitud coseno.
import matplotlib.pyplot as plt

etiquetas = [f"{t}:{f[:28]}…" for t, f in zip(temas, frases)]
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(S, cmap="viridis")
ax.set_xticks(range(len(frases)))
ax.set_yticks(range(len(frases)))
ax.set_xticklabels(range(len(frases)))
ax.set_yticklabels(etiquetas, fontsize=8)
for i in range(len(frases)):
    for j in range(len(frases)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
plt.colorbar(im, ax=ax, label="similitud coseno")
plt.title("Similitud coseno entre frases (gte-large-en)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Proyección a 2D con PCA (implementada con numpy SVD, sin dependencias extra).
X = E_norm - E_norm.mean(axis=0)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
P = X @ Vt[:2].T
varianza = (s[:2] ** 2) / (s ** 2).sum()

colores = {"agua": "#4C78A8", "bateria": "#F58518", "chip": "#54A24B", "cocina": "#E45756"}
plt.figure(figsize=(8, 6))
for (x, y), t, f in zip(P, temas, frases):
    plt.scatter(x, y, color=colores[t], s=80)
    plt.annotate(f[:30] + "…", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
plt.title(f"PCA 2D de embeddings (varianza explicada: {varianza.sum():.0%})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Búsqueda "a mano": embeber una consulta y rankear chunks reales de la tabla Gold por coseno.
# Esto es exactamente lo que hace Vector Search, pero con fuerza bruta en lugar de un índice ANN.
chunks_pdf = spark.read.table("gold_docs_chunked").select("id", "path", "chunk").limit(64).toPandas()

consulta = "Which phone survives high-pressure water jets?"
q = embeber([consulta])[0]
q = q / np.linalg.norm(q)

C = embeber(chunks_pdf["chunk"].tolist())
C = C / np.linalg.norm(C, axis=1, keepdims=True)
chunks_pdf["coseno"] = C @ q

top = chunks_pdf.sort_values("coseno", ascending=False).head(5)
top["documento"] = top["path"].str.split("/").str[-1]
top["vista_previa"] = top["chunk"].str.slice(0, 120)
display(top[["id", "documento", "coseno", "vista_previa"]])

# COMMAND ----------

# Salida esperada:
# - Dimensiones del embedding: 1024; norma L2 cercana a 1 (vectores normalizados o casi).
# - Mapa de calor: bloques claros (alta similitud) dentro de cada tema; la frase de cocina tiene similitudes bajas con todas.
# - PCA: las frases se agrupan por color (agua, batería, chip) y la frase de cocina queda alejada.
# - Búsqueda manual: el top-1 por coseno es un chunk que menciona IP69 / water jets, coherente con la
#   búsqueda semántica del índice.
