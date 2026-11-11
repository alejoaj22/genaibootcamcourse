# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 3 — Comparar splitters (fixed, recursive, token, document-aware y semántico)
# MAGIC **Objetivo:** aplicar cinco estrategias de chunking al mismo texto y comparar dónde cortan.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** estrategias de chunking — tamaño fijo, recursiva, basada en tokens, consciente del documento (Markdown) y semántica (cortes por cambio de similitud entre oraciones).
# MAGIC **Requisitos previos:** acceso al endpoint `databricks-gte-large-en` (solo para la estrategia semántica); `Includes/Lab_Setup`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade langchain_text_splitters tiktoken

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# Texto de prueba: ficha técnica del dPhone D1 (misma que el notebook 1.1C, con caracteres normalizados).
# Viene en una sola línea, tal como la produce ai_parse_document con marcadores "== page ==".
doc = (
    "DerarTel MOBILE TECHNOLOGY ## Product Specifications: dPhone D1 Status: Internal/Confidential "
    "### 1. Executive Summary The dPhone D1 is engineered to redefine the premium smartphone market by bridging "
    "the gap between high-performance computing and minimalist design. The D1 prioritizes Open Ecosystem "
    "integration, Adaptive Hardware, and Quantum-Safe Encryption. **Figure 1.** dPhone D1 product image "
    "showcasing the full design and display features == page == ### 2. Key Value Propositions The Fluid OS: "
    "A proprietary operating system built on a microkernel architecture that allows for 100% app compatibility "
    "with both Android and iOS environments. Sustainability: Built with a 93% recycled titanium chassis and a "
    "user-replaceable battery. | Feature | Specification | |---------|---------------| | Processor | Helios G1M "
    "Neural Chip (5nm architecture) | | Display | 6.5\" Super-AMOLED Infinity Display | | Storage | 512GB | "
    "| Refresh Rate | Dynamic 1Hz to 144Hz | | Battery | 5,500 mAh with 60W Rapid Charging | | Security | "
    "Biometric Vein-Scan (Under-display) | | Connectivity | 6G Ready & Satellite SOS | ### 3. Environmental "
    "Resistance To ensure maximum durability, the dPhone D1 features industry-leading waterproof protection: "
    "IP69 Rating: The highest rating for ingress protection, certifying the dPhone D1 is completely dust-tight "
    "and capable of withstanding high-pressure, high-temperature water jets. Hydro-Guard Internal Seal: "
    "Proprietary nanocoating protects internal components. == page == ### 4. Performance Benchmarks (Scale 0-100) "
    "AI Processing: The dPhone D1 scores a 98, leveraging the Helios G1 Neural Chip, compared to the competitor's "
    "85. Battery Life: Optimized power management gives the D1 a score of 95, outlasting the 88 of the rival. "
    "Thermal Efficiency: With its advanced liquid-cooling and titanium chassis, the D1 maintains 92% efficiency "
    "under load. Signal Stability: The integrated 6G-ready modem provides a 94 score for connection reliability "
    "in weak signal areas. - End of Document -"
)
print("Longitud:", len(doc), "caracteres")

# COMMAND ----------

# Utilidad para resumir cada estrategia de forma homogénea.
resumen = []

def registrar(nombre, chunks):
    chunks = [c if isinstance(c, str) else c.page_content for c in chunks]
    chunks = [c for c in chunks if c.strip()]
    largos = [len(c) for c in chunks]
    resumen.append((nombre, len(chunks), min(largos), max(largos), round(sum(largos) / len(largos), 1)))
    print(f"\n===== {nombre}: {len(chunks)} chunks =====")
    for i, c in enumerate(chunks):
        print(f"[{i}] ({len(c)} car.) inicio: «{c[:70]}» … fin: «{c[-50:]}»")
    return chunks

# COMMAND ----------

# 1) Tamaño fijo: corta cada 500 caracteres sin respetar palabras ni frases.
from langchain_text_splitters import CharacterTextSplitter

fixed = CharacterTextSplitter(separator="", chunk_size=500, chunk_overlap=50)
chunks_fixed = registrar("1. Fixed-size (500/50)", fixed.split_text(doc))

# COMMAND ----------

# 2) Recursiva: la del notebook 1.1C, con chunk_size reducido a 500 para que se note el efecto.
from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n== page ==\n", "== page ==", "\n\n", "\n", " ", ""],
)
chunks_recursive = registrar("2. Recursive (500/50)", recursive.split_text(doc))

# COMMAND ----------

# 3) Basada en tokens: garantiza un máximo de tokens (útil si el modelo de embeddings tiene límite estricto).
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=120, chunk_overlap=12,
    separators=["== page ==", "\n\n", "\n", " ", ""],
)
chunks_token = registrar("3. Token-based (120 tokens)", token_splitter.split_text(doc))

# COMMAND ----------

# 4) Consciente del documento: usa los encabezados Markdown.
# Truco: el texto viene en una sola línea, así que insertamos saltos de línea antes de cada encabezado
# y convertimos "== page ==" en salto de párrafo.
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter

doc_md = doc.replace("== page ==", "\n\n")
doc_md = re.sub(r"\s*(#{1,6} )", r"\n\1", doc_md)

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
docs_md = md_splitter.split_text(doc_md)
chunks_md = registrar("4. Document-aware (Markdown)", docs_md)

# La metadata del encabezado viaja con cada chunk: ideal para filtros y citas.
for d in docs_md:
    print(d.metadata)

# COMMAND ----------

# 5) Semántica (versión simple): embeddings por oración y corte cuando la similitud con la oración previa cae.
import numpy as np
from databricks.sdk import WorkspaceClient

oraciones = [o.strip() for o in re.split(r"(?<=[.!?])\s+|==\s*page\s*==|(?=### )", doc) if o and o.strip()]
print("Oraciones:", len(oraciones))

def embeber(textos, tamano_lote=16):
    """Llama al endpoint de embeddings gestionado por lotes y devuelve una matriz numpy (n, 1024)."""
    w = WorkspaceClient()
    vectores = []
    for i in range(0, len(textos), tamano_lote):
        resp = w.serving_endpoints.query(name="databricks-gte-large-en", input=textos[i:i + tamano_lote])
        vectores.extend(d.embedding for d in resp.data)
    return np.array(vectores)

try:
    E = embeber(oraciones)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = (E[:-1] * E[1:]).sum(axis=1)                 # coseno entre oraciones consecutivas
    umbral = np.percentile(sim, 25)                     # cortamos en el 25 % de similitudes más bajas
    chunks_sem, actual = [], [oraciones[0]]
    for i, s in enumerate(sim):
        if s < umbral:
            chunks_sem.append(" ".join(actual))
            actual = []
        actual.append(oraciones[i + 1])
    chunks_sem.append(" ".join(actual))
    print(f"Umbral de corte (percentil 25): {umbral:.3f}")
    chunks_semantic = registrar("5. Semantic (percentil 25)", chunks_sem)
except Exception as e:
    print("No se pudo usar el endpoint de embeddings; se omite la estrategia semántica.", e)

# COMMAND ----------

# Tabla comparativa final.
display(spark.createDataFrame(resumen, ["estrategia", "n_chunks", "min_len", "max_len", "avg_len"]))

# COMMAND ----------

# Prueba de integridad: ¿qué estrategias mantienen "IP69" junto a "dPhone D1" en el mismo chunk?
for nombre, chunks in [("fixed", chunks_fixed), ("recursive", chunks_recursive),
                       ("token", chunks_token), ("markdown", chunks_md)]:
    textos = [c if isinstance(c, str) else c.page_content for c in chunks]
    ok = any("IP69" in t and "dPhone D1" in t for t in textos)
    print(f"{nombre:<10} → IP69 junto a 'dPhone D1': {'SÍ' if ok else 'NO'}")

# COMMAND ----------

# Salida esperada:
# - Fixed-size: cortes a mitad de palabra (p. ej. "...rec" | "ycled") y filas de tabla partidas.
# - Recursive: cortes en "== page ==" o en espacios, nunca a mitad de palabra.
# - Token-based: chunks de ≤120 tokens (≈400-500 caracteres).
# - Markdown: un chunk por sección ("### 1. Executive Summary", "### 2. Key Value Propositions", ...)
#   con metadata {'h2': ..., 'h3': ...}.
# - Semantic: número variable de chunks con tamaños desiguales; cortes donde cambia el tema.
# - Tabla comparativa con 4 o 5 filas y la prueba de integridad de "IP69".
