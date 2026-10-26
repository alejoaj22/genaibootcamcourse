# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 2: Multimodalidad Aplicada: Preview de `ai_parse_document()` y Extracción de Documentos No Estructurados
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Generar programáticamente en memoria un documento sintético (imagen/PDF con ficha técnica y tabla de especificaciones de dPhone D1 Pro), procesarlo mediante la función nativa de Databricks `ai_parse_document(content, map('version', '2.0'))` (con fallback de emulación para clústeres estándar), inspeccionar la jerarquía de elementos `document.elements` (texto, tablas HTML, figuras) y pasar el contenido estructurado a un LLM para extraer especificaciones técnicas.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Procesamiento multimodal de documentos escaneados, extracción automática de tablas preservando estructura tabular en HTML y conexión directa con la etapa Bronze -> Silver del pipeline RAG de la Clase 5.
# MAGIC * **Requisitos previos:** Clúster Databricks Runtime 17.1+ o Serverless (para ejecución nativa de `ai_parse_document`), o cualquier runtime estándar gracias a la arquitectura con fallback.

# COMMAND ----------

# MAGIC %pip install pillow reportlab

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración de Invocación y Helper Estándar

# COMMAND ----------

import io
import json
import re
import time
from typing import Dict, Any, List
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

MODELO_20B = "databricks-gpt-oss-20b"


def _esc(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace("'", "\\'")


def extraer_texto(salida) -> str:
    if salida is None:
        return ""
    s = str(salida).strip()
    if s.startswith("[") and '"type"' in s:
        try:
            bloques = json.loads(s)
            textos = [b.get("text", "") for b in bloques if isinstance(b, dict) and b.get("type") == "text"]
            if textos:
                return "\n".join(textos).strip()
        except Exception:
            pass
    return s


def llamar_llm(prompt: str, modelo: str = MODELO_20B, temperature: float = 0.0,
               max_tokens: int = 1000) -> dict:
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{_esc(prompt)}',
        modelParameters => named_struct('temperature', CAST({float(temperature)} AS DOUBLE),
                                        'max_tokens', CAST({int(max_tokens)} AS INT))
    ) AS respuesta
    """
    t0 = time.time()
    try:
        salida = spark.sql(sql).collect()[0]["respuesta"]
    except Exception as e:
        salida = spark.sql(f"SELECT ai_query('{modelo}', '{_esc(prompt)}') AS respuesta").collect()[0]["respuesta"]
    return {
        "texto": extraer_texto(salida),
        "latencia_s": round(time.time() - t0, 2)
    }


print("Helper de inferencia configurado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generación en Memoria del Documento Técnico Escaneado (PNG)
# MAGIC Creamos una imagen sintética que simula una ficha técnica escaneada con encabezado, descripción y tabla de especificaciones técnicas del **dPhone D1 Pro**.

# COMMAND ----------

def generar_imagen_ficha_dphone() -> bytes:
    ancho, alto = 850, 600
    img = Image.new("RGB", (ancho, alto), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Título y metadata
    draw.text((40, 30), "DERARTEL MOBILE TECHNOLOGY - ESPECIFICACIONES TÉCNICAS", fill=(15, 23, 42))
    draw.text((40, 55), "Producto: dPhone D1 Pro | Estado: Confidencial / Preventa", fill=(71, 85, 105))
    draw.line([(40, 80), (810, 80)], fill=(203, 213, 225), width=2)

    # Texto descriptivo
    parrafo = (
        "El dPhone D1 Pro representa el buque insignia de DerarTel para 2028. "
        "Construido en chasis de titanio con nanorecubrimiento Hydro-Guard IP69, "
        "integra la revolucionaria camara Halo de 200MP y procesador Helios G1M Pro."
    )
    draw.text((40, 100), parrafo, fill=(30, 41, 59))

    # Dibujar Tabla de Especificaciones
    y_inicio = 160
    filas_tabla = [
        ("Componente", "Especificacion Oficial", "Detalle"),
        ("Procesador", "Helios G1M Pro (5nm)", "NPU de 98 TOPS integrada"),
        ("Pantalla", "6.8\" Super-AMOLED", "Refresco dinamico 1Hz-144Hz"),
        ("Almacenamiento", "1TB UFS 4.1", "Expandible con ranura sellada"),
        ("Bateria y Carga", "6,000 mAh", "Carga Ultra Rapida 80W"),
        ("Camara Principal", "Halo 200MP", "Sensor periscopico 10x optico"),
        ("Resistencia", "IP69 + Hydro-Guard", "Sumergible alta presion"),
        ("Precio de Venta", "1,299 USD", "Garantia 24 meses DerarTel")
    ]

    for idx, (c1, c2, c3) in enumerate(filas_tabla):
        y = y_inicio + (idx * 35)
        color_fondo = (241, 245, 249) if idx == 0 else (255, 255, 255)
        color_texto = (15, 23, 42) if idx == 0 else (51, 65, 85)
        draw.rectangle([(40, y), (810, y + 30)], fill=color_fondo, outline=(203, 213, 225))
        draw.text((50, y + 7), c1, fill=color_texto)
        draw.text((250, y + 7), c2, fill=color_texto)
        draw.text((550, y + 7), c3, fill=color_texto)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


bytes_png_documento = generar_imagen_ficha_dphone()
print(f"Documento sintético generado en memoria: {len(bytes_png_documento)} bytes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Procesamiento Multimodal con `ai_parse_document()`
# MAGIC Registramos el documento binario en un DataFrame de Spark e invocamos `ai_parse_document(content, map('version', '2.0'))`. Si el clúster actual no cuenta con DBR 17.1+, se activa la simulación estructurada que replica exactamente la salida en formato Spark Variant / JSON jerárquico.

# COMMAND ----------

# Creación de DataFrame binario simulando lectura de volumen
df_binario = spark.createDataFrame(
    [("ficha_dphone_d1_pro.png", bytearray(bytes_png_documento))],
    ["path", "content"]
)
df_binario.createOrReplaceTempView("documentos_bronze_preview")

# Estructura canónica de fallback según especificación Databricks ai_parse_document v2.0
FALLBACK_PARSE_RESULT = {
    "document": {
        "pages": [{"id": 0, "image_uri": "dbfs:/Volumes/dataset/imgs/page_0.png"}],
        "elements": [
            {
                "id": 0,
                "type": "title",
                "content": "DERARTEL MOBILE TECHNOLOGY - ESPECIFICACIONES TÉCNICAS",
                "confidence": 0.99
            },
            {
                "id": 1,
                "type": "text",
                "content": "Producto: dPhone D1 Pro | Estado: Confidencial / Preventa. El dPhone D1 Pro representa el buque insignia de DerarTel para 2028. Construido en chasis de titanio con nanorecubrimiento Hydro-Guard IP69, integra la revolucionaria camara Halo de 200MP y procesador Helios G1M Pro.",
                "confidence": 0.98
            },
            {
                "id": 2,
                "type": "table",
                "content": "<table><tr><th>Componente</th><th>Especificacion Oficial</th><th>Detalle</th></tr><tr><td>Procesador</td><td>Helios G1M Pro (5nm)</td><td>NPU de 98 TOPS integrada</td></tr><tr><td>Pantalla</td><td>6.8\" Super-AMOLED</td><td>Refresco dinamico 1Hz-144Hz</td></tr><tr><td>Almacenamiento</td><td>1TB UFS 4.1</td><td>Expandible con ranura sellada</td></tr><tr><td>Bateria y Carga</td><td>6,000 mAh</td><td>Carga Ultra Rapida 80W</td></tr><tr><td>Camara Principal</td><td>Halo 200MP</td><td>Sensor periscopico 10x optico</td></tr><tr><td>Resistencia</td><td>IP69 + Hydro-Guard</td><td>Sumergible alta presion</td></tr><tr><td>Precio de Venta</td><td>1,299 USD</td><td>Garantia 24 meses DerarTel</td></tr></table>",
                "confidence": 0.97
            }
        ]
    },
    "error_status": [],
    "metadata": {"version": "2.0"}
}

try:
    sql_parse = """
    SELECT
        path,
        ai_parse_document(content, map('version', '2.0')) AS parsed_variant
    FROM documentos_bronze_preview
    """
    df_parsed = spark.sql(sql_parse)
    fila = df_parsed.collect()[0]
    parsed_json_str = fila["parsed_variant"]
    parsed_obj = json.loads(parsed_json_str) if isinstance(parsed_json_str, str) else parsed_json_str
    print("Éxito: ai_parse_document ejecutado nativamente en Databricks.")
except Exception as e:
    print(f"[Aviso Informativo]: ai_parse_document requiere DBR 17.1+ Serverless. Activando emulación canónica ({type(e).__name__}).")
    parsed_obj = FALLBACK_PARSE_RESULT

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspección de la Jerarquía de Elementos Extraídos (`elements`)
# MAGIC En la versión 2.0, Databricks descompone el documento en elementos clasificados (`text`, `title`, `table`, `figure`). Las tablas se extraen directamente como HTML estructurado, lo que evita desalineaciones de columnas típicas del OCR clásico.

# COMMAND ----------

elementos = parsed_obj.get("document", {}).get("elements", [])
resumen_elementos = [
    {
        "ID": el.get("id"),
        "Tipo": el.get("type"),
        "Confianza": el.get("confidence"),
        "Contenido_Snippet": el.get("content", "")[:90] + "..."
    }
    for el in elementos
]

display(spark.createDataFrame(pd.DataFrame(resumen_elementos)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Extracción y Limpieza de la Tabla HTML con Pandas y LLM
# MAGIC Tomamos el elemento clasificado como `table` y lo convertimos en un DataFrame tabular de Spark, conectando este paso con la transformación Silver del pipeline RAG.

# COMMAND ----------

# Extraer el elemento tipo tabla
elemento_tabla = next((el for el in elementos if el.get("type") == "table"), None)
if elemento_tabla:
    html_tabla = elemento_tabla["content"]
    try:
        dfs = pd.read_html(io.StringIO(html_tabla))
        df_especificaciones = dfs[0]
        print("Tabla HTML deserializada exitosamente con Pandas:")
        display(spark.createDataFrame(df_especificaciones))
    except Exception:
        print("HTML crudo de la tabla extraída:")
        print(html_tabla)
        
    # Extracción final con LLM a esquema JSON de especificaciones
    prompt_extraccion_specs = f"""
<instructions>
Eres un Ingeniero de Datos en DerarTel.
A partir de la siguiente tabla técnica HTML extraída por ai_parse_document(), genera un JSON con las especificaciones clave.
</instructions>

<html_table>
{html_tabla}
</html_table>

<output_schema>
Objeto JSON con: modelo, procesador, bateria_mah (int), precio_usd (int), camara_principal.
</output_schema>
"""
    resultado_llm = llamar_llm(prompt_extraccion_specs, modelo=MODELO_20B, temperature=0.0)
    print("=== ESPECIFICACIONES TÉCNICAS ESTRUCTURADAS (JSON) ===")
    print(resultado_llm["texto"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Conexión con la Clase 5: De `ai_parse_document` a Vector Search y RAG
# MAGIC En la Clase 5 profundizaremos en el flujo completo de producción:
# MAGIC 1. **Bronze:** Almacenar manuales de usuario y PDFs en Volumes de Unity Catalog (`read_files(..., format => 'binaryFile')`).
# MAGIC 2. **Silver:** Aplicar `ai_parse_document(content, map('version', '2.0'))` para extraer texto limpio y tablas HTML sin pérdida de columnas.
# MAGIC 3. **Gold:** Chunking contextual preservando tablas completas + generación de embeddings automáticos.
# MAGIC 4. **Inferencia:** Consulta en lenguaje natural resuelta mediante Databricks Vector Search + `ai_query`.

# COMMAND ----------

# Salida esperada:
# 1. Generación del documento sintético en memoria sin dependencias externas.
# 2. Ejecución o emulación canónica de ai_parse_document versión 2.0.
# 3. Identificación y aislamiento del elemento 'table' en formato HTML.
# 4. Deserialización tabular con Pandas y conversión final a especificaciones JSON mediante LLM.
