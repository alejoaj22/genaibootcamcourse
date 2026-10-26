# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejercicio (Starter): Pipeline Multimodal de Procesamiento de Órdenes de Compra dPhone
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Implementar un pipeline multimodal completo en Databricks: generar una orden de compra escaneada en imagen, parsearla con `ai_parse_document()`, extraer sus datos con `ai_query()` y `json_schema`, validar la coherencia matemática de totales y cargar los resultados a un DataFrame de Spark.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Procesamiento documental inteligente (IDP), decodificación restringida con JSON Schema, aserciones aritméticas de calidad y bridging Bronze -> Silver.
# MAGIC * **Requisitos previos:** Clúster o Serverless con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %pip install pillow

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración: Helper Estándar de Invocación

# COMMAND ----------

import io
import json
import re
import time
from typing import Dict, Any, List
import pandas as pd
from PIL import Image, ImageDraw

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
               max_tokens: int = 1000, response_format: str = None) -> dict:
    rf_clause = f", responseFormat => '{_esc(response_format)}'" if response_format else ""
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{_esc(prompt)}',
        modelParameters => named_struct('temperature', CAST({float(temperature)} AS DOUBLE),
                                        'max_tokens', CAST({int(max_tokens)} AS INT))
        {rf_clause}
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


def parsear_json_limpio(texto: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (texto or "").strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
    return None


print("Helper llamar_llm() configurado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. TODO 1: Generación de la Orden de Compra Sintética en Memoria (PNG)
# MAGIC
# MAGIC Construye la imagen con Pillow conteniendo los datos de una orden de compra para el distribuidor "TechAndes Mayorista":
# MAGIC - Orden: PO-2028-9941
# MAGIC - Ítems:
# MAGIC   1. dPhone D1 Lite (Cant: 10, Precio Unitario: 699, Subtotal: 6990)
# MAGIC   2. dPhone D1 (Cant: 5, Precio Unitario: 999, Subtotal: 4995)
# MAGIC   3. dPhone D1 Pro (Cant: 3, Precio Unitario: 1299, Subtotal: 3897)
# MAGIC   4. dPhone D1 Fold (Cant: 2, Precio Unitario: 1899, Subtotal: 3798)
# MAGIC - Total Orden: 19,680 USD

# COMMAND ----------

def generar_orden_compra_png() -> bytes:
    # TODO 1: Genera y dibuja la imagen de la orden de compra con Pillow
    ancho, alto = 800, 500
    img = Image.new("RGB", (ancho, alto), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Encabezado
    draw.text((40, 30), "ORDEN DE COMPRA COMERCIAL - DERARTEL DISTRIBUCION", fill=(15, 23, 42))
    draw.text((40, 55), "Distribuidor: TechAndes Mayorista | Orden: PO-2028-9941 | Fecha: 2028-04-15", fill=(71, 85, 105))
    draw.line([(40, 80), (760, 80)], fill=(203, 213, 225), width=2)
    
    # Dibujar filas de ítems
    filas = [
        ("Modelo", "Cantidad", "Precio Unit (USD)", "Subtotal (USD)"),
        ("dPhone D1 Lite", "10", "699", "6990"),
        ("dPhone D1", "5", "999", "4995"),
        ("dPhone D1 Pro", "3", "1299", "3897"),
        ("dPhone D1 Fold", "2", "1899", "3798"),
        ("TOTAL GENERAL", "", "", "19680")
    ]
    
    y = 110
    for idx, (col1, col2, col3, col4) in enumerate(filas):
        color_fondo = (241, 245, 249) if idx in (0, len(filas)-1) else (255, 255, 255)
        color_texto = (15, 23, 42) if idx in (0, len(filas)-1) else (51, 65, 85)
        draw.rectangle([(40, y), (760, y + 30)], fill=color_fondo, outline=(203, 213, 225))
        draw.text((50, y + 7), col1, fill=color_texto)
        draw.text((320, y + 7), col2, fill=color_texto)
        draw.text((480, y + 7), col3, fill=color_texto)
        draw.text((640, y + 7), col4, fill=color_texto)
        y += 35
        
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


bytes_orden = generar_orden_compra_png()
print(f"Orden de compra generada: {len(bytes_orden)} bytes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Procesamiento Documental con `ai_parse_document()` (o Fallback Emulado)

# COMMAND ----------

df_orden_raw = spark.createDataFrame([("po_2028_9941.png", bytearray(bytes_orden))], ["path", "content"])
df_orden_raw.createOrReplaceTempView("orden_raw_temp")

FALLBACK_PARSE_ORDEN = {
    "document": {
        "elements": [
            {
                "id": 0,
                "type": "title",
                "content": "ORDEN DE COMPRA COMERCIAL - DERARTEL DISTRIBUCION"
            },
            {
                "id": 1,
                "type": "text",
                "content": "Distribuidor: TechAndes Mayorista | Orden: PO-2028-9941 | Fecha: 2028-04-15"
            },
            {
                "id": 2,
                "type": "table",
                "content": "<table><tr><th>Modelo</th><th>Cantidad</th><th>Precio Unit (USD)</th><th>Subtotal (USD)</th></tr><tr><td>dPhone D1 Lite</td><td>10</td><td>699</td><td>6990</td></tr><tr><td>dPhone D1</td><td>5</td><td>999</td><td>4995</td></tr><tr><td>dPhone D1 Pro</td><td>3</td><td>1299</td><td>3897</td></tr><tr><td>dPhone D1 Fold</td><td>2</td><td>1899</td><td>3798</td></tr><tr><td>TOTAL GENERAL</td><td></td><td></td><td>19680</td></tr></table>"
            }
        ]
    }
}

try:
    sql_ai_parse = "SELECT ai_parse_document(content, map('version', '2.0')) AS parsed FROM orden_raw_temp"
    raw_res = spark.sql(sql_ai_parse).collect()[0]["parsed"]
    doc_parseado = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
except Exception:
    doc_parseado = FALLBACK_PARSE_ORDEN

elemento_tabla = next((el for el in doc_parseado["document"]["elements"] if el["type"] == "table"), None)
html_tabla_orden = elemento_tabla["content"] if elemento_tabla else ""
print("HTML de la tabla recuperado:")
print(html_tabla_orden)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. TODO 2 y TODO 3: JSON Schema Estricto y Extracción con `ai_query`

# COMMAND ----------

# TODO 2: Define el esquema JSON Schema para la orden de compra
SCHEMA_ORDEN = {
    # Completa el esquema con las propiedades requeridas
}

RESPONSE_FORMAT_ORDEN = None
# Pista: Construye el payload json_schema y serialízalo a string


# COMMAND ----------

# TODO 3: Construye el prompt e invoca llamar_llm con response_format
prompt_extraccion = f"""
<instructions>
Extrae los datos estructurados de la siguiente tabla HTML de orden de compra.
</instructions>

<html_table>
{html_tabla_orden}
</html_table>
"""

# Invoca el modelo aquí:
# resultado_orden = llamar_llm(prompt_extraccion, modelo=MODELO_20B, response_format=RESPONSE_FORMAT_ORDEN)

print("Pendiente: completa el TODO 2 y TODO 3 para extraer los datos de la orden.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. TODO 4: Validación Aritmética de Totales y Deserialización Spark

# COMMAND ----------

def validar_aritmetica_orden(datos_orden: Dict[str, Any]) -> Dict[str, Any]:
    # TODO 4: Verifica que cada subtotal == cantidad * precio_unitario
    # y que suma(subtotales) == total_orden
    if not datos_orden or "items" not in datos_orden:
        return {"valido": False, "motivo": "Datos incompletos"}
    
    # Implementa tus aserciones aquí
    return {"valido": True}


# COMMAND ----------

# Salida esperada:
# 1. La tabla de la orden de compra se procesa mediante ai_parse_document.
# 2. ai_query con Structured Outputs devuelve un JSON con la orden PO-2028-9941 y 4 ítems.
# 3. La función de validación comprueba que 10*699 + 5*999 + 3*1299 + 2*1899 == 19,680 USD.
