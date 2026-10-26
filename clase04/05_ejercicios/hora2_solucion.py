# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejercicio (Solución): Pipeline Multimodal de Procesamiento de Órdenes de Compra dPhone
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Solución de referencia completa del pipeline multimodal en Databricks: generación de la orden sintética en imagen PNG, parseo con `ai_parse_document()`, extracción estructurada mediante `ai_query()` con `responseFormat` (`json_schema`), validación matemática estricta y consolidación en tabla de Spark SQL.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Procesamiento documental inteligente (IDP), decodificación restringida, auditoría determinista de coherencia aritmética y transformación Medallion Bronze -> Silver.
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b`.

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


print("Infraestructura lista.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generación de la Orden de Compra Sintética en Memoria (PNG)

# COMMAND ----------

def generar_orden_compra_png() -> bytes:
    ancho, alto = 820, 520
    img = Image.new("RGB", (ancho, alto), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Encabezado
    draw.text((40, 30), "ORDEN DE COMPRA COMERCIAL - DERARTEL DISTRIBUCION", fill=(15, 23, 42))
    draw.text((40, 55), "Distribuidor: TechAndes Mayorista | Orden: PO-2028-9941 | Fecha: 2028-04-15", fill=(71, 85, 105))
    draw.line([(40, 80), (780, 80)], fill=(203, 213, 225), width=2)

    # Filas de ítems
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
        color_fondo = (241, 245, 249) if idx in (0, len(filas) - 1) else (255, 255, 255)
        color_texto = (15, 23, 42) if idx in (0, len(filas) - 1) else (51, 65, 85)
        draw.rectangle([(40, y), (780, y + 30)], fill=color_fondo, outline=(203, 213, 225))
        draw.text((50, y + 7), col1, fill=color_texto)
        draw.text((320, y + 7), col2, fill=color_texto)
        draw.text((480, y + 7), col3, fill=color_texto)
        draw.text((640, y + 7), col4, fill=color_texto)
        y += 35

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


bytes_orden = generar_orden_compra_png()
print(f"Orden de compra binaria generada: {len(bytes_orden)} bytes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Procesamiento Documental: `ai_parse_document()` con Fallback Emulado

# COMMAND ----------

df_orden_raw = spark.createDataFrame([("po_2028_9941.png", bytearray(bytes_orden))], ["path", "content"])
df_orden_raw.createOrReplaceTempView("orden_raw_solucion_temp")

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
    sql_ai_parse = "SELECT ai_parse_document(content, map('version', '2.0')) AS parsed FROM orden_raw_solucion_temp"
    raw_res = spark.sql(sql_ai_parse).collect()[0]["parsed"]
    doc_parseado = json.loads(raw_res) if isinstance(raw_res, str) else raw_res
except Exception:
    doc_parseado = FALLBACK_PARSE_ORDEN

elemento_tabla = next((el for el in doc_parseado["document"]["elements"] if el["type"] == "table"), None)
html_tabla_orden = elemento_tabla["content"] if elemento_tabla else ""
print("Tabla HTML de la orden:")
print(html_tabla_orden)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extracción con Structured Outputs (`json_schema`)

# COMMAND ----------

SCHEMA_ORDEN = {
    "type": "object",
    "properties": {
        "numero_orden": {"type": "string"},
        "distribuidor": {"type": "string"},
        "fecha": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "modelo": {"type": "string"},
                    "cantidad": {"type": "integer"},
                    "precio_unitario_usd": {"type": "number"},
                    "subtotal_usd": {"type": "number"}
                },
                "required": ["modelo", "cantidad", "precio_unitario_usd", "subtotal_usd"],
                "additionalProperties": False
            }
        },
        "total_orden_usd": {"type": "number"}
    },
    "required": ["numero_orden", "distribuidor", "items", "total_orden_usd"],
    "additionalProperties": False
}

RESPONSE_FORMAT_ORDEN = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "orden_compra_dphone",
        "schema": SCHEMA_ORDEN,
        "strict": True
    }
})

prompt_extraccion_solucion = f"""
<instructions>
Eres un Analista Financiero de Facturación para DerarTel.
Extrae los metadatos y las partidas de productos de la siguiente tabla HTML de una orden de compra comercial.
No incluyas la fila del TOTAL GENERAL dentro del arreglo de items; colócala en total_orden_usd.
</instructions>

<metadata_header>
Distribuidor: TechAndes Mayorista | Orden: PO-2028-9941 | Fecha: 2028-04-15
</metadata_header>

<html_table>
{html_tabla_orden}
</html_table>
"""

res_llm_orden = llamar_llm(
    prompt_extraccion_solucion,
    modelo=MODELO_20B,
    temperature=0.0,
    response_format=RESPONSE_FORMAT_ORDEN
)

datos_orden_json = parsear_json_limpio(res_llm_orden["texto"])
print("=== DATOS ESTRUCTURADOS EXTRAÍDOS ===")
print(json.dumps(datos_orden_json, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validación Aritmética Rigurosa y Carga en Spark DataFrame

# COMMAND ----------

def validar_aritmetica_orden(datos: Dict[str, Any]) -> Dict[str, Any]:
    if not datos or "items" not in datos:
        return {"valido": False, "motivo": "Estructura incompleta"}

    suma_calculada = 0.0
    errores_partidas = []

    for idx, it in enumerate(datos["items"]):
        cant = it.get("cantidad", 0)
        p_unit = it.get("precio_unitario_usd", 0.0)
        subt_declarado = it.get("subtotal_usd", 0.0)
        subt_calculado = round(cant * p_unit, 2)

        if abs(subt_calculado - subt_declarado) > 0.01:
            errores_partidas.append(
                f"Ítem {idx+1} ({it.get('modelo')}): Cant {cant} * Precio {p_unit} = {subt_calculado} != {subt_declarado}"
            )
        suma_calculada += subt_declarado

    gran_total_declarado = datos.get("total_orden_usd", 0.0)
    diferencia_total = abs(suma_calculada - gran_total_declarado)

    if errores_partidas:
        return {"valido": False, "motivo": "Discrepancia en subtotales: " + "; ".join(errores_partidas)}
    if diferencia_total > 0.01:
        return {
            "valido": False,
            "motivo": f"Discrepancia en gran total: Suma {suma_calculada} != Total {gran_total_declarado}"
        }

    return {
        "valido": True,
        "items_procesados": len(datos["items"]),
        "total_verificado_usd": gran_total_declarado
    }


audit_resultado = validar_aritmetica_orden(datos_orden_json)
print("Resultado de la auditoría aritmética:", audit_resultado)

# Despliegue en Spark DataFrame
filas_spark = []
for it in datos_orden_json["items"]:
    filas_spark.append({
        "orden_id": datos_orden_json["numero_orden"],
        "distribuidor": datos_orden_json["distribuidor"],
        "modelo": it["modelo"],
        "cantidad": it["cantidad"],
        "precio_unitario_usd": it["precio_unitario_usd"],
        "subtotal_usd": it["subtotal_usd"],
        "auditoria_ok": audit_resultado["valido"]
    })

df_orden_final = spark.createDataFrame(pd.DataFrame(filas_spark))
display(df_orden_final)

# COMMAND ----------

# Salida esperada:
# 1. Extracción de orden PO-2028-9941 con sus 4 productos dPhone.
# 2. Validación aritmética exitosa (10*699 + 5*999 + 3*1299 + 2*1899 = 19,680.00 USD).
# 3. DataFrame de Spark mostrando las 4 partidas con su columna de auditoria_ok = True.
