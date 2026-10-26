# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 1: Salidas Estructuradas Deterministas: JSON Mode vs Strict JSON Schema en Databricks
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Comparar empíricamente tres métodos de extracción estructurada sobre 4 reseñas/tickets de soporte de dPhone: (a) Prompting tradicional en texto libre, (b) JSON Mode (`json_object`), y (c) Structured Outputs (`responseFormat` con `json_schema` strict). Medir cumplimiento estricto de esquema y transformar el resultado en columnas nativas de Spark SQL usando `from_json`.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Decodificación restringida (*Constrained Decoding* / Grammars), contratos de interfaz hacia sistemas downstream, eliminación de parsers heurísticos frágiles y deserialización directa a tablas Delta Lake.
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %pip install pydantic

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura de Invocación y Helper con Soporte para `responseFormat`

# COMMAND ----------

import json
import re
import time
from typing import Dict, Any, List
import pandas as pd
from pydantic import BaseModel, Field

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
               max_tokens: int = 800, response_format: str = None) -> dict:
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
        # Fallback si el workspace no admite responseFormat
        salida = spark.sql(f"SELECT ai_query('{modelo}', '{_esc(prompt)}') AS respuesta").collect()[0]["respuesta"]
    return {
        "texto": extraer_texto(salida),
        "latencia_s": round(time.time() - t0, 2)
    }


print("Helper de Structured Outputs configurado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Definición del Contrato de Interfaz: JSON Schema Estricto
# MAGIC Buscamos extraer de cada reseña/ticket:
# MAGIC - `producto`: modelo de dPhone detectado
# MAGIC - `sentimiento`: 'POSITIVO', 'NEUTRO' o 'NEGATIVO'
# MAGIC - `problemas`: lista de incidentes técnicos detectados
# MAGIC - `severidad`: número entero entre 1 y 5
# MAGIC - `requiere_escalamiento`: booleano

# COMMAND ----------

SCHEMA_EXTRACCION = {
    "type": "object",
    "properties": {
        "producto": {
            "type": "string",
            "enum": ["dPhone D1", "dPhone D1 Lite", "dPhone D1 Pro", "dPhone D1 Fold", "DESCONOCIDO"]
        },
        "sentimiento": {
            "type": "string",
            "enum": ["POSITIVO", "NEUTRO", "NEGATIVO"]
        },
        "problemas": {
            "type": "array",
            "items": {"type": "string"}
        },
        "severidad": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "requiere_escalamiento": {
            "type": "boolean"
        }
    },
    "required": ["producto", "sentimiento", "problemas", "severidad", "requiere_escalamiento"],
    "additionalProperties": False
}

RESPONSE_FORMAT_JSON_SCHEMA = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "extraccion_soporte_dphone",
        "schema": SCHEMA_EXTRACCION,
        "strict": True
    }
})

RESPONSE_FORMAT_JSON_OBJECT = json.dumps({"type": "json_object"})

print("Esquemas JSON Schema preparados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Casos de Prueba (4 Tickets / Reseñas de dPhone)

# COMMAND ----------

TICKETS_PRUEBA = [
    {
        "id": "T1",
        "texto": "Compré el dPhone D1 Pro hace 3 días. La cámara Halo de 200MP es increíble para fotos nocturnas, pero el dispositivo calienta excesivamente al cargar a 80W y se apaga solo. Temo por la batería."
    },
    {
        "id": "T2",
        "texto": "Excelente experiencia con el dPhone D1 estándar. La pantalla a 144Hz es muy fluida y el escaneo de venas funciona perfecto. Cero quejas, vale cada dólar."
    },
    {
        "id": "T3",
        "texto": "El dPhone D1 Fold que recibí tiene la bisagra trabada a 90 grados y la pantalla interna parpadea con líneas verdes. Es inaceptable para un teléfono de 1900 dólares. Exijo solución ya."
    },
    {
        "id": "T4",
        "texto": "¿Saben si el dPhone D1 Lite viene con cargador en la caja? Estuve revisando la web y no me queda claro ese punto antes de comprar."
    }
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Experimento: Comparativa de Métodos de Extracción

# COMMAND ----------

def validar_cumplimiento_esquema(obj: Any) -> bool:
    """Valida si el objeto cumple todas las claves y tipos del contrato."""
    if not isinstance(obj, dict):
        return False
    keys_requeridas = {"producto", "sentimiento", "problemas", "severidad", "requiere_escalamiento"}
    if not keys_requeridas.issubset(obj.keys()):
        return False
    if obj["sentimiento"] not in ["POSITIVO", "NEUTRO", "NEGATIVO"]:
        return False
    if not isinstance(obj["problemas"], list):
        return False
    if not isinstance(obj["severidad"], int) or not (1 <= obj["severidad"] <= 5):
        return False
    if not isinstance(obj["requiere_escalamiento"], bool):
        return False
    return True


resultados_comparativa = []

for item in TICKETS_PRUEBA:
    tid = item["id"]
    texto_ticket = item["texto"]
    
    # Método A: Solo Prompting (Instrucción de texto)
    prompt_a = f"""
    Analiza este ticket: "{texto_ticket}".
    Extrae un JSON con: producto, sentimiento, problemas, severidad (1-5), requiere_escalamiento (true/false).
    Responde solo el JSON.
    """
    res_a = llamar_llm(prompt_a, modelo=MODELO_20B, temperature=0.0)
    json_a_limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", res_a["texto"].strip())
    try:
        obj_a = json.loads(json_a_limpio)
        valido_a = validar_cumplimiento_esquema(obj_a)
    except Exception:
        obj_a, valido_a = None, False

    # Método B: responseFormat json_object (JSON Mode)
    prompt_b = f"""
    Extrae la información del ticket en un objeto JSON con las claves exactas:
    producto, sentimiento (POSITIVO|NEUTRO|NEGATIVO), problemas (lista), severidad (1-5), requiere_escalamiento (booleano).
    Ticket: "{texto_ticket}"
    """
    res_b = llamar_llm(prompt_b, modelo=MODELO_20B, temperature=0.0, response_format=RESPONSE_FORMAT_JSON_OBJECT)
    try:
        obj_b = json.loads(res_b["texto"])
        valido_b = validar_cumplimiento_esquema(obj_b)
    except Exception:
        obj_b, valido_b = None, False

    # Método C: responseFormat json_schema (Structured Outputs Estricto)
    prompt_c = f"""
    Extrae los datos estructurados del ticket técnico de acuerdo con el esquema especificado.
    Ticket: "{texto_ticket}"
    """
    res_c = llamar_llm(prompt_c, modelo=MODELO_20B, temperature=0.0, response_format=RESPONSE_FORMAT_JSON_SCHEMA)
    try:
        obj_c = json.loads(res_c["texto"])
        valido_c = validar_cumplimiento_esquema(obj_c)
    except Exception:
        obj_c, valido_c = None, False

    resultados_comparativa.append({
        "Ticket": tid,
        "Texto": texto_ticket[:60] + "...",
        "JSON_Valido_Metodo_A": valido_a,
        "JSON_Valido_Metodo_B": valido_b,
        "JSON_Valido_Metodo_C": valido_c,
        "Objeto_Metodo_C": json.dumps(obj_c, ensure_ascii=False) if obj_c else "ERROR"
    })

df_res_comp = pd.DataFrame(resultados_comparativa)
display(spark.createDataFrame(df_res_comp))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ingesta Directa a Tablas Delta mediante Spark SQL y `from_json`
# MAGIC Con Structured Outputs garantizado, podemos deserializar las respuestas de `ai_query` directamente en columnas fuertemente tipadas de Apache Spark.

# COMMAND ----------

# Definición del esquema DDL en Spark SQL compatible con el JSON Schema
spark_schema_ddl = """
    producto STRING,
    sentimiento STRING,
    problemas ARRAY<STRING>,
    severidad INT,
    requiere_escalamiento BOOLEAN
"""

# Simulamos la consulta SQL batch que en producción lee una tabla Bronze
df_raw_tickets = spark.createDataFrame([
    ("T1", "El dPhone D1 Pro recalienta al cargar a 80W y se apaga solo."),
    ("T3", "El dPhone D1 Fold tiene bisagra trabada y líneas verdes en pantalla.")
], ["ticket_id", "contenido"])

df_raw_tickets.createOrReplaceTempView("tickets_bronze_temp")

sql_parsing_pipeline = f"""
WITH respuestas_ia AS (
    SELECT
        ticket_id,
        ai_query(
            '{MODELO_20B}',
            CONCAT('Extrae la información técnica del ticket: ', contenido),
            responseFormat => '{_esc(RESPONSE_FORMAT_JSON_SCHEMA)}'
        ) AS json_crudo
    FROM tickets_bronze_temp
)
SELECT
    ticket_id,
    from_json(json_crudo, '{spark_schema_ddl}') AS datos_estructurados
FROM respuestas_ia
"""

df_silver_estructurado = spark.sql(sql_parsing_pipeline).select(
    "ticket_id",
    "datos_estructurados.producto",
    "datos_estructurados.sentimiento",
    "datos_estructurados.problemas",
    "datos_estructurados.severidad",
    "datos_estructurados.requiere_escalamiento"
)

print("=== TABLA SILVER CON TIPADO NATIVO SPARK SQL ===")
display(df_silver_estructurado)

# COMMAND ----------

# Salida esperada:
# 1. Comparativa empírica donde el Método C (json_schema) alcanza 100% de cumplimiento estricto de tipos, enums y claves.
# 2. Deserialización nativa con from_json en Spark SQL creando columnas fuertemente tipadas (STRING, ARRAY<STRING>, INT, BOOLEAN) listas para analítica y pipelines Medallion.
