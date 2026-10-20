# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejemplo 1: Prompt Chaining de 3 Pasos para Diseño e Implementación de un Pipeline ETL
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Construir una arquitectura de **Prompt Chaining (Encadenamiento de Prompts)** de 3 etapas modulares en Databricks que transforma una solicitud de negocio desordenada en: (Paso 1) Especificación técnica JSON de reglas, (Paso 2) Código SQL/PySpark de transformación Medallion y (Paso 3) Auditoría automatizada de calidad y casos borde.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Descomposición de tareas complejas en cadenas secuenciales con validación intermedia en Python y asignación inteligente de modelos (`20b` vs `120b`) por eslabón.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura del Pipeline de Encadenamiento
# MAGIC Definimos el motor de invocación y el correo desestructurado enviado por el equipo de negocio que servirá como entrada inicial de nuestra cadena.

# COMMAND ----------

import json
import time
import re
from typing import Dict, Any
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"

CORREO_NEGOCIO_CRUDO = """
Asunto: Urgente - Necesitamos la tabla consolidada de Suscripciones Activas para el comité
Hola equipo de datos,
Para el cierre mensual necesitamos cruzar la tabla `raw_catalog.billing.suscripciones` (tiene sub_id, usuario_id, plan, valor_mensual_usd, fecha_inicio, estado_sub) con la tabla `raw_catalog.usuarios.perfil_clientes` (tiene usuario_id, pais, segmento_empresa, cuenta_prueba).
Ojo con esto:
1. No vayan a meter las cuentas donde `cuenta_prueba = true` porque son demos internos.
2. Solo nos interesan las suscripciones con `estado_sub` en ('ACTIVA', 'PERIODO_GRACIA').
3. Si el `valor_mensual_usd` viene nulo o menor a cero, pónganlo en 0.0.
4. Necesitamos crear una columna nueva llamada `arr_anual_usd` que sea `valor_mensual_usd * 12`.
5. El resultado debe guardarse o consultarse como vista hacia `gold_catalog.finanzas.arr_por_pais_segmento` agrupando por `pais` y `segmento_empresa`, calculando el total de suscripciones y la suma de `arr_anual_usd`.
¡Gracias!
""".strip()

def ejecutar_eslabon(nombre_paso: str, prompt: str, modelo: str, temp: float = 0.0, max_tokens: int = 650) -> Dict[str, Any]:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_esc}',
        modelParameters => named_struct(
            'temperature', CAST({float(temp)} AS DOUBLE),
            'max_tokens', CAST({int(max_tokens)} AS INT)
        )
    ) AS respuesta
    """
    try:
        salida = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        salida = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    duracion = round(time.time() - t0, 3)
    texto = salida.strip() if salida else ""
    print(f"[{nombre_paso}] Completado en {duracion}s con {modelo} ({len(texto)} caracteres).")
    return {
        "paso": nombre_paso,
        "modelo": modelo,
        "temperature": temp,
        "latencia_seg": duracion,
        "salida": texto
    }

print("Entrada de negocio lista para iniciar el Prompt Chaining.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Eslabón 1 (`databricks-gpt-oss-20b`): Extracción Estructurada de Entidades y Reglas ETL
# MAGIC El primer eslabón convierte el correo informal en una especificación técnica estructurada en formato JSON. Validamos en Python que el JSON contenga las llaves requeridas antes de pasar al Eslabón 2.

# COMMAND ----------

prompt_paso_1 = f"""
### ROL
Actúa como un Analista Técnico de Ingeniería de Datos.

### TAREA
Extrae del siguiente correo de negocio los metadatos técnicos del requerimiento ETL y devuélvelos ÚNICAMENTE como un objeto JSON válido con las siguientes llaves exactas:
- "tablas_origen": lista de nombres de tablas origen
- "tabla_destino": string con la tabla destino
- "llave_join": string con el campo de cruce
- "filtros_obligatorios": lista de strings con las condiciones de filtrado
- "transformaciones": lista de strings con las reglas de limpieza y columnas calculadas
- "agrupaciones": lista de columnas para el GROUP BY

### CORREO DE ENTRADA
{CORREO_NEGOCIO_CRUDO}

### RESTRICCIÓN DE SALIDA
Responde SOLO con el objeto JSON plano, sin bloques ```json ni texto adicional.
""".strip()

res_paso_1 = ejecutar_eslabon("Paso 1: Extracción de Especificación", prompt_paso_1, modelo=MODELO_20B, temp=0.0)

texto_json_1 = re.sub(r"^```(?:json)?|```$", "", res_paso_1["salida"].strip(), flags=re.MULTILINE).strip()
try:
    especificacion_etl = json.loads(texto_json_1)
    validacion_paso_1 = "VALIDACIÓN JSON EXITOSA"
except json.JSONDecodeError:
    especificacion_etl = {"especificacion_texto": texto_json_1}
    validacion_paso_1 = "FALLBACK TEXTO ESTRUCTURADO"

print(f"Estado de compuerta de calidad Paso 1: {validacion_paso_1}")
print(json.dumps(especificacion_etl, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Eslabón 2 (`databricks-gpt-oss-120b`): Síntesis de Código Databricks SQL (Medallion Gold)
# MAGIC Inyectamos la salida validada del **Paso 1** como contexto de entrada del **Paso 2**. Aquí usamos `databricks-gpt-oss-120b` con `temperature = 0.0` para generar la sentencia `CREATE OR REPLACE TABLE ... AS SELECT` siguiendo mejores prácticas de Databricks SQL.

# COMMAND ----------

json_formateado_paso_1 = json.dumps(especificacion_etl, indent=2, ensure_ascii=False)

prompt_paso_2 = f"""
### ROL
Actúa como un Lead Data Engineer experto en Databricks SQL y arquitectura Medallion.

### ESPECIFICACIÓN TÉCNICA VALIDADA (SALIDA DEL PASO 1)
{json_formateado_paso_1}

### TAREA
Escribe el script completo en Databricks SQL que cree o reemplace la tabla destino (`CREATE OR REPLACE TABLE ...`) implementando mediante CTEs (Common Table Expressions):
1. El filtrado de cuentas de prueba y estados válidos.
2. El manejo de nulos y valores negativos en `valor_mensual_usd` usando `CASE WHEN` o `COALESCE` y `GREATEST`.
3. El cálculo de `arr_anual_usd`.
4. El `INNER JOIN` por `usuario_id` y la agregación final por `pais` y `segmento_empresa`.

### RESTRICCIÓN DE SALIDA
Devuelve únicamente código SQL válido para Databricks, sin bloques markdown ni comentarios fuera del SQL.
""".strip()

res_paso_2 = ejecutar_eslabon("Paso 2: Generación de Código SQL", prompt_paso_2, modelo=MODELO_120B, temp=0.0)
codigo_sql_generado = res_paso_2["salida"]

print("=== CÓDIGO DATABRICKS SQL GENERADO EN EL PASO 2 ===")
print(codigo_sql_generado)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Eslabón 3 (`databricks-gpt-oss-20b`): Auditoría de Calidad (QA) y Generación de Asserts SQL
# MAGIC Finalmente, el **Paso 3** recibe tanto la especificación del Paso 1 como el código SQL del Paso 2 para verificar el cumplimiento de cada regla de negocio y generar una consulta de prueba de calidad de datos (*Data Quality Check*).

# COMMAND ----------

prompt_paso_3 = f"""
### ROL
Actúa como un Ingeniero de QA de Datos (Data Quality Auditor).

### ENTRADA 1: ESPECIFICACIÓN DE REGLAS
{json_formateado_paso_1}

### ENTRADA 2: CÓDIGO SQL GENERADO
{codigo_sql_generado}

### TAREA
1. Verifica si el código SQL cumple las 5 reglas del negocio (cuentas de prueba, filtro de estado_sub, manejo de negativos/nulos, cálculo de arr_anual_usd y agrupación final).
2. Genera una consulta SQL de validación (Data Quality Assertion) sobre la tabla `gold_catalog.finanzas.arr_por_pais_segmento` que devuelva 0 filas si todos los `arr_anual_usd` son mayores o iguales a 0.
""".strip()

res_paso_3 = ejecutar_eslabon("Paso 3: Auditoría QA y Tests", prompt_paso_3, modelo=MODELO_20B, temp=0.1)
print("=== REPORTE DE AUDITORÍA Y DATA QUALITY CHECK (PASO 3) ===")
print(res_paso_3["salida"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Telemetría Consolidada del Pipeline de 3 Pasos

# COMMAND ----------

df_telemetria = pd.DataFrame([
    {
        "Eslabón de la Cadena": r["paso"],
        "Modelo Utilizado": r["modelo"],
        "Temperature": r["temperature"],
        "Latencia (s)": r["latencia_seg"],
        "Tamaño Salida (chars)": len(r["salida"])
    }
    for r in [res_paso_1, res_paso_2, res_paso_3]
])

display(spark.createDataFrame(df_telemetria))

# COMMAND ----------

# Salida esperada:
# 1. Eslabón 1: Objeto JSON limpio con las tablas origen, destino, llave `usuario_id` y las reglas de transformación extraídas del correo.
# 2. Eslabón 2: Sentencia `CREATE OR REPLACE TABLE gold_catalog.finanzas.arr_por_pais_segmento AS WITH ...` en Databricks SQL implementando el filtro `NOT cuenta_prueba`, `estado_sub IN ('ACTIVA', 'PERIODO_GRACIA')` y el cálculo de `arr_anual_usd`.
# 3. Eslabón 3: Checklist de cumplimiento (5/5 reglas) y consulta SQL de Data Quality Assertion, seguido de la tabla de telemetría del pipeline.
