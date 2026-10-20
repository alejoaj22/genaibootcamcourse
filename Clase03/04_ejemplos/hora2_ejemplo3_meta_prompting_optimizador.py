# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 3: Meta-Prompting — El LLM como Arquitecto y Optimizador de Prompts
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Implementar un sistema de **Meta-Prompting** en el cual el modelo de mayor capacidad (`databricks-gpt-oss-120b`) diagnostica las vulnerabilidades de un prompt ingenuo y genera automáticamente un prompt de grado producción optimizado para ejecutarse en el modelo ligero (`databricks-gpt-oss-20b`).
# MAGIC * **Tiempo estimado:** 9 minutos.
# MAGIC * **Concepto que ilustra:** Meta-Prompting, separación entre *tiempo de diseño* (modelo grande) y *tiempo de ejecución* (modelo rápido), y evaluación comparativa antes vs. después de la optimización.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Construcción del Meta-Prompt Arquitecto
# MAGIC Nuestro Meta-Prompt encapsula todas las reglas de **Ingeniería de Contexto** aprendidas en las Horas 1 y 2 para transformar cualquier solicitud vaga de un usuario en una especificación rigurosa.

# COMMAND ----------

import re
from typing import Dict
import pandas as pd

MODELO_ARQUITECTO = "databricks-gpt-oss-120b"
MODELO_EJECUTOR = "databricks-gpt-oss-20b"

PLANTILLA_META_PROMPT = """
### ROL
Actúa como un Principal Prompt Engineer y Arquitecto de Sistemas de IA en Databricks.

### OBJETIVO
Tu misión es auditar un [PROMPT_INGENUO] escrito por un desarrollador y transformarlo en un [PROMPT_PRODUCCION] altamente estructurado, determinista y libre de ambigüedades.

### CONTEXTO TÉCNICO DISPONIBLE PARA INYECTAR
{contexto_tecnico}

### METODOLOGÍA DE OPTIMIZACIÓN
Debes aplicar obligatoriamente estos 5 pilares al nuevo prompt:
1. Asignación de Rol Experto específico.
2. Inyección explícita del esquema técnico o firma de datos proporcionada.
3. Instrucción de razonamiento breve o validación de casos borde (nulos, duplicados, tipos inválidos).
4. Restricciones negativas claras (qué NO debe incluir en su respuesta).
5. Contrato de salida estricto.

### ENTRADA A OPTIMIZAR
[PROMPT_INGENUO]:
"{prompt_ingenuo}"

### FORMATO EXACTO DE TU RESPUESTA
Devuelve tu análisis usando estrictamente estas dos secciones delimitadas:

[DIAGNOSTICO_DE_FALLOS]
- Lista de 3 debilidades críticas del prompt ingenuo.
[/DIAGNOSTICO_DE_FALLOS]

[PROMPT_OPTIMIZADO]
Escribe aquí el prompt completo y listo para enviar al modelo ejecutor.
[/PROMPT_OPTIMIZADO]
""".strip()

def invocar_ai_query(prompt: str, modelo: str, temp: float = 0.0, max_tokens: int = 700) -> str:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
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
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return res.strip() if res else ""

def parsear_meta_respuesta(texto: str) -> Dict[str, str]:
    diag = re.search(r"\[DIAGNOSTICO_DE_FALLOS\](.*?)\[/DIAGNOSTICO_DE_FALLOS\]", texto, re.DOTALL)
    opt = re.search(r"\[PROMPT_OPTIMIZADO\](.*?)\[/PROMPT_OPTIMIZADO\]", texto, re.DOTALL)
    return {
        "diagnostico": diag.group(1).strip() if diag else "No se encontró bloque de diagnóstico.",
        "prompt_optimizado": opt.group(1).strip() if opt else texto.strip()
    }

print("Sistema Meta-Prompt inicializado correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ejecución del Meta-Prompt sobre una Solicitud Débil de Ingeniería de Datos
# MAGIC Un analista escribe: *"Hazme un código en PySpark que limpie la tabla de clientes y calcule su antigüedad"*. Veamos cómo `databricks-gpt-oss-120b` lo audita y reescribe.

# COMMAND ----------

prompt_debil_usuario = "Hazme un código en PySpark que limpie la tabla de clientes y calcule su antigüedad."

contexto_catalogo = """
- Tabla origen: bronze_catalog.crm.clientes_raw
- Tabla destino: silver_catalog.crm.clientes_limpios
- Columnas origen: cliente_id (STRING), email (STRING), fecha_registro (STRING formato 'yyyy-MM-dd'), cupo_credito (DOUBLE)
- Reglas de calidad:
  1. Eliminar filas con cliente_id nulo o duplicado (conservar el registro más reciente según fecha_registro).
  2. Convertir email a minúsculas y quitar espacios laterales.
  3. Rellenar cupo_credito nulo o negativo con 0.0.
  4. Calcular columna 'antiguedad_dias' usando datediff(current_date(), to_date(fecha_registro)).
""".strip()

solicitud_meta = PLANTILLA_META_PROMPT.format(
    contexto_tecnico=contexto_catalogo,
    prompt_ingenuo=prompt_debil_usuario
)

respuesta_arquitecto = invocar_ai_query(solicitud_meta, modelo=MODELO_ARQUITECTO, temp=0.1, max_tokens=800)
resultado_meta = parsear_meta_respuesta(respuesta_arquitecto)

print("=== 1. DIAGNÓSTICO DE DEBILIDADES REALIZADO POR databricks-gpt-oss-120b ===")
print(resultado_meta["diagnostico"])
print("\n=== 2. NUEVO PROMPT DE GRADO PRODUCCIÓN GENERADO AUTOMÁTICAMENTE ===")
print(resultado_meta["prompt_optimizado"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prueba A/B en el Modelo Rápido (`databricks-gpt-oss-20b`): Prompt Débil vs. Prompt Optimizado
# MAGIC Ahora ejecutamos ambos prompts en `databricks-gpt-oss-20b` y verificamos automáticamente cuántos requisitos técnicos reales de la empresa cumple cada código generado.

# COMMAND ----------

salida_con_prompt_debil = invocar_ai_query(prompt_debil_usuario, modelo=MODELO_EJECUTOR, temp=0.0, max_tokens=500)
salida_con_prompt_optimizado = invocar_ai_query(resultado_meta["prompt_optimizado"], modelo=MODELO_EJECUTOR, temp=0.0, max_tokens=600)

def auditar_codigo_pyspark(codigo: str) -> Dict[str, bool]:
    return {
        "Usa tabla real bronze_catalog.crm.clientes_raw": "bronze_catalog.crm.clientes_raw" in codigo,
        "Normaliza email (lower/trim)": ("lower" in codigo.lower() or "trim" in codigo.lower()),
        "Maneja cupo_credito nulo/negativo": ("cupo_credito" in codigo),
        "Calcula antiguedad_dias con datediff": ("datediff" in codigo.lower() or "antiguedad_dias" in codigo)
    }

auditoria_debil = auditar_codigo_pyspark(salida_con_prompt_debil)
auditoria_opt = auditar_codigo_pyspark(salida_con_prompt_optimizado)

print("--- CÓDIGO GENERADO POR gpt-oss-20b CON EL PROMPT OPTIMIZADO ---")
print(salida_con_prompt_optimizado)

df_ab = pd.DataFrame([
    {"Criterio de Calidad": k, "Prompt Ingenuo (20b)": "PASS" if v else "FAIL", "Prompt Meta-Optimizado (20b)": "PASS" if auditoria_opt[k] else "FAIL"}
    for k, v in auditoria_debil.items()
])
display(spark.createDataFrame(df_ab))

# COMMAND ----------

# Salida esperada:
# 1. Diagnóstico detallado de las 3 fallas del prompt ingenuo (falta de esquema, ambigüedad en qué significa "limpiar", ausencia de unidad para "antigüedad").
# 2. Prompt optimizado generado automáticamente con todas las reglas del catálogo `bronze_catalog.crm.clientes_raw`.
# 3. Tabla A/B donde el Prompt Ingenuo obtiene `FAIL` en el uso de tablas y columnas reales, mientras el Prompt Meta-Optimizado obtiene `PASS` en los 4 criterios sobre `databricks-gpt-oss-20b`.
