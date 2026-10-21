# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 2 (Solución Completa): Zero-Shot vs. Few-Shot vs. Chain-of-Thought
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Solución completa y comentada que contrasta Zero-Shot, Few-Shot Balanceado y Chain-of-Thought con Scratchpad sobre los 3 problemas del laboratorio en Databricks.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Ganancia incremental de exactitud al añadir ejemplos demostrativos (Few-Shot) y presupuesto de cómputo intermedio (`[SCRATCHPAD]`).
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Definición de los 3 Problemas y Motor de Evaluación

# COMMAND ----------

import re
import time
from typing import Dict, Any, List
import pandas as pd

MODELO_EVAL = "databricks-gpt-oss-20b"

PROBLEMAS: List[Dict[str, str]] = [
    {
        "id": "P1_CLASIFICACION_DQ",
        "enunciado": (
            "Clasifica la siguiente alerta de calidad de datos usando ÚNICAMENTE una de estas etiquetas: "
            "DQ_SCHEMA_DRIFT, DQ_NULL_SPIKE, DQ_LATENCY_SLA, DQ_FALSE_ALARM.\n"
            "Alerta: 'Atención: la columna telefono_secundario pasó de 12% a 14% de valores vacíos tras la campaña "
            "opcional de registro rápido web, manteniéndose por debajo del umbral máximo permitido del 25%.'"
        ),
        "esperado": "DQ_FALSE_ALARM"
    },
    {
        "id": "P2_CALCULO_INVENTARIO",
        "enunciado": (
            "Calcula el número entero exacto de unidades disponibles para venta web (stock_vendible) según estas reglas:\n"
            "- Bodega Central tiene 450 unidades y Bodega Norte tiene 350 unidades.\n"
            "- Del total físico combinado se descarta primero el 5% por merma de humedad.\n"
            "- De las unidades buenas restantes se descuentan 160 unidades reservadas para pedidos B2B y un colchón fijo de seguridad de 40 unidades.\n"
            "Devuelve como veredicto únicamente el número entero final (ejemplo: 500)."
        ),
        "esperado": "560"
    },
    {
        "id": "P3_LOGICA_REGLAS",
        "enunciado": (
            "Analiza estas tres reglas de un motor de facturación:\n"
            "- REGLA_A: Si antiguedad >= 24 meses -> aplicar DESCUENTO_15.\n"
            "- REGLA_B: Si mora_activa == True -> prohibido otorgar cualquier descuento (SIN_DESCUENTO).\n"
            "- REGLA_C: Si segmento == 'CORPORATIVO' -> otorgar siempre DESCUENTO_20 sin importar el estado de mora.\n"
            "Si llega un cliente con segmento='CORPORATIVO', antiguedad=36 y mora_activa=True, ¿cuáles son las dos reglas "
            "que tienen una contradicción directa e irreconciliable sobre la mora? Responde con el código exacto: "
            "CONFLICTO_AB, CONFLICTO_AC o CONFLICTO_BC."
        ),
        "esperado": "CONFLICTO_BC"
    }
]

def invocar_llm(prompt: str, max_tokens: int = 450) -> Dict[str, Any]:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
    sql = f"""
    SELECT ai_query(
        '{MODELO_EVAL}',
        '{prompt_esc}',
        modelParameters => named_struct('temperature', CAST(0.0 AS DOUBLE), 'max_tokens', CAST({int(max_tokens)} AS INT))
    ) AS respuesta
    """
    try:
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{MODELO_EVAL}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return {
        "latencia": round(time.time() - t0, 3),
        "texto": res.strip() if res else ""
    }

def extraer_veredicto(texto: str) -> str:
    match = re.search(r"\[VEREDICTO_FINAL\](.*?)\[/VEREDICTO_FINAL\]", texto, re.DOTALL)
    if match:
        return match.group(1).strip()
    return texto.strip().splitlines()[-1].strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Implementación de las 3 Plantillas (Zero-Shot, Few-Shot Balanceado y CoT Scratchpad)

# COMMAND ----------

def prompt_zeroshot(enunciado: str) -> str:
    return f"""
{enunciado}

RESTRICCIÓN: Prohibido incluir razonamiento o pasos intermedios. Escribe directamente la respuesta dentro de [VEREDICTO_FINAL] y [/VEREDICTO_FINAL].
""".strip()

def prompt_fewshot(enunciado: str) -> str:
    return f"""
Actúa como un Analista Senior de Ingeniería de Datos y Calidad de Software.
Estudia los siguientes ejemplos resueltos y aplica el mismo rigor al problema final:

### EJEMPLO 1 (Alerta por encima vs por debajo del umbral):
Entrada: "Alerta: el porcentaje de nulos en direccion_envio subió de 5% a 8%, pero el límite máximo tolerado por gobernanza es 15%."
Salida: [VEREDICTO_FINAL]DQ_FALSE_ALARM[/VEREDICTO_FINAL]

### EJEMPLO 2 (Alerta crítica real de nulos):
Entrada: "Alerta: la columna llave cliente_id registró 32% de nulos cuando el umbral máximo permitido es 0%."
Salida: [VEREDICTO_FINAL]DQ_NULL_SPIKE[/VEREDICTO_FINAL]

### EJEMPLO 3 (Cálculo secuencial de inventario disponible):
Entrada: "Bodega Sur=600, Bodega Este=400 (Total=1000). Merma del 10% (quedan 900). Reservas B2B=200 y colchón=50."
Salida: [VEREDICTO_FINAL]650[/VEREDICTO_FINAL]

### EJEMPLO 4 (Auditoría de contradicción entre regla prohibitiva y regla mandatoria):
Entrada: "REGLA_A da bono por antigüedad. REGLA_B prohíbe bonos a cuentas suspendidas. REGLA_C obliga a dar bono siempre a cuentas VIP incluso si están suspendidas."
Salida: [VEREDICTO_FINAL]CONFLICTO_BC[/VEREDICTO_FINAL]

### PROBLEMA A RESOLVER:
{enunciado}
Salida:
""".strip()

def prompt_cot_scratchpad(enunciado: str) -> str:
    return f"""
Actúa como un Ingeniero Principal de Datos y Lógica Formal.
Resuelve el problema siguiendo rigurosamente el método Scratchpad Chain-of-Thought:

### PROBLEMA
{enunciado}

### ESTRUCTURA OBLIGATORIA DE RESPUESTA
[SCRATCHPAD]
- Paso 1: Extrae todas las variables, cifras y condiciones explícitas del enunciado.
- Paso 2: Si hay porcentajes o umbrales, realiza el cálculo matemático explícito paso a paso.
- Paso 3: Verifica cuál etiqueta, número o código cumple exactamente las condiciones sin dejarte llevar por palabras distractivas.
[/SCRATCHPAD]
[VEREDICTO_FINAL]Escribe aquí únicamente la etiqueta, número o código exacto solicitado[/VEREDICTO_FINAL]
""".strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ejecución y Análisis Detallado de Resultados

# COMMAND ----------

registros_comparativos = []

for prob in PROBLEMAS:
    esperado = prob["esperado"]
    
    r_zero = invocar_llm(prompt_zeroshot(prob["enunciado"]), max_tokens=80)
    v_zero = extraer_veredicto(r_zero["texto"])
    
    r_few = invocar_llm(prompt_fewshot(prob["enunciado"]), max_tokens=120)
    v_few = extraer_veredicto(r_few["texto"])
    
    r_cot = invocar_llm(prompt_cot_scratchpad(prob["enunciado"]), max_tokens=500)
    v_cot = extraer_veredicto(r_cot["texto"])
    
    print("=" * 85)
    print(f"PROBLEMA: {prob['id']} | ESPERADO: {esperado}")
    print(f"  -> Zero-Shot Verdict       : {v_zero}")
    print(f"  -> Few-Shot Verdict        : {v_few}")
    print(f"  -> CoT + Scratchpad Verdict: {v_cot}")
    print("  -> Razonamiento CoT completo:")
    print(r_cot["texto"])
    
    registros_comparativos.append({
        "Problema": prob["id"],
        "Esperado": esperado,
        "Zero-Shot": v_zero,
        "Few-Shot": v_few,
        "CoT + Scratchpad": v_cot,
        "Acierto Zero-Shot": (esperado == v_zero or esperado in v_zero),
        "Acierto Few-Shot": (esperado == v_few or esperado in v_few),
        "Acierto CoT": (esperado == v_cot or esperado in v_cot)
    })

df_final = pd.DataFrame(registros_comparativos)
display(spark.createDataFrame(df_final))

# COMMAND ----------

# Salida esperada:
# 1. Impresión detallada en consola del bloque `[SCRATCHPAD]` para cada uno de los 3 problemas (`DQ_FALSE_ALARM`, `560`, `CONFLICTO_BC`).
# 2. Tabla comparativa donde CoT + Scratchpad obtiene `True` en los 3 problemas y deja trazabilidad auditable de cada operación lógica y aritmética.
