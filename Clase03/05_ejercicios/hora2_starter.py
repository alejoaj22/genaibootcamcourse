# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 2 (Starter): Resolver 3 Problemas con Zero-Shot, Few-Shot y Chain-of-Thought
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Implementar y comparar empíricamente **Zero-Shot**, **Few-Shot** y **Chain-of-Thought (Scratchpad)** sobre 3 problemas de clasificación, cálculo de inventario y lógica de reglas en Databricks.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Selección de la técnica de prompting óptima según la naturaleza cognitiva del problema.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración y Definición de los 3 Problemas

# COMMAND ----------

import re
from typing import Dict, Any, List
import pandas as pd

MODELO = "databricks-gpt-oss-20b"

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

def invocar_llm(prompt: str, max_tokens: int = 400) -> str:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    sql = f"""
    SELECT ai_query(
        '{MODELO}',
        '{prompt_esc}',
        modelParameters => named_struct('temperature', CAST(0.0 AS DOUBLE), 'max_tokens', CAST({int(max_tokens)} AS INT))
    ) AS respuesta
    """
    try:
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{MODELO}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return res.strip() if res else ""

def extraer_veredicto(texto: str) -> str:
    match = re.search(r"\[VEREDICTO_FINAL\](.*?)\[/VEREDICTO_FINAL\]", texto, re.DOTALL)
    if match:
        return match.group(1).strip()
    return texto.strip().splitlines()[-1].strip()

print("3 problemas cargados correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Construcción de las 3 Estrategias de Prompting (`Zero-Shot`, `Few-Shot` y `CoT`)
# MAGIC Completa `# TODO 1` (ejemplos Few-Shot) y `# TODO 2` (instrucciones paso a paso de Chain-of-Thought con Scratchpad).

# COMMAND ----------

def prompt_zeroshot(enunciado: str) -> str:
    return f"""
{enunciado}

RESTRICCIÓN: No des ninguna explicación. Escribe directamente la respuesta final dentro de [VEREDICTO_FINAL] y [/VEREDICTO_FINAL].
""".strip()

# TODO 1: Revisa y completa el bloque de ejemplos Few-Shot para cubrir los 3 tipos de problemas
def prompt_fewshot(enunciado: str) -> str:
    return f"""
Resuelve el problema técnico guiándote por el formato y criterio de estos ejemplos demostrativos:

### EJEMPLO DEMOSTRATIVO 1 (Clasificación con umbral no superado):
Entrada: "El tiempo de carga del reporte subió de 4s a 6s, pero el SLA máximo permitido es de 15s."
Salida: [VEREDICTO_FINAL]DQ_FALSE_ALARM[/VEREDICTO_FINAL]

### EJEMPLO DEMOSTRATIVO 2 (Cálculo de stock neto):
Entrada: "Bodega A=200, Bodega B=100 (Total=300). Merma 10% (quedan 270). Reserva B2B=50, Colchón=20."
Salida: [VEREDICTO_FINAL]200[/VEREDICTO_FINAL]

### EJEMPLO DEMOSTRATIVO 3 (Detección de reglas mutuamente excluyentes):
Entrada: "REGLA_X prohíbe envío si peso > 50kg. REGLA_Y obliga envío gratis siempre a clientes VIP sin importar peso."
Salida: [VEREDICTO_FINAL]CONFLICTO_XY[/VEREDICTO_FINAL]

### PROBLEMA ACTUAL A RESOLVER:
{enunciado}
Salida:
""".strip()

# TODO 2: Completa las instrucciones del patrón Scratchpad CoT para que el modelo desglose los datos antes del veredicto
def prompt_cot_scratchpad(enunciado: str) -> str:
    return f"""
Resuelve el siguiente problema aplicando razonamiento paso a paso (Chain-of-Thought):

### PROBLEMA
{enunciado}

### INSTRUCCIONES DE RAZONAMIENTO
1. Abre un bloque [SCRATCHPAD] y [/SCRATCHPAD] donde:
   - Identifiques los datos numéricos o umbrales de entrada.
   - Ejecutes cada operación o comparación de reglas una por una en orden secuencial.
   - Verifiques que no exista error aritmético ni de lectura de umbrales.
2. Al terminar el razonamiento, escribe ÚNICAMENTE el código o número solicitado dentro de [VEREDICTO_FINAL] y [/VEREDICTO_FINAL].
""".strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ejecución de la Matriz Comparativa (`3 Problemas x 3 Técnicas`)

# COMMAND ----------

filas_evaluacion = []

for prob in PROBLEMAS:
    esperado = prob["esperado"]
    
    sal_zero = invocar_llm(prompt_zeroshot(prob["enunciado"]), max_tokens=80)
    ver_zero = extraer_veredicto(sal_zero)
    
    sal_few = invocar_llm(prompt_fewshot(prob["enunciado"]), max_tokens=120)
    ver_few = extraer_veredicto(sal_few)
    
    sal_cot = invocar_llm(prompt_cot_scratchpad(prob["enunciado"]), max_tokens=450)
    ver_cot = extraer_veredicto(sal_cot)
    
    filas_evaluacion.append({
        "Problema": prob["id"],
        "Esperado": esperado,
        "Zero-Shot": ver_zero,
        "Few-Shot": ver_few,
        "CoT + Scratchpad": ver_cot,
        "¿Zero OK?": esperado in ver_zero,
        "¿Few OK?": esperado in ver_few,
        "¿CoT OK?": esperado in ver_cot
    })

df_resultados = pd.DataFrame(filas_evaluacion)
display(spark.createDataFrame(df_resultados))

# COMMAND ----------

# Salida esperada:
# Tabla comparativa de los 3 problemas (P1_CLASIFICACION_DQ, P2_CALCULO_INVENTARIO, P3_LOGICA_REGLAS) mostrando cómo CoT + Scratchpad alcanza el 100% de exactitud en los 3 problemas, superando al enfoque Zero-Shot directo especialmente en el cálculo multi-etapa de inventario.
