# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 2: Chain-of-Thought (CoT), Scratchpad y Self-Consistency
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Demostrar cómo **Chain-of-Thought (CoT)** combinado con el patrón **Scratchpad** y **Self-Consistency (votación por mayoría)** resuelve problemas de lógica de negocio y conciliación numérica donde el prompting directo (sin razonamiento intermedio) falla.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Expansión del presupuesto computacional por token mediante pasos intermedios explícitos (`[SCRATCHPAD]`), extracción determinista de `[MONTO_FINAL]` y ensamblaje estocástico por consenso (*Self-Consistency*).
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. El Problema de Conciliación de Costos Cloud y SLA
# MAGIC Planteamos un problema realista de ingeniería de datos y finanzas tecnológicas (*FinOps*) que involucra múltiples condiciones encadenadas, conversiones de unidades y penalizaciones porcentuales.
# MAGIC
# MAGIC **Cálculo manual de la verdad terreno (Ground Truth):**
# MAGIC 1. Consumo base de cómputo: $400 \text{ DBUs} \times \$0.50 = \$200.00$.
# MAGIC 2. Consumo de inferencia IA: $150 \text{ DBUs} \times \$0.80 = \$120.00$.
# MAGIC 3. Subtotal bruto: $\$200.00 + \$120.00 = \$320.00$.
# MAGIC 4. Como el total de DBUs ($400 + 150 = 550$) supera las $500 \text{ DBUs}$, aplica descuento por volumen del $10\%$ sobre el subtotal bruto: $\$320.00 \times 0.90 = \$288.00$.
# MAGIC 5. Tiempo total del pipeline nocturno: Ingesta ($8\text{ min}$) + Transformación ($540\text{ seg} = 9\text{ min}$) + Publicación ($300\text{ seg} = 5\text{ min}$) = $22\text{ minutos}$.
# MAGIC 6. Como $22\text{ min} > 20\text{ min}$ (SLA pactado), el proveedor otorga un crédito de compensación de $\$18.00$.
# MAGIC 7. **Total Neto a Pagar:** $\$288.00 - \$18.00 = \$270.00$.

# COMMAND ----------

import re
from collections import Counter
from typing import Dict, Any, List
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"
RESPUESTA_CORRECTA_USD = "270.00"

ENUNCIADO_PROBLEMA = """
Una empresa ejecutó su pipeline de datos en la nube con las siguientes métricas del día:
- Cómputo ETL consumido: 400 DBUs a una tarifa de $0.50 USD por DBU.
- Cómputo de Inferencia IA consumido: 150 DBUs a una tarifa de $0.80 USD por DBU.
- Regla de Descuento por Volumen: Si la suma total de DBUs del día (ETL + IA) es estrictamente mayor a 500 DBUs, se aplica un 10% de descuento sobre el subtotal combinado en dólares.
- Regla de SLA Operativo: El SLA máximo permitido para el pipeline completo es de 20 minutos. Si la duración total supera los 20 minutos, se descuenta un crédito fijo adicional de $18.00 USD después de haber aplicado el descuento por volumen.
- Tiempos registrados hoy: Etapa Ingesta = 8 minutos; Etapa Transformación = 540 segundos; Etapa Publicación = 300 segundos.

¿Cuál es el monto neto final en USD que debe pagar la empresa hoy (con dos decimales)?
""".strip()

def consultar_llm(prompt: str, modelo: str = MODELO_20B, temp: float = 0.0, max_tokens: int = 450) -> str:
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

print("Problema de conciliación FinOps configurado. Respuesta esperada: 270.00")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Comparación: Respuesta Directa (Sin CoT) vs. Scratchpad Chain-of-Thought
# MAGIC En el primer prompt prohibimos cualquier explicación intermedia para forzar al modelo a responder en el primer token. En el segundo prompt habilitamos el bloque `[SCRATCHPAD]` antes de emitir `[MONTO_FINAL]`.

# COMMAND ----------

prompt_sin_cot = f"""
{ENUNCIADO_PROBLEMA}

RESTRICCIÓN ESTRICTA: Prohibido explicar o mostrar pasos. Responde ÚNICAMENTE con el número final con dos decimales (ejemplo: 123.45).
""".strip()

prompt_con_scratchpad_cot = f"""
{ENUNCIADO_PROBLEMA}

INSTRUCCIONES DE RAZONAMIENTO (CHAIN-OF-THOUGHT):
1. Dentro de las etiquetas [SCRATCHPAD] y [/SCRATCHPAD], piensa paso a paso:
   - Paso A: Calcula el costo de ETL, el costo de IA y el subtotal bruto.
   - Paso B: Suma las DBUs totales y verifica si aplica el 10% de descuento por volumen.
   - Paso C: Convierte todos los tiempos de segundos a minutos, súmalos y verifica si se violó el SLA de 20 minutos.
   - Paso D: Resta el crédito de SLA si corresponde.
2. Al final, escribe el resultado numérico con dos decimales dentro de [MONTO_FINAL] y [/MONTO_FINAL].
""".strip()

def extraer_scratchpad_y_monto(texto: str) -> Dict[str, str]:
    bloque_scratch = re.search(r"\[SCRATCHPAD\](.*?)\[/SCRATCHPAD\]", texto, re.DOTALL)
    bloque_monto = re.search(r"\[MONTO_FINAL\](.*?)\[/MONTO_FINAL\]", texto, re.DOTALL)
    monto_limpio = bloque_monto.group(1).strip().replace("$", "").replace("USD", "").strip() if bloque_monto else texto.strip()
    match_num = re.search(r"\d+(?:\.\d+)?", monto_limpio)
    valor_num = f"{float(match_num.group(0)):.2f}" if match_num else monto_limpio
    return {
        "scratchpad": bloque_scratch.group(1).strip() if bloque_scratch else "(Sin Scratchpad)",
        "monto_extraido": valor_num
    }

salida_directa_20b = consultar_llm(prompt_sin_cot, modelo=MODELO_20B, temp=0.0, max_tokens=30)
salida_cot_20b = consultar_llm(prompt_con_scratchpad_cot, modelo=MODELO_20B, temp=0.0, max_tokens=500)
datos_cot_20b = extraer_scratchpad_y_monto(salida_cot_20b)

print("=== 1. SALIDA DIRECTA SIN COT (Forzando respuesta en el primer token) ===")
print(f"Respuesta cruda: {salida_directa_20b} | ¿Es igual a {RESPUESTA_CORRECTA_USD}? -> {RESPUESTA_CORRECTA_USD in salida_directa_20b}")

print("\n=== 2. SALIDA CON SCRATCHPAD CHAIN-OF-THOUGHT ===")
print("--- Borrador de Razonamiento Auditable [SCRATCHPAD] ---")
print(datos_cot_20b["scratchpad"])
print("-" * 60)
print(f"Monto Final Extraído: {datos_cot_20b['monto_extraido']} | ¿Exacto ({RESPUESTA_CORRECTA_USD})? -> {datos_cot_20b['monto_extraido'] == RESPUESTA_CORRECTA_USD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Implementación de Self-Consistency (Votación por Mayoría sobre Caminos CoT)
# MAGIC Para decisiones financieras críticas, ejecutamos $N = 5$ cadenas de pensamiento independientes con `temperature = 0.5` y tomamos el consenso por mayoría (*Majority Voting*).

# COMMAND ----------

N_CAMINOS = 5
caminos_registrados: List[Dict[str, Any]] = []
votos: List[str] = []

for i in range(1, N_CAMINOS + 1):
    respuesta_camino = consultar_llm(prompt_con_scratchpad_cot, modelo=MODELO_120B, temp=0.5, max_tokens=500)
    extraido = extraer_scratchpad_y_monto(respuesta_camino)
    votos.append(extraido["monto_extraido"])
    caminos_registrados.append({
        "Camino_ID": f"Ruta_CoT_{i}",
        "Monto_Calculado": extraido["monto_extraido"],
        "¿Correcto?": extraido["monto_extraido"] == RESPUESTA_CORRECTA_USD,
        "Extracto_Razonamiento": extraido["scratchpad"][:130].replace("\n", " ")
    })

conteo_votos = Counter(votos)
ganador_consenso, votos_ganador = conteo_votos.most_common(1)[0]
confianza_consenso = round((votos_ganador / N_CAMINOS) * 100, 1)

print(f"=== RESULTADO SELF-CONSISTENCY ({N_CAMINOS} CAMINOS INDEPENDIENTES) ===")
print(f"Distribución de votos: {dict(conteo_votos)}")
print(f"Veredicto por Mayoría: ${ganador_consenso} USD (Confianza del ensemble: {confianza_consenso}%)")

display(spark.createDataFrame(pd.DataFrame(caminos_registrados)))

# COMMAND ----------

# Salida esperada:
# 1. La respuesta directa sin CoT suele equivocarse (por ejemplo respondiendo 288.00 o 302.00 al omitir la conversión de segundos a minutos).
# 2. La salida con `[SCRATCHPAD]` desglosa explícitamente los pasos A, B, C y D y extrae `270.00` en `[MONTO_FINAL]`.
# 3. La tabla de Self-Consistency muestra los 5 caminos de razonamiento y el veredicto por consenso en `$270.00 USD`.
