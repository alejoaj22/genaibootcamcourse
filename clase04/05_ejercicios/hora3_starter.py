# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejercicio (Starter): Eval Dataset y Comparativa A/B con LLM-as-a-Judge
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Construir un dataset de evaluación de 5 casos canónicos de dPhone (con al menos una pregunta no respondible), evaluar dos variantes de prompt (Prompt A básico vs Prompt B con XML y Grounding) mediante métricas cuantitativas (Keyword Match, ROUGE-L manual con LCS) y un evaluador LLM-as-a-Judge (`databricks-gpt-oss-120b`).
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Metodología Eval-Driven Development, sesgos del evaluador, evaluación multi-métrica y toma de decisiones para el Checkpoint 1 del Reto Integrador.
# MAGIC * **Requisitos previos:** Clúster o Serverless con acceso a `ai_query()` y modelos `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`.

# COMMAND ----------

import json
import re
import time
import unicodedata
from typing import Dict, Any, List
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"


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
               max_tokens: int = 800) -> dict:
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


def normalizar(texto: str) -> str:
    if not texto:
        return ""
    t = texto.lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


print("Infraestructura de evaluación inicializada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. TODO 1: Definir el Eval Dataset de 5 Casos sobre dPhone
# MAGIC
# MAGIC Completa la lista `EVAL_DATASET_5` con 5 preguntas:
# MAGIC - 4 preguntas respondibles con base en la documentación de dPhone (batería, chip Helios G1M, resistencia IP69, Fluid OS, etc.).
# MAGIC - 1 pregunta no respondible (ej. precio en pesos colombianos, gama de colores, fecha de dPhone D2).
# MAGIC - Cada caso debe incluir: `id`, `pregunta`, `referencia`, `palabras_clave` (lista), `es_respondible` (bool).

# COMMAND ----------

FRASE_ABSTENCION_CANONICA = "No encuentro esa información en la documentación proporcionada."

# TODO 1: Completa los 5 casos de evaluación
EVAL_DATASET_5 = [
    {
        "id": "Q1",
        "pregunta": "¿Qué capacidad de batería tiene el dPhone D1?",
        "referencia": "El dPhone D1 cuenta con una batería de 5,500 mAh.",
        "palabras_clave": ["5500", "bateria"],
        "es_respondible": True
    },
    # Agrega Q2, Q3, Q4 y Q5 (no respondible) aquí:
]

if len(EVAL_DATASET_5) < 5:
    print(f"Pendiente: completa el TODO 1. Actualmente hay {len(EVAL_DATASET_5)} de 5 casos.")
else:
    print(f"Eval dataset cargado con {len(EVAL_DATASET_5)} casos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Documentación Base y TODO 2: Variantes de Prompt A vs B

# COMMAND ----------

DOCS_DPHONE = """
Catálogo dPhone 2028:
- dPhone D1: Batería 5,500 mAh (60W carga, reemplazable por el usuario), procesador Helios G1M Neural Chip (5nm), pantalla 6.5" Super-AMOLED 1-144Hz, almacenamiento 512GB, certificación IP69 + Hydro-Guard, Fluid OS (100% compatible Android/iOS), precio 999 USD.
- dPhone D1 Pro: Batería 6,000 mAh (80W), Helios G1M Pro, pantalla 6.8", almacenamiento 1TB, cámara Halo de 200MP, precio 1,299 USD.
"""

def prompt_variante_a(pregunta: str) -> str:
    # TODO 2: Redacta un prompt básico sin delimitadores XML estrictos
    return f"Responde sobre dPhone: {pregunta}\nInfo: {DOCS_DPHONE}"


def prompt_variante_b(pregunta: str) -> str:
    # TODO 2: Redacta un prompt avanzado con etiquetas XML, grounding estricto y regla de abstención
    return f"""
<role>Asistente de dPhone</role>
<documents>{DOCS_DPHONE}</documents>
<rules>
Si la información no está en <documents>, responde exactamente: "{FRASE_ABSTENCION_CANONICA}"
</rules>
<question>{pregunta}</question>
""".strip()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. TODO 3: Implementación de ROUGE-L Manual con LCS (Subsecuencia Común Más Larga)

# COMMAND ----------

def lcs_longitud(x: List[str], y: List[str]) -> int:
    # TODO 3: Implementa el algoritmo de programación dinámica para LCS
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def calcular_rouge_l(candidato: str, referencia: str) -> float:
    # TODO 3: Calcula la métrica F1 de ROUGE-L basada en LCS
    cand_tokens = normalizar(candidato).split()
    ref_tokens = normalizar(referencia).split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_longitud(cand_tokens, ref_tokens)
    prec = lcs / len(cand_tokens)
    rec = lcs / len(ref_tokens)
    if (prec + rec) == 0:
        return 0.0
    return round((2 * prec * rec) / (prec + rec), 4)


print("Función ROUGE-L verificada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. TODO 4: Configuración del Evaluador LLM-as-a-Judge

# COMMAND ----------

def evaluar_con_juez(candidato: str, referencia: str) -> Dict[str, Any]:
    # TODO 4: Construye el prompt para el modelo 120b y devuelve puntaje (1-5)
    prompt_juez = f"""
    Eres un juez de calidad. Evalúa la respuesta:
    Referencia: "{referencia}"
    Candidato: "{candidato}"
    Devuelve un JSON con: "puntaje" (1 a 5) y "es_correcto" (bool).
    """
    res = llamar_llm(prompt_juez, modelo=MODELO_120B, temperature=0.0)
    try:
        limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", res["texto"].strip())
        return json.loads(limpio)
    except Exception:
        return {"puntaje": 3, "es_correcto": False}


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. TODO 5: Ejecución del Experimento y Tabla Comparativa

# COMMAND ----------

if len(EVAL_DATASET_5) < 5:
    print("Pendiente: completa el TODO 1 para ejecutar la evaluación comparativa.")
else:
    resultados = []
    print("Ejecutando evaluación de las 5 preguntas...")
    for caso in EVAL_DATASET_5:
        p = caso["pregunta"]
        ref = caso["referencia"]
        
        # Inferencia Variante A
        resp_a = llamar_llm(prompt_variante_a(p), modelo=MODELO_20B)["texto"]
        rouge_a = calcular_rouge_l(resp_a, ref)
        juez_a = evaluar_con_juez(resp_a, ref)
        
        # Inferencia Variante B
        resp_b = llamar_llm(prompt_variante_b(p), modelo=MODELO_20B)["texto"]
        rouge_b = calcular_rouge_l(resp_b, ref)
        juez_b = evaluar_con_juez(resp_b, ref)
        
        resultados.append({
            "ID": caso["id"],
            "ROUGE_A": rouge_a,
            "ROUGE_B": rouge_b,
            "Puntaje_Juez_A": juez_a.get("puntaje"),
            "Puntaje_Juez_B": juez_b.get("puntaje"),
            "Resp_A": resp_a[:60] + "...",
            "Resp_B": resp_b[:60] + "..."
        })
        
    df_eval = pd.DataFrame(resultados)
    display(spark.createDataFrame(df_eval))

# COMMAND ----------

# Salida esperada:
# 1. Definición completa de los 5 casos de evaluación.
# 2. La Variante B supera a la Variante A en el caso no respondible (obtiene puntaje 5 del juez por abstención).
# 3. La tabla comparativa resume los resultados de ROUGE-L y del Juez 120b.
