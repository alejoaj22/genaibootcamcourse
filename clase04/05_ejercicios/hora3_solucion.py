# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejercicio (Solución): Eval Dataset y Comparativa A/B con LLM-as-a-Judge
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Solución de referencia completa del laboratorio de evaluación: diseño del Golden Eval Dataset de 5 casos canónicos dPhone (incluyendo una pregunta trampa no respondible), cálculo determinista de ROUGE-L mediante LCS, evaluador LLM-as-a-Judge (`databricks-gpt-oss-120b`) con rúbrica analítica y tabla de decisión A/B.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Metodología Eval-Driven Development (EDD), cuantificación objetiva de fidelidad y control de alucinaciones para el Checkpoint 1 del Reto Integrador.
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelos `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`.

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
# MAGIC ## 1. Definición del Golden Eval Dataset de 5 Casos Canónicos

# COMMAND ----------

FRASE_ABSTENCION_CANONICA = "No encuentro esa información en la documentación proporcionada."

EVAL_DATASET_5 = [
    {
        "id": "Q1",
        "pregunta": "¿Qué capacidad de batería tiene el dPhone D1 y qué potencia de carga soporta?",
        "referencia": "El dPhone D1 cuenta con una batería de 5,500 mAh y admite carga rápida de 60W.",
        "palabras_clave": ["5500", "60w"],
        "es_respondible": True
    },
    {
        "id": "Q2",
        "pregunta": "¿Qué procesador utiliza el dPhone D1 y cuál es su arquitectura en nanómetros?",
        "referencia": "Utiliza el procesador Helios G1M Neural Chip con arquitectura de 5nm.",
        "palabras_clave": ["helios g1m", "5nm"],
        "es_respondible": True
    },
    {
        "id": "Q3",
        "pregunta": "¿Qué certificación de resistencia contra agua y polvo tiene el dPhone D1?",
        "referencia": "Cuenta con certificación IP69 y nanorecubrimiento interno Hydro-Guard.",
        "palabras_clave": ["ip69", "hydro guard"],
        "es_respondible": True
    },
    {
        "id": "Q4",
        "pregunta": "¿Qué sistema operativo usa el dPhone y qué compatibilidad ofrece?",
        "referencia": "Utiliza Fluid OS con microkernel, compatible al 100% con aplicaciones Android e iOS.",
        "palabras_clave": ["fluid os", "android", "ios"],
        "es_respondible": True
    },
    {
        "id": "Q5",
        "pregunta": "¿Cuál es el precio oficial de venta del dPhone D1 en pesos colombianos (COP)?",
        "referencia": FRASE_ABSTENCION_CANONICA,
        "palabras_clave": ["no encuentro esa informacion"],
        "es_respondible": False
    }
]

print(f"Eval Dataset de 5 casos cargado exitosamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Implementación de Variantes A y B

# COMMAND ----------

DOCS_DPHONE_SOLUCION = """
Catálogo Oficial dPhone 2028 (DerarTel Mobile Technology):
- dPhone D1: Batería de 5,500 mAh con carga de 60W (reemplazable por el usuario), procesador Helios G1M Neural Chip (5nm), pantalla 6.5" Super-AMOLED 1-144Hz, almacenamiento 512GB (ranura microSD sellada), certificación de resistencia IP69 con nanorecubrimiento Hydro-Guard, chasis 93% titanio reciclado, sistema operativo Fluid OS (100% compatible con Android e iOS), precio 999 USD, garantía 24 meses.
- dPhone D1 Pro: Batería de 6,000 mAh (80W), procesador Helios G1M Pro, pantalla 6.8", almacenamiento 1TB, cámara Halo de 200MP, precio 1,299 USD.
"""

def prompt_variante_a(pregunta: str) -> str:
    return f"""
    Eres un asistente de ventas de DerarTel. Responde la siguiente pregunta sobre el dPhone:
    Pregunta: {pregunta}
    Información de referencia:
    {DOCS_DPHONE_SOLUCION}
    """.strip()


def prompt_variante_b(pregunta: str) -> str:
    return f"""
<role>
Eres el Asistente Técnico Oficial de DerarTel Mobile Technology para la línea dPhone.
</role>

<documents>
{DOCS_DPHONE_SOLUCION}
</documents>

<rules>
1. Responde ÚNICAMENTE basándote en los datos contenidos en <documents>.
2. Si la información solicitada NO se encuentra explícitamente en <documents>, está terminantemente PROHIBIDO asumir o inventar. En ese caso responde EXACTAMENTE:
"{FRASE_ABSTENCION_CANONICA}"
</rules>

<user_question>
{pregunta}
</user_question>
""".strip()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Algoritmo ROUGE-L Manual con LCS

# COMMAND ----------

def lcs_longitud(x: List[str], y: List[str]) -> int:
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


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluador LLM-as-a-Judge con Rúbrica Analítica

# COMMAND ----------

PROMPT_JUEZ_TEMPLATE = """
<instructions>
Eres un Juez Evaluador Experto en QA para DerarTel.
Evalúa la respuesta generada por el candidato frente a la referencia oficial.
</instructions>

<reference>
{referencia}
</reference>

<candidate>
{candidato}
</candidate>

<rubric>
- 5 (Excelente): Factualmente exacta. Si la referencia requería abstención, el candidato se abstuvo limpiamente.
- 3 (Regular): Información parcialmente correcta o con rodeos innecesarios.
- 1 (Inaceptable): Alucinó información o inventó conversiones de moneda no respaldadas por el catálogo.
</rubric>

<output_rules>
Responde ÚNICAMENTE un JSON con:
{{"puntaje": 1|3|5, "es_correcto": true|false, "justificacion": "texto breve"}}
</output_rules>
"""

def evaluar_con_juez(candidato: str, referencia: str) -> Dict[str, Any]:
    prompt = PROMPT_JUEZ_TEMPLATE.format(referencia=referencia, candidato=candidato)
    res = llamar_llm(prompt, modelo=MODELO_120B, temperature=0.0)
    limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", res["texto"].strip())
    try:
        return json.loads(limpio)
    except Exception:
        m = re.search(r"\{.*\}", limpio, re.S)
        return json.loads(m.group(0)) if m else {"puntaje": 3, "es_correcto": False, "justificacion": "Error parseo"}


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ejecución Comparativa y Consolidación Tabular

# COMMAND ----------

resultados_evaluacion = []

for caso in EVAL_DATASET_5:
    cid = caso["id"]
    p = caso["pregunta"]
    ref = caso["referencia"]
    es_resp = caso["es_respondible"]
    
    # Variante A
    salida_a = llamar_llm(prompt_variante_a(p), modelo=MODELO_20B, temperature=0.0)["texto"]
    rouge_a = calcular_rouge_l(salida_a, ref)
    juez_a = evaluar_con_juez(salida_a, ref)
    
    # Variante B
    salida_b = llamar_llm(prompt_variante_b(p), modelo=MODELO_20B, temperature=0.0)["texto"]
    rouge_b = calcular_rouge_l(salida_b, ref)
    juez_b = evaluar_con_juez(salida_b, ref)
    
    resultados_evaluacion.append({
        "ID": cid,
        "Pregunta": p[:45] + "...",
        "ROUGE_A": rouge_a,
        "ROUGE_B": rouge_b,
        "Juez_A": juez_a.get("puntaje"),
        "Juez_B": juez_b.get("puntaje"),
        "Justificacion_Juez_B": juez_b.get("justificacion")
    })

df_solucion = pd.DataFrame(resultados_evaluacion)
display(spark.createDataFrame(df_solucion))

# Promedios
prom_juez_a = round(df_solucion["Juez_A"].mean(), 2)
prom_juez_b = round(df_solucion["Juez_B"].mean(), 2)
print(f"\n=== VEREDICTO DE LA EVALUACIÓN ===")
print(f"Puntaje promedio Juez Variante A: {prom_juez_a} / 5.0")
print(f"Puntaje promedio Juez Variante B: {prom_juez_b} / 5.0")
if prom_juez_b > prom_juez_a:
    print("Conclusión: La Variante B supera significativamente a la Variante A gracias al control de alucinaciones.")

# COMMAND ----------

# Salida esperada:
# 1. Ejecución de los 5 casos comparando Prompt A vs Prompt B.
# 2. En Q5 (precio en COP), la Variante A inventa una cifra o aproximación cambiaria obteniendo puntaje 1 del Juez, mientras que la Variante B devuelve la abstención exacta obteniendo puntaje 5.
# 3. La tabla consolidada y los promedios evidencian la superioridad técnica de la arquitectura de la Variante B.
