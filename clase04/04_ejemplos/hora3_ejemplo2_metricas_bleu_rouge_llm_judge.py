# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejemplo 2: Evaluación Sistemática: Métricas Determinísticas (BLEU, ROUGE, F1) vs LLM-as-a-Judge
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Comparar empíricamente las métricas léxicas tradicionales de NLP (BLEU, ROUGE-1, ROUGE-2, ROUGE-L, Token F1 y Exact Match) frente al paradigma moderno de **LLM-as-a-Judge** (usando `databricks-gpt-oss-120b` con rúbrica estructurada) sobre 6 pares de respuestas técnicas de dPhone, evidenciando las fallas críticas del solapamiento n-grama ante paráfrasis y negaciones, e identificando los sesgos del juez LLM y sus mitigaciones.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Límites de las métricas de coincidencia de tokens, diseño de rúbricas analíticas para LLMs evaluadores, sesgo de posición (*position bias*), sesgo de verbosidad (*verbosity bias*), sesgo de autopreferencia y panorámica de benchmarks de la industria (MMLU, MT-Bench, Chatbot Arena).
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-120b`.

# COMMAND ----------

# MAGIC %pip install sacrebleu rouge-score

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura de Invocación y Helper Estándar

# COMMAND ----------

import json
import re
import string
import time
import unicodedata
from typing import Dict, Any, List, Tuple
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer, scoring

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


def llamar_llm(prompt: str, modelo: str = MODELO_120B, temperature: float = 0.0,
               max_tokens: int = 1000) -> dict:
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


print("Entorno de evaluación inicializado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Implementación de Tokenizer en Español y Métricas Léxicas (F1 y Exact Match)

# COMMAND ----------

def normalizar_texto(texto: str) -> str:
    """Elimina tildes, signos de puntuación y pasa a minúsculas."""
    if not texto:
        return ""
    texto = texto.lower()
    # Descomponer caracteres acentuados
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # Eliminar signos de puntuación
    texto = re.sub(r"[^\w\s]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def calcular_exact_match(candidato: str, referencia: str) -> float:
    return 1.0 if normalizar_texto(candidato) == normalizar_texto(referencia) else 0.0


def calcular_token_f1(candidato: str, referencia: str) -> float:
    tokens_cand = normalizar_texto(candidato).split()
    tokens_ref = normalizar_texto(referencia).split()
    if not tokens_cand or not tokens_ref:
        return 0.0
    
    interseccion = set(tokens_cand) & set(tokens_ref)
    if not interseccion:
        return 0.0
        
    prec = len(interseccion) / len(tokens_cand)
    rec = len(interseccion) / len(tokens_ref)
    return round((2 * prec * rec) / (prec + rec), 4)


# Tokenizer custom para ROUGE compatible con español
class SpanishTokenizer(scoring.BaseTokenizer):
    def tokenize(self, text: str) -> List[str]:
        return normalizar_texto(text).split()


scorer_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=SpanishTokenizer())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Banco de Evaluación: 6 Casos de Estudio dPhone
# MAGIC Incluimos intencionalmente:
# MAGIC - Caso 1: Coincidencia casi textual.
# MAGIC - Caso 2: **Paráfrasis impecable** (mismo significado, vocabulario diferente) -> *Falla esperada en BLEU*.
# MAGIC - Caso 3: **Negación catastrófica** (alta coincidencia de n-gramas pero significado opuesto) -> *BLEU alto pero falsamente positivo*.
# MAGIC - Caso 4: Alucinación de cifras.
# MAGIC - Caso 5: Abstención correcta.
# MAGIC - Caso 6: Respuesta con exceso de verborrea irrelevante.

# COMMAND ----------

CASOS_EVALUACION = [
    {
        "id": "CASO-01",
        "descripcion": "Coincidencia Casi Idéntica",
        "referencia": "El dPhone D1 tiene una batería de 5,500 mAh con soporte de carga rápida de 60W.",
        "candidato": "El dPhone D1 cuenta con una batería de 5,500 mAh y admite carga rápida a 60W."
    },
    {
        "id": "CASO-02",
        "descripcion": "Paráfrasis Correcta (Vocabulario Distinto)",
        "referencia": "El procesador Helios G1M Neural Chip está fabricado en tecnología de 5 nanómetros.",
        "candidato": "DerarTel integró un microprocesador de última generación con litografía de 5nm denominado Helios G1M."
    },
    {
        "id": "CASO-03",
        "descripcion": "Negación Catastrófica (Falso Positivo de N-Gramas)",
        "referencia": "El dPhone D1 tiene una batería de 5,500 mAh.",
        "candidato": "El dPhone D1 NO tiene una batería de 5,500 mAh."
    },
    {
        "id": "CASO-04",
        "descripcion": "Alucinación de Cifra Crítica",
        "referencia": "El precio de lanzamiento del dPhone D1 Pro es de 1,299 USD.",
        "candidato": "El dPhone D1 Pro cuesta 499 USD en su versión básica."
    },
    {
        "id": "CASO-05",
        "descripcion": "Abstención Exacta",
        "referencia": "No encuentro esa información en la documentación proporcionada.",
        "candidato": "No encuentro esa información en la documentación proporcionada."
    },
    {
        "id": "CASO-06",
        "descripcion": "Verborrea Irrelevante",
        "referencia": "La certificación de resistencia del dPhone D1 es IP69.",
        "candidato": "El dPhone D1 es un smartphone moderno desarrollado por DerarTel en 2028 y cuenta con una certificación IP69 junto con otros detalles estéticos y tecnológicos avanzados."
    }
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cálculo de Métricas Determinísticas (Léxicas)

# COMMAND ----------

resultados_lexicos = []

for c in CASOS_EVALUACION:
    cand = c["candidato"]
    ref = c["referencia"]
    
    # BLEU con sacrebleu
    score_bleu = round(sacrebleu.sentence_bleu(cand, [ref]).score, 2)
    
    # ROUGE
    r_scores = scorer_rouge.score(ref, cand)
    rouge_1 = round(r_scores["rouge1"].fmeasure * 100, 2)
    rouge_2 = round(r_scores["rouge2"].fmeasure * 100, 2)
    rouge_l = round(r_scores["rougeL"].fmeasure * 100, 2)
    
    f1 = round(calcular_token_f1(cand, ref) * 100, 2)
    em = calcular_exact_match(cand, ref)
    
    resultados_lexicos.append({
        "ID": c["id"],
        "Descripcion": c["descripcion"],
        "BLEU": score_bleu,
        "ROUGE_L": rouge_l,
        "Token_F1": f1,
        "Exact_Match": em,
        "Candidato": cand[:60] + "..."
    })

df_lexicos = pd.DataFrame(resultados_lexicos)
display(spark.createDataFrame(df_lexicos))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Implementación de LLM-as-a-Judge con Rúbrica Estructurada
# MAGIC Utilizamos `databricks-gpt-oss-120b` como juez evaluador. Le proporcionamos la referencia de oro (*Ground Truth*), el texto del candidato y una rúbrica de 1 a 5 puntos que evalúa **Fidelidad Factual**, **Corrección Semántica** y **Concisión**.

# COMMAND ----------

PROMPT_JUEZ_TEMPLATE = """
<instructions>
Eres un Juez Evaluador Experto en Sistemas de Pregunta-Respuesta (QA Evaluation Judge).
Evalúa la respuesta generada por un modelo (CANDIDATO) contra la respuesta oficial de referencia (GROUND_TRUTH).
</instructions>

<ground_truth>
{referencia}
</ground_truth>

<candidate_response>
{candidato}
</candidate_response>

<rubric>
Califica del 1 al 5 en escala entera:
- 5 (Excelente): Factualmente idéntica a la referencia, aunque use otras palabras. Sin contradicciones.
- 4 (Buena): Factualmente correcta, pero incluye leves redundancias o rodeos.
- 3 (Aceptable con reservas): Contiene parte de la verdad pero omite detalles esenciales.
- 2 (Deficiente): Contradice la referencia o alucina hechos críticos (ej. niega un hecho o cambia un número).
- 1 (Catastrófica / Inaceptable): Completamente errónea, alucinación total o invento sin fundamento.
</rubric>

<output_rules>
Responde ÚNICAMENTE un objeto JSON plano con las claves:
- "puntaje": entero entre 1 y 5
- "es_factualmente_correcto": booleano (true/false)
- "justificacion": string breve en español explicando la discrepancia
</output_rules>
"""

resultados_juez = []

for c in CASOS_EVALUACION:
    prompt_juez = PROMPT_JUEZ_TEMPLATE.format(referencia=c["referencia"], candidato=c["candidato"])
    res = llamar_llm(prompt_juez, modelo=MODELO_120B, temperature=0.0)
    
    texto_juez = re.sub(r"^```(?:json)?\s*|\s*```$", "", res["texto"].strip())
    try:
        eval_dict = json.loads(texto_juez)
    except Exception:
        m = re.search(r"\{.*\}", texto_juez, re.S)
        eval_dict = json.loads(m.group(0)) if m else {"puntaje": 0, "es_factualmente_correcto": False, "justificacion": "Error parseo juez"}
        
    resultados_juez.append({
        "ID": c["id"],
        "Puntaje_Juez": eval_dict.get("puntaje"),
        "Factualmente_Correcto": eval_dict.get("es_factualmente_correcto"),
        "Justificacion_Juez": eval_dict.get("justificacion")
    })

df_juez = pd.DataFrame(resultados_juez)
df_consolidado = pd.merge(df_lexicos, df_juez, on="ID")
display(spark.createDataFrame(df_consolidado))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Comparativa Crítica: ¿Por qué BLEU/ROUGE no bastan y cómo mitigar sesgos del Juez?
# MAGIC
# MAGIC 1. **La Paradoja de la Negación (Caso 3):**
# MAGIC    - BLEU/ROUGE otorgan puntajes altos (~80%) porque comparten casi todas las palabras.
# MAGIC    - El Juez LLM otorga **Puntaje 2** detectando la contradicción fáctica inmediata.
# MAGIC 2. **La Paradoja de la Paráfrasis (Caso 2):**
# MAGIC    - BLEU castiga la paráfrasis (puntaje bajo) porque no coinciden n-gramas.
# MAGIC    - El Juez LLM otorga **Puntaje 5** reconociendo equivalencia semántica total.
# MAGIC 3. **Sesgos Conocidos del LLM-as-a-Judge y sus Mitigaciones:**
# MAGIC    - **Position Bias (Sesgo de Posición):** En comparaciones por pares (A vs B), el modelo tiende a favorecer la primera opción. *Mitigación:* Evaluar (A, B) y (B, A) e invertir la asignación.
# MAGIC    - **Verbosity Bias (Sesgo de Verbosidad):** Tiende a calificar mejor respuestas largas y detalladas aunque contengan paja. *Mitigación:* Penalizar explícitamente en la rúbrica la información innecesaria.
# MAGIC    - **Self-Enhancement Bias (Autopreferencia):** Prefiere textos generados por su propia familia arquitectónica. *Mitigación:* Usar un juez de tamaño y familia superior (ej. `databricks-gpt-oss-120b`).
# MAGIC 4. **Benchmarks Globales vs Eval Datasets Propios:**
# MAGIC    - MMLU / MT-Bench / Chatbot Arena miden capacidades generales de razonamiento.
# MAGIC    - En proyectos de datos empresariales, **los benchmarks públicos no garantizan que el modelo conozca el catálogo de dPhone**. La métrica de oro es tu propio **Golden Eval Dataset**.

# COMMAND ----------

# Salida esperada:
# 1. Tabla comparativa mostrando cómo BLEU falla estrepitosamente en CASO-02 (paráfrasis correcta calificada baja) y CASO-03 (negación incorrecta calificada alta).
# 2. El evaluador LLM-as-a-Judge califica con precisión semántica (5 en paráfrasis, 2 en negación) y emite justificaciones cualitativas.
# 3. Documentación clara de sesgos del evaluador y su impacto en sistemas de producción.
