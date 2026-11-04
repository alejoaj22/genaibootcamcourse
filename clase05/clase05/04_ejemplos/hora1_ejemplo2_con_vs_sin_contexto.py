# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 2 — Respuestas CON vs SIN contexto externo
# MAGIC **Objetivo:** demostrar con evidencia por qué un LLM sin contexto alucina sobre datos privados de dPhone y cómo cambia con contexto.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** limitaciones del prompting sin contexto y la idea central de RAG (Pregunta + Evidencia → Respuesta).
# MAGIC **Requisitos previos:** haber ejecutado `Includes/Lab_Setup`; endpoint `databricks-gpt-oss-20b` disponible.

# COMMAND ----------

# MAGIC %run ../Includes/Lab_Setup

# COMMAND ----------

# Función auxiliar: llama a un LLM de Databricks con ai_query desde Python.
# Escapamos comillas simples y barras para poder incrustar el prompt en SQL.
ENDPOINT = "databricks-gpt-oss-20b"

def preguntar(prompt: str, endpoint: str = ENDPOINT) -> str:
    seguro = prompt.replace("\\", "\\\\").replace("'", "\\'")
    consulta = f"""
        SELECT ai_query('{endpoint}', '{seguro}',
                        responseFormat => '{{"type":"text"}}') AS respuesta
    """
    return spark.sql(consulta).collect()[0]["respuesta"]

# COMMAND ----------

# Pregunta sobre un dato interno que el modelo NO puede conocer.
pregunta = "¿Cuántos meses de garantía tiene el dPhone y qué cubre exactamente la garantía?"

# 1) SIN contexto: el modelo responde solo con su memoria paramétrica.
respuesta_sin_contexto = preguntar(pregunta + " Responde en máximo 3 frases.")
print("=== SIN CONTEXTO ===")
print(respuesta_sin_contexto)

# COMMAND ----------

# 2) CON contexto: simulamos lo que haría el retriever de RAG insertando un fragmento.
# NOTA: este fragmento es ILUSTRATIVO (ficticio) para la demo; en el pipeline real
# el contexto proviene de los documentos dPhone parseados en Silver.
contexto = """[DOC-01] Política de garantía dPhone (fragmento de ejemplo):
La garantía estándar del dPhone es de 12 meses desde la fecha de compra.
Cubre defectos de fabricación de hardware. No cubre daños por agua,
golpes ni uso de accesorios no certificados."""

prompt_rag = f"""Eres un asistente de soporte de dPhone.
Responde SOLO con base en el contexto. Si la información no está en el contexto, di "No tengo esa información".
Cita la fuente entre corchetes.

<contexto>
{contexto}
</contexto>

<pregunta>
{pregunta}
</pregunta>"""

respuesta_con_contexto = preguntar(prompt_rag)
print("=== CON CONTEXTO ===")
print(respuesta_con_contexto)

# COMMAND ----------

# 3) Pregunta fuera del contexto: el modelo debe abstenerse (grounding estricto).
pregunta_fuera = "¿Cuál es el precio del dPhone en Colombia?"
prompt_fuera = prompt_rag.replace(pregunta, pregunta_fuera)
print("=== CON CONTEXTO, PREGUNTA FUERA DE ALCANCE ===")
print(preguntar(prompt_fuera))

# COMMAND ----------

# Comparación tabular de las tres respuestas para discusión en clase.
filas = [
    ("Sin contexto", pregunta, respuesta_sin_contexto),
    ("Con contexto", pregunta, respuesta_con_contexto),
    ("Fuera de alcance", pregunta_fuera, preguntar(prompt_fuera)),
]
display(spark.createDataFrame(filas, ["escenario", "pregunta", "respuesta"]))

# COMMAND ----------

# Salida esperada:
# - SIN CONTEXTO: respuesta plausible con cifras (por ejemplo "12 o 24 meses") sin fuente verificable.
# - CON CONTEXTO: "12 meses ... cubre defectos de fabricación ... [DOC-01]".
# - FUERA DE ALCANCE: "No tengo esa información".
# - Una tabla de 3 filas. El texto exacto varía entre ejecuciones (los LLM no son deterministas).
