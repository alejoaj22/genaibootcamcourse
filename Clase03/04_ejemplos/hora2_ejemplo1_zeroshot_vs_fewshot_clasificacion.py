# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 1: Zero-Shot vs. One-Shot vs. Few-Shot en Clasificación y Extracción
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Demostrar empíricamente cómo el aprendizaje en contexto (*In-Context Learning*) mediante ejemplos **Few-Shot balanceados** permite que un modelo rápido (`databricks-gpt-oss-20b`) adopte criterios de clasificación específicos de una organización y respete un formato de salida estricto sin necesidad de *fine-tuning*.
# MAGIC * **Tiempo estimado:** 8 minutos.
# MAGIC * **Concepto que ilustra:** Progresión Zero-Shot $\to$ One-Shot $\to$ Few-Shot, cobertura de casos borde y prevención del sesgo de etiquetas (*Label Bias*) y de recencia (*Recency Bias*).
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Conjunto de Casos de Prueba (Tickets de Incidentes de Plataforma de Datos)
# MAGIC Contamos con 4 tickets reales de operación tecnológica que presentan matices importantes: algunos parecen graves por el tono del usuario pero son solicitudes estéticas, mientras que otros parecen breves pero implican pérdida de datos o exposición de credenciales.

# COMMAND ----------

from typing import List, Dict, Any
import pandas as pd

MODELO_CLASIFICADOR = "databricks-gpt-oss-20b"

CASOS_EVALUACION: List[Dict[str, str]] = [
    {
        "id": "INC-101",
        "ticket": "¡URGENTE! ¡El botón de exportar a Excel en el dashboard de mercadeo se ve azul claro en vez de azul oscuro! ¡Necesito que lo cambien ya mismo!",
        "etiqueta_real": "P4_ESTETICO | COMPONENTE: BI_FRONTEND | ACCION: COLA_SPRINT"
    },
    {
        "id": "INC-102",
        "ticket": "Hola equipo, notamos que el job nocturno de ingesta de pagos duplicó los cargos de PSE entre las 02:00 y las 03:00 AM en la tabla Gold.",
        "etiqueta_real": "P1_CRITICO | COMPONENTE: ETL_FINANZAS | ACCION: ROLLBACK_INMEDIATO"
    },
    {
        "id": "INC-103",
        "ticket": "Un practicante subió por accidente un archivo .env con un token de lectura al repositorio interno de pruebas hace 10 minutos.",
        "etiqueta_real": "P1_CRITICO | COMPONENTE: SEGURIDAD_IAM | ACCION: REVOCAR_CREDENCIAL"
    },
    {
        "id": "INC-104",
        "ticket": "La consulta de histórico de inventario de 2022 tarda 45 segundos en vez de los 12 segundos habituales cuando filtramos por bodega norte.",
        "etiqueta_real": "P3_MENOR | COMPONENTE: OPTIMIZACION_SQL | ACCION: ANALISIS_ZORDER"
    }
]

def llamar_llm(prompt: str, modelo: str = MODELO_CLASIFICADOR) -> str:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_esc}',
        modelParameters => named_struct('temperature', CAST(0.0 AS DOUBLE), 'max_tokens', CAST(120 AS INT))
    ) AS respuesta
    """
    try:
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return res.strip() if res else ""

print(f"Cargados {len(CASOS_EVALUACION)} tickets de evaluación con su verdad terreno (Ground Truth).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Construcción de los 3 Niveles de Prompting: Zero-Shot, One-Shot y Few-Shot Balanceado
# MAGIC Observa cómo en el enfoque **Few-Shot** incluimos ejemplos que enseñan explícitamente al modelo a **no dejarse engañar por signos de exclamación o palabras como "¡URGENTE!"** cuando el problema real es cosmético.

# COMMAND ----------

def construir_prompt_zeroshot(ticket: str) -> str:
    return f"""
Clasifica el siguiente ticket de soporte técnico indicando su prioridad (P1_CRITICO, P2_ALTO, P3_MENOR, P4_ESTETICO), el componente afectado y la acción recomendada.
Sé muy breve, en una sola línea.

Ticket: "{ticket}"
Respuesta:
""".strip()

def construir_prompt_oneshot(ticket: str) -> str:
    return f"""
Clasifica el ticket técnico siguiendo exactamente la estructura del ejemplo proporcionado.

Ejemplo 1:
Ticket: "El conector de Salesforce dejó de sincronizar prospectos hace 2 horas por vencimiento de certificado."
Respuesta: P2_ALTO | COMPONENTE: INGESTA_CRM | ACCION: RENOVAR_CERTIFICADO

Ahora clasifica este caso:
Ticket: "{ticket}"
Respuesta:
""".strip()

def construir_prompt_fewshot(ticket: str) -> str:
    return f"""
Actúa como el Sistema de Triaje SRE de Ingeniería de Datos.
Clasifica cada ticket determinando la severidad real del impacto técnico (ignorando el tono emocional del usuario) y devuelve UNA SOLA LÍNEA con el formato exacto:
<PRIORIDAD> | COMPONENTE: <MODULO> | ACCION: <ACCION_TECNICA>

Valores permitidos de <PRIORIDAD>: P1_CRITICO, P2_ALTO, P3_MENOR, P4_ESTETICO
Valores permitidos de <MODULO>: BI_FRONTEND, ETL_FINANZAS, SEGURIDAD_IAM, OPTIMIZACION_SQL, INGESTA_CRM

### EJEMPLOS DE ENTRENAMIENTO EN CONTEXTO (FEW-SHOT BALANCEADO)

Ejemplo 1 (Falsa urgencia emocional - Caso Cosmético):
Ticket: "¡¡AUXILIO!! ¡El título del gráfico de ventas tiene la fuente Arial en vez de Roboto! ¡Arréglenlo ya!"
Respuesta: P4_ESTETICO | COMPONENTE: BI_FRONTEND | ACCION: COLA_SPRINT

Ejemplo 2 (Corrupción de datos financieros en producción):
Ticket: "Detectamos que el proceso de cierre contable multiplicó por dos las notas crédito de ayer en la tabla de facturación."
Respuesta: P1_CRITICO | COMPONENTE: ETL_FINANZAS | ACCION: ROLLBACK_INMEDIATO

Ejemplo 3 (Degradación moderada de rendimiento en consultas):
Ticket: "El reporte semanal de rotación por sucursal está demorando 30 segundos adicionales al agrupar por categoría."
Respuesta: P3_MENOR | COMPONENTE: OPTIMIZACION_SQL | ACCION: ANALISIS_ZORDER

Ejemplo 4 (Exposición de secretos o llaves de acceso):
Ticket: "Quedó expuesta una llave de servicio de lectura en un notebook compartido públicamente."
Respuesta: P1_CRITICO | COMPONENTE: SEGURIDAD_IAM | ACCION: REVOCAR_CREDENCIAL

### CASO A CLASIFICAR
Ticket: "{ticket}"
Respuesta:
""".strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ejecución Comparativa y Medición de Exactitud Estructural y Semántica

# COMMAND ----------

comparativa_filas = []

for caso in CASOS_EVALUACION:
    ticket = caso["ticket"]
    esperado = caso["etiqueta_real"]
    prioridad_esperada = esperado.split("|")[0].strip()
    
    out_zero = llamar_llm(construir_prompt_zeroshot(ticket))
    out_one = llamar_llm(construir_prompt_oneshot(ticket))
    out_few = llamar_llm(construir_prompt_fewshot(ticket))
    
    comparativa_filas.append({
        "ID": caso["id"],
        "Verdad Terreno (Esperado)": esperado,
        "Zero-Shot Output": out_zero,
        "One-Shot Output": out_one,
        "Few-Shot Output": out_few,
        "Acierto Prioridad (Zero)": prioridad_esperada in out_zero,
        "Acierto Exacto (Few-Shot)": (out_few.strip() == esperado)
    })

df_comparativa = pd.DataFrame(comparativa_filas)
display(spark.createDataFrame(df_comparativa))

# COMMAND ----------

# Salida esperada:
# 1. En el caso INC-101 ("¡URGENTE! ¡El botón... se ve azul claro!"), Zero-Shot suele engañarse por la palabra "¡URGENTE!" o produce un formato libre no estandarizado.
# 2. One-Shot estandariza la estructura `<PRIORIDAD> | COMPONENTE: ... | ACCION: ...`, pero aún puede dudar en la taxonomía exacta de módulos.
# 3. Few-Shot Balanceado logra 100% de coincidencia exacta (`Acierto Exacto (Few-Shot) = True`) en los 4 casos usando `databricks-gpt-oss-20b`.
