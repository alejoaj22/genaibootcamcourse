# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 — Ejemplo 2: Grounding Riguroso sobre Documentación Técnica dPhone y Control de Alucinaciones
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Implementar una técnica de **grounding estricto (anclaje documental)** que fuerce al modelo a responder exclusivamente con la información contenida en `<documents>`, exigiendo citas de fuentes y forzando la abstención explícita ante preguntas no cubiertas.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Reducción de alucinaciones mediante contratos de fidelidad, citas por identificador de documento (`[DOC-X]`), política de abstención determinista ("quote first, then answer") y medición automática de tasa de alucinación frente a preguntas trampa.
# MAGIC * **Requisitos previos:** Clúster o Serverless en Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura de Invocación y Helper Estándar

# COMMAND ----------

import json
import re
import time
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"


def _esc(texto: str) -> str:
    """Escapa texto para incrustarlo como literal SQL."""
    return texto.replace("\\", "\\\\").replace("'", "\\'")


def extraer_texto(salida) -> str:
    """Extrae texto limpio de salidas con bloques reasoning/text de Databricks."""
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
               max_tokens: int = 1000) -> dict:
    """Ejecuta ai_query en Databricks SQL."""
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
        "latencia_s": round(time.time() - t0, 2),
        "modelo": modelo
    }


print("Infraestructura de inferencia inicializada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Base de Conocimiento Canónica: Documentación Oficial de dPhone (DerarTel)
# MAGIC En la Clase 5 automatizaremos la búsqueda vectorial con Databricks Vector Search (RAG). Hoy inyectamos el contexto de manera explícita y delimitada en XML para dominar el anclaje cognitivo del LLM.

# COMMAND ----------

DOCUMENTOS_DPHONE_XML = """<documents>
  <document id="DOC-01" titulo="Ficha Técnica dPhone D1">
    Fabricante: DerarTel Mobile Technology.
    Lanzamiento oficial: 28 de marzo de 2028.
    Procesador: Helios G1M Neural Chip con arquitectura de 5nm.
    Pantalla: 6.5 pulgadas Super-AMOLED Infinity Display, refresco dinámico 1Hz a 144Hz.
    Almacenamiento: 512GB UFS 4.1. Memoria expandible mediante ranura dual-purpose microSD sellada.
    Batería: 5,500 mAh con soporte de carga rápida de 60W por cable. Batería reemplazable por el usuario.
    Seguridad biométrica: Vein-Scan (escáner vascular infrarrojo bajo pantalla).
    Conectividad: 6G Ready y enlace satelital directo de emergencia (Satellite SOS).
    Resistencia ambiental: Certificación IP69 (a prueba de chorros de agua a alta presión y temperatura) y nanorecubrimiento interno Hydro-Guard.
    Materiales: Chasis unibody construido en 93% de titanio reciclado.
    Sistema Operativo: Fluid OS basado en microkernel, compatible al 100% con aplicaciones Android e iOS.
    Precio oficial: 999 USD. Garantía de fábrica: 24 meses.
    Benchmarks de rendimiento (escala 0-100): Procesamiento de IA: 98 (vs 85 competencia), Duración de batería: 95, Eficiencia térmica: 92, Estabilidad de señal: 94.
  </document>
  <document id="DOC-02" titulo="Línea de Modelos dPhone 2028">
    dPhone D1 Lite: Pantalla 6.1 pulgadas AMOLED 120Hz, chip Helios G1 Lite (6nm), 256GB almacenamiento, batería 4,500 mAh con carga de 45W, resistencia IP68. Precio: 699 USD.
    dPhone D1 Pro: Pantalla 6.8 pulgadas Super-AMOLED 1-144Hz, chip Helios G1M Pro, 1TB almacenamiento, batería 6,000 mAh con carga ultra de 80W, cámara Halo de 200MP con sensor periscópico, resistencia IP69 con Hydro-Guard. Precio: 1,299 USD.
    dPhone D1 Fold: Pantalla plegable interna de 7.9 pulgadas + pantalla externa de 6.3 pulgadas AMOLED, chip Helios G1M, 512GB almacenamiento, batería dual de 5,000 mAh con carga de 60W, resistencia IPX8. Precio: 1,899 USD.
  </document>
</documents>""".strip()

print(f"Base de conocimiento cargada ({len(DOCUMENTOS_DPHONE_XML)} caracteres).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Banco de Evaluación: Preguntas Respondibles vs Preguntas Trampa (No Respondibles)
# MAGIC Para medir objetivamente el grounding necesitamos evaluar dos tipos de preguntas:
# MAGIC 1. **Respondibles:** Información presente explícitamente en el texto.
# MAGIC 2. **No Respondibles / Fuera de Dominio:** Datos plausibles que el modelo tendería a "inventar" si no está debidamente anclado (p. ej. precio en COP, fecha de lanzamiento de dPhone D2, colores disponibles).

# COMMAND ----------

BANCO_PREGUNTAS = [
    {
        "id": "P1",
        "tipo": "respondible",
        "pregunta": "¿Qué capacidad de batería tiene el dPhone D1 y permite reemplazo por el usuario?",
        "hecho_clave": "5,500 mAh"
    },
    {
        "id": "P2",
        "tipo": "respondible",
        "pregunta": "¿Cuál es la resolución y nombre de la cámara del dPhone D1 Pro?",
        "hecho_clave": "Halo de 200MP"
    },
    {
        "id": "P3",
        "tipo": "respondible",
        "pregunta": "¿Qué certificación de resistencia tiene el dPhone D1 contra agua y polvo?",
        "hecho_clave": "IP69"
    },
    {
        "id": "P4",
        "tipo": "respondible",
        "pregunta": "¿Qué sistema operativo usa el dPhone y qué compatibilidad tiene?",
        "hecho_clave": "Fluid OS"
    },
    {
        "id": "P5",
        "tipo": "no_respondible",
        "pregunta": "¿Cuál es el precio oficial del dPhone D1 en pesos colombianos (COP)?",
        "hecho_clave": "ABSTENCION"
    },
    {
        "id": "P6",
        "tipo": "no_respondible",
        "pregunta": "¿En qué colores viene disponible el dPhone D1 en tiendas físicas?",
        "hecho_clave": "ABSTENCION"
    }
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Experimento: Prompt Ingenuo (Sin Grounding) vs Prompt Estructurado con Grounding

# COMMAND ----------

FRASE_ABSTENCION_CANONICA = "No encuentro esa información en la documentación proporcionada."


def construir_prompt_ingenuo(pregunta: str) -> str:
    """Prompt sin reglas estrictas de anclaje ni delimitadores."""
    return f"""
    Eres un asistente de ventas de DerarTel. Responde la siguiente pregunta sobre el dPhone:
    Pregunta: {pregunta}
    """.strip()


def construir_prompt_grounded(pregunta: str) -> str:
    """Prompt con anclaje riguroso, etiquetas XML, citas obligatorias y cláusula de abstención."""
    return f"""
<instructions>
Eres un Asistente Técnico Especializado de dPhone (DerarTel Mobile Technology).
Tu única fuente de verdad es la información delimitada en la etiqueta <documents>.
</instructions>

{DOCUMENTOS_DPHONE_XML}

<rules>
1. Responde ÚNICAMENTE fundamentándote en los hechos declarados en <documents>.
2. En cada afirmación, incluye la cita explícita del documento de origen entre corchetes, por ejemplo [DOC-01] o [DOC-02].
3. REGLA DE ABSTENCIÓN ESTRICTA: Si la respuesta a la pregunta NO se encuentra explícitamente en <documents>, está terminantemente PROHIBIDO asumir, deducir o inventar información externa. En ese caso debes responder EXACTAMENTE con la frase:
"{FRASE_ABSTENCION_CANONICA}"
4. No agregues cortesías ni disculpas cuando te abstengas.
</rules>

<user_question>
{pregunta}
</user_question>
""".strip()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ejecución Comparativa sobre el Banco de Preguntas

# COMMAND ----------

resultados_experimento = []

print("Ejecutando pruebas de grounding...")
for item in BANCO_PREGUNTAS:
    pid = item["id"]
    tipo = item["tipo"]
    pregunta = item["pregunta"]
    
    # 1. Ejecución con Prompt Ingenuo
    res_ingenuo = llamar_llm(construir_prompt_ingenuo(pregunta), modelo=MODELO_20B, temperature=0.0)
    texto_ing = res_ingenuo["texto"]
    
    # 2. Ejecución con Prompt Grounded
    res_grounded = llamar_llm(construir_prompt_grounded(pregunta), modelo=MODELO_20B, temperature=0.0)
    texto_gro = res_grounded["texto"]
    
    # Evaluación automática
    tiene_cita = bool(re.search(r"\[DOC-\d+\]", texto_gro))
    se_abstuvo_gro = FRASE_ABSTENCION_CANONICA.lower() in texto_gro.lower()
    
    # Detección de alucinación en preguntas no respondibles
    if tipo == "no_respondible":
        alucino_ing = not (FRASE_ABSTENCION_CANONICA.lower() in texto_ing.lower() or "no tengo" in texto_ing.lower() or "no está" in texto_ing.lower())
        alucino_gro = not se_abstuvo_gro
    else:
        alucino_ing = False
        alucino_gro = False
        
    resultados_experimento.append({
        "ID": pid,
        "Tipo": tipo,
        "Pregunta": pregunta,
        "Resp_Ingenua": texto_ing[:120] + ("..." if len(texto_ing) > 120 else ""),
        "Resp_Grounded": texto_gro[:120] + ("..." if len(texto_gro) > 120 else ""),
        "Tiene_Cita": tiene_cita,
        "Abstencion_Correcta": se_abstuvo_gro if tipo == "no_respondible" else True,
        "Alucinacion_Ingenua": alucino_ing,
        "Alucinacion_Grounded": alucino_gro
    })

df_comparativo = pd.DataFrame(resultados_experimento)
display(spark.createDataFrame(df_comparativo))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Métricas Consolidadas de Grounding y Alucinación

# COMMAND ----------

total_no_resp = sum(1 for r in resultados_experimento if r["Tipo"] == "no_respondible")
aluc_ing_count = sum(1 for r in resultados_experimento if r["Tipo"] == "no_respondible" and r["Alucinacion_Ingenua"])
aluc_gro_count = sum(1 for r in resultados_experimento if r["Tipo"] == "no_respondible" and r["Alucinacion_Grounded"])
citas_ok_count = sum(1 for r in resultados_experimento if r["Tipo"] == "respondible" and r["Tiene_Cita"])
total_resp = sum(1 for r in resultados_experimento if r["Tipo"] == "respondible")

tasa_aluc_ingenuo = round((aluc_ing_count / total_no_resp) * 100, 1)
tasa_aluc_grounded = round((aluc_gro_count / total_no_resp) * 100, 1)
tasa_citas_grounded = round((citas_ok_count / total_resp) * 100, 1)

df_metricas = pd.DataFrame([
    {
        "Estrategia": "Prompt Ingenuo (Sin Grounding)",
        "Tasa de Alucinación en No Respondibles": f"{tasa_aluc_ingenuo}%",
        "Cumplimiento de Citas [DOC-X]": "0.0%",
        "Comportamiento": "Inventa conversiones a COP o colores del mercado general"
    },
    {
        "Estrategia": "Prompt Grounded (XML + Regla Canónica)",
        "Tasa de Alucinación en No Respondibles": f"{tasa_aluc_grounded}%",
        "Cumplimiento de Citas [DOC-X]": f"{tasa_citas_grounded}%",
        "Comportamiento": "Se abstiene determinísticamente y respalda afirmaciones con fuentes"
    }
])

display(spark.createDataFrame(df_metricas))

# COMMAND ----------

# Salida esperada:
# 1. En preguntas respondibles (P1-P4), el prompt grounded responde de forma concisa citando [DOC-01] o [DOC-02] según corresponda.
# 2. En preguntas no respondibles (P5 precio en COP, P6 colores), el prompt ingenuo inventa cifras aproximadas o colores genéricos (alucinación 100%), mientras que el prompt grounded devuelve exactamente "No encuentro esa información en la documentación proporcionada." (tasa de alucinación 0%).
# 3. La tabla final muestra el contraste drástico de fidelidad entre ambas estrategias.
