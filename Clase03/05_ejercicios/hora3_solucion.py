# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 3 (Solución Completa): Pipeline de Prompt Chaining de 3 Pasos para el Reto Integrador
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Solución de referencia completa que implementa un pipeline encadenado de 3 etapas en Databricks (Especificación + Diagrama Mermaid renderizado $\to$ Generación de Función Python Tipada $\to$ Auditoría QA y Ejecución Automática de Tests en Vivo).
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Arquitectura modular de Prompt Chaining con verificación ejecutable de extremo a extremo.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Definición del Caso del Reto Integrador y Utilidades del Pipeline

# COMMAND ----------

import time
import re
from typing import Dict, Any
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"

RETO_INTEGRADOR_EQUIPO = {
    "nombre_equipo": "Equipo NODO - Guardianes de Datos",
    "titulo_reto": "Sistema Inteligente de Priorización de Reclamos Financieros",
    "reglas_negocio": """
    Recibimos reclamos de clientes bancarios con: monto_reclamado_usd (float), edad_cliente (int), dias_abierto (int) y canal (str).
    Reglas de enrutamiento:
    1. Si monto_reclamado_usd < 0 o dias_abierto < 0: devolver 'ERROR_DATOS_INVALIDOS'.
    2. Si edad_cliente >= 65 (adulto mayor) o monto_reclamado_usd >= 2000: clasificar como 'PRIORIDAD_ALTA_VIP'.
    3. Si dias_abierto >= 5 y no cayó en la regla anterior: clasificar como 'ESCALAMIENTO_SLA'.
    4. En cualquier otro caso: clasificar como 'FLUJO_AUTOMATICO_ESTANDAR'.
    """.strip()
}

def ejecutar_paso_cadena(nombre_paso: str, prompt: str, modelo: str, temp: float = 0.0, max_tokens: int = 700) -> Dict[str, Any]:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
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
    latencia = round(time.time() - t0, 3)
    texto = res.strip() if res else ""
    return {
        "paso": nombre_paso,
        "modelo": modelo,
        "temperature": temp,
        "latencia_seg": latencia,
        "salida": texto
    }

def renderizar_mermaid_html(codigo_mermaid: str) -> None:
    html = f"""
    <div style="border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; background: #fff;">
        <h4 style="margin-top:0;">Diagrama de Flujo del Reto Integrador (Generado en Paso 1)</h4>
        <div class="mermaid">
{codigo_mermaid}
        </div>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    try:
        displayHTML(html)
    except NameError:
        print(codigo_mermaid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Eslabón 1: Diseño Algorítmico y Diagrama `Mermaid.js`

# COMMAND ----------

prompt_paso_1 = f"""
### ROL
Actúa como un Arquitecto Principal de Soluciones y Especialista en Mermaid.js.

### REGLAS DEL RETO INTEGRADOR ({RETO_INTEGRADOR_EQUIPO['titulo_reto']})
{RETO_INTEGRADOR_EQUIPO['reglas_negocio']}

### TAREA
Genera tu respuesta dividida exactamente en estos dos bloques:

[ESPECIFICACION]
Tabla de parámetros de entrada, tipos en Python y orden exacto de evaluación condicional.
[/ESPECIFICACION]

[MERMAID]
Código `flowchart TD` válido en Mermaid.js con todas las etiquetas encerradas entre comillas dobles (ejemplo: `N1{{"¿monto < 0 o dias < 0?"}}`). No incluyas backticks dentro de este bloque.
[/MERMAID]
""".strip()

res_paso_1 = ejecutar_paso_cadena("Paso 1: Diseño Algorítmico + Mermaid", prompt_paso_1, modelo=MODELO_20B, temp=0.0)
salida_paso_1 = res_paso_1["salida"]

match_mermaid = re.search(r"\[MERMAID\](.*?)\[/MERMAID\]", salida_paso_1, re.DOTALL)
mermaid_extraido = match_mermaid.group(1).strip() if match_mermaid else salida_paso_1
mermaid_extraido = re.sub(r"^```(?:mermaid)?|```$", "", mermaid_extraido, flags=re.MULTILINE).strip()

print("=== ESPECIFICACIÓN Y FLUJO GENERADOS EN EL PASO 1 ===")
print(salida_paso_1)
renderizar_mermaid_html(mermaid_extraido)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Eslabón 2: Síntesis de Código Python Productivo

# COMMAND ----------

prompt_paso_2 = f"""
### ROL
Actúa como un Staff Software Engineer en Python.

### DISEÑO VALIDADO DEL PASO 1
{salida_paso_1}

### TAREA
Escribe únicamente el código Python de la función `evaluar_reclamo(monto_reclamado_usd: float, edad_cliente: int, dias_abierto: int, canal: str = "APP") -> str` que implemente exactamente las 4 reglas en el orden especificado.
Encierra el código dentro de [CODIGO_PYTHON] y [/CODIGO_PYTHON] sin usar bloques markdown ```python.
""".strip()

res_paso_2 = ejecutar_paso_cadena("Paso 2: Implementación Python", prompt_paso_2, modelo=MODELO_120B, temp=0.0)
salida_paso_2 = res_paso_2["salida"]

match_py = re.search(r"\[CODIGO_PYTHON\](.*?)\[/CODIGO_PYTHON\]", salida_paso_2, re.DOTALL)
codigo_py_paso_2 = match_py.group(1).strip() if match_py else salida_paso_2
codigo_py_paso_2 = re.sub(r"^```(?:python)?|```$", "", codigo_py_paso_2, flags=re.MULTILINE).strip()

print("=== CÓDIGO PYTHON GENERADO EN EL PASO 2 ===")
print(codigo_py_paso_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Eslabón 3: Auditoría QA y Ejecución de Suite de Pruebas Unitarias

# COMMAND ----------

prompt_paso_3 = f"""
### ROL
Actúa como un Ingeniero Principal de QA y Automatización de Pruebas en Python.

### REGLAS DE NEGOCIO
{RETO_INTEGRADOR_EQUIPO['reglas_negocio']}

### CÓDIGO GENERADO EN EL PASO 2
{codigo_py_paso_2}

### TAREA
Devuelve dos bloques:
[AUDITORIA_QA]
Veredicto de cumplimiento de las 4 reglas y análisis de condiciones de frontera.
[/AUDITORIA_QA]

[TESTS_PYTHON]
Escribe únicamente una función Python `correr_tests_reto()` con 5 sentencias `assert` que verifiquen cada rama de `evaluar_reclamo` (incluyendo valores negativos, adulto mayor, monto >= 2000, dias >= 5 y flujo estándar) e imprima al final `"PIPELINE DEL RETO INTEGRADOR VERIFICADO AL 100%"`. Sin bloques markdown.
[/TESTS_PYTHON]
""".strip()

res_paso_3 = ejecutar_paso_cadena("Paso 3: Auditoría QA y Suite de Tests", prompt_paso_3, modelo=MODELO_20B, temp=0.0)
salida_paso_3 = res_paso_3["salida"]

match_tests = re.search(r"\[TESTS_PYTHON\](.*?)\[/TESTS_PYTHON\]", salida_paso_3, re.DOTALL)
codigo_tests_paso_3 = match_tests.group(1).strip() if match_tests else ""
codigo_tests_paso_3 = re.sub(r"^```(?:python)?|```$", "", codigo_tests_paso_3, flags=re.MULTILINE).strip()

print("=== REPORTE DE AUDITORÍA Y TESTS GENERADOS EN EL PASO 3 ===")
print(salida_paso_3)

if "def evaluar_reclamo" in codigo_py_paso_2:
    exec(codigo_py_paso_2, globals())
if "def correr_tests_reto" in codigo_tests_paso_3:
    exec(codigo_tests_paso_3, globals())
    correr_tests_reto()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Dashboard de Telemetría del Pipeline

# COMMAND ----------

df_telemetria = pd.DataFrame([
    {
        "Equipo": RETO_INTEGRADOR_EQUIPO["nombre_equipo"],
        "Eslabón": r["paso"],
        "Modelo": r["modelo"],
        "Temperature": r["temperature"],
        "Latencia (s)": r["latencia_seg"],
        "Caracteres": len(r["salida"])
    }
    for r in [res_paso_1, res_paso_2, res_paso_3]
])
display(spark.createDataFrame(df_telemetria))

# COMMAND ----------

# Salida esperada:
# 1. Renderizado visual del diagrama Mermaid del reto integrador en el Paso 1.
# 2. Función Python `evaluar_reclamo` generada en el Paso 2.
# 3. Ejecución automática de `correr_tests_reto()` imprimiendo `"PIPELINE DEL RETO INTEGRADOR VERIFICADO AL 100%"` y tabla de telemetría.
