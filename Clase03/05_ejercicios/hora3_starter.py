# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 3 (Starter): Pipeline de Prompt Chaining de 3 Pasos para el Reto Integrador
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Construir en equipo un pipeline modular de 3 prompts encadenados en Databricks (Diseño Mermaid $\to$ Implementación de Código $\to$ Auditoría y Tests) aplicado al Reto Integrador.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Prompt Chaining, generación de diagramas como código (`Mermaid.js`) y aseguramiento de calidad asistido por IA.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Definir el Caso de Uso del Reto Integrador del Equipo
# MAGIC Personaliza `# TODO 1` con los datos del reto de tu equipo (puedes usar el caso base preconfigurado como guía o reemplazarlo por el de tu proyecto).

# COMMAND ----------

import time
import re
from typing import Dict, Any
import pandas as pd

MODELO_RAPIDO = "databricks-gpt-oss-20b"
MODELO_POTENTE = "databricks-gpt-oss-120b"

# TODO 1: Personaliza los datos del Reto Integrador de tu equipo
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

def ejecutar_paso_cadena(nombre_paso: str, prompt: str, modelo: str, temp: float = 0.0, max_tokens: int = 650) -> Dict[str, Any]:
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

print(f"Configurado reto para: {RETO_INTEGRADOR_EQUIPO['nombre_equipo']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Eslabón 1: De Reglas del Reto a Especificación Tipada + Diagrama `Mermaid.js`
# MAGIC Completa `# TODO 2` para que el modelo genere la tabla de variables tipadas y el árbol de decisión en formato `flowchart TD`.

# COMMAND ----------

# TODO 2: Completa el prompt del Paso 1
prompt_paso_1 = f"""
### ROL
Actúa como un Arquitecto de Soluciones de IA y Diseñador de Algoritmos.

### CONTEXTO DEL RETO INTEGRADOR ({RETO_INTEGRADOR_EQUIPO['titulo_reto']})
{RETO_INTEGRADOR_EQUIPO['reglas_negocio']}

### TAREA
Genera dos secciones claramente delimitadas:
1. [ESPECIFICACION]: Lista de parámetros de entrada con su tipo de dato en Python y las condiciones exactas de frontera.
2. [MERMAID]: Un diagrama de flujo en sintaxis `flowchart TD` de Mermaid.js que modele todo el árbol de decisión. Recuerda encerrar todas las etiquetas de nodos entre comillas dobles (ejemplo: `A["Inicio"]`).
""".strip()

res_paso_1 = ejecutar_paso_cadena("Paso 1: Diseño y Mermaid", prompt_paso_1, modelo=MODELO_RAPIDO, temp=0.1)
salida_paso_1 = res_paso_1["salida"]
print("=== SALIDA DEL PASO 1 (ESPECIFICACIÓN + MERMAID) ===")
print(salida_paso_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Eslabón 2: Generación de la Función Python Tipada a partir del Paso 1
# MAGIC Completa `# TODO 3` asegurándote de inyectar `{salida_paso_1}` como contexto para que `databricks-gpt-oss-120b` escriba la función en Python.

# COMMAND ----------

# TODO 3: Construye el prompt del Paso 2 encadenando la variable `salida_paso_1`
prompt_paso_2 = f"""
### ROL
Actúa como un Desarrollador Senior de Python especialista en código limpio y tipado estricto.

### DISEÑO ALGORÍTMICO APROBADO (SALIDA DEL PASO 1)
{salida_paso_1}

### TAREA
Escribe en Python una función llamada `evaluar_reclamo(monto_reclamado_usd: float, edad_cliente: int, dias_abierto: int, canal: str) -> str` que implemente exactamente la lógica del Paso 1, incluyendo docstring y validación de valores nulos o negativos.
Devuelve ÚNICAMENTE el código Python sin texto adicional.
""".strip()

res_paso_2 = ejecutar_paso_cadena("Paso 2: Generación Código Python", prompt_paso_2, modelo=MODELO_POTENTE, temp=0.0)
salida_paso_2 = res_paso_2["salida"]
print("=== SALIDA DEL PASO 2 (CÓDIGO PYTHON GENERADO) ===")
print(salida_paso_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Eslabón 3: Auditoría de Calidad y Generación de Casos de Prueba (`assert`)
# MAGIC Completa `# TODO 4` encadenando tanto las reglas como `{salida_paso_2}` para que el modelo audite el código y escriba 4 pruebas unitarias.

# COMMAND ----------

# TODO 4: Construye el prompt del Paso 3 inyectando `salida_paso_2`
prompt_paso_3 = f"""
### ROL
Actúa como un Lead QA Engineer especialista en pruebas unitarias de software financiero.

### REGLAS ORIGINALES
{RETO_INTEGRADOR_EQUIPO['reglas_negocio']}

### CÓDIGO GENERADO EN EL PASO 2
{salida_paso_2}

### TAREA
1. Verifica si el código cumple el 100% de las reglas de negocio y si tiene algún bug potencial con casos borde (ej. `monto_reclamado_usd = 2000` exacto o `dias_abierto = 0`).
2. Escribe 4 sentencias `assert` en Python probando las 4 salidas posibles de `evaluar_reclamo`.
""".strip()

res_paso_3 = ejecutar_paso_cadena("Paso 3: Auditoría QA y Tests", prompt_paso_3, modelo=MODELO_RAPIDO, temp=0.0)
salida_paso_3 = res_paso_3["salida"]
print("=== SALIDA DEL PASO 3 (REPORTE QA Y ASSERTS) ===")
print(salida_paso_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Resumen de Telemetría del Pipeline del Equipo

# COMMAND ----------

df_resumen_cadena = pd.DataFrame([
    {
        "Equipo": RETO_INTEGRADOR_EQUIPO["nombre_equipo"],
        "Eslabón": r["paso"],
        "Modelo": r["modelo"],
        "Temp": r["temperature"],
        "Latencia (s)": r["latencia_seg"],
        "Caracteres Generados": len(r["salida"])
    }
    for r in [res_paso_1, res_paso_2, res_paso_3]
])
display(spark.createDataFrame(df_resumen_cadena))

# COMMAND ----------

# Salida esperada:
# 1. Paso 1: Especificación de tipos y código `flowchart TD` en Mermaid.js para el reto del equipo.
# 2. Paso 2: Función Python `evaluar_reclamo(...)` tipada y documentada.
# 3. Paso 3: Verificación de casos de frontera y 4 sentencias `assert`, junto con la tabla de telemetría de los 3 pasos.
