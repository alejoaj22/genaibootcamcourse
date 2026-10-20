# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 — Ejemplo 2: Efecto de Nucleus Sampling (`top_p`) y Control de Repetición
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Analizar cómo el parámetro `top_p` (Nucleus Sampling) recorta dinámicamente la cola de la distribución de probabilidades y cómo evitar la repetición léxica al generar datos sintéticos de prueba en Databricks.
# MAGIC * **Tiempo estimado:** 8 minutos.
# MAGIC * **Concepto que ilustra:** Selección del núcleo mínimo de tokens cuya probabilidad acumulada alcanza el umbral $p$ ($\sum_{w \in V^{(p)}} P(w) \ge p$), y contraste entre un núcleo estrecho (`top_p = 0.1`) y un núcleo amplio (`top_p = 0.95`).
# MAGIC * **Requisitos previos:** Acceso a `ai_query()` en Databricks con `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Función de Invocación Parametrizada con `top_p` y `temperature`
# MAGIC Siguiendo la regla de oro de calibración estocástica, mantenemos fija una temperatura moderada (`temperature = 0.8`) para observar el efecto aislado de variar `top_p` desde `0.10` hasta `0.98`.

# COMMAND ----------

import time
from typing import Dict, Any, List
import pandas as pd

MODELO = "databricks-gpt-oss-20b"

def consultar_con_top_p(
    prompt: str,
    top_p: float = 0.9,
    temperature: float = 0.8,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """
    Invoca ai_query() configurando explícitamente top_p y temperature.
    """
    prompt_seguro = prompt.replace("\\", "\\\\").replace("'", "''")
    inicio = time.time()
    
    sql = f"""
    SELECT ai_query(
        '{MODELO}',
        '{prompt_seguro}',
        modelParameters => named_struct(
            'temperature', CAST({float(temperature)} AS DOUBLE),
            'top_p', CAST({float(top_p)} AS DOUBLE),
            'max_tokens', CAST({int(max_tokens)} AS INT)
        )
    ) AS respuesta
    """
    try:
        resultado = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        sql_fallback = f"SELECT ai_query('{MODELO}', '{prompt_seguro}') AS respuesta"
        resultado = spark.sql(sql_fallback).collect()[0]["respuesta"]
        
    duracion = round(time.time() - inicio, 3)
    texto = resultado.strip() if resultado else ""
    palabras = [p.strip(".,;:()[]{}\"'").lower() for p in texto.split() if p.strip()]
    tasa_repeticion = round(1.0 - (len(set(palabras)) / max(len(palabras), 1)), 3)
    
    return {
        "top_p": top_p,
        "temperature_fija": temperature,
        "latencia_seg": duracion,
        "total_palabras": len(palabras),
        "vocabulario_unico": len(set(palabras)),
        "tasa_repeticion": tasa_repeticion,
        "respuesta": texto
    }

print("Función consultar_con_top_p() definida correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experimento: Generación de Datos Sintéticos de Reseñas de Clientes
# MAGIC En ingeniería de datos y QA de software es muy común usar LLMs para generar **datasets sintéticos** de prueba. Si `top_p` es demasiado bajo (`0.10`), todos los registros sintéticos suenan monótonos y usan exactamente los mismos adjetivos. Si `top_p` se abre a `0.90 - 0.95`, obtenemos variedad realista sin perder coherencia.

# COMMAND ----------

prompt_datos_sinteticos = """
Genera 5 reseñas sintéticas cortas (1 línea cada una) de usuarios reales sobre una aplicación móvil de entregas de supermercado en Colombia.
Cada reseña debe describir una situación distinta (retraso, producto frío, cobro duplicado, excelente atención, cupón fallido).
Formato requerido:
1. [RESEÑA]
2. [RESEÑA]
3. [RESEÑA]
4. [RESEÑA]
5. [RESEÑA]
""".strip()

valores_top_p = [0.10, 0.50, 0.90, 0.98]
resultados_top_p: List[Dict[str, Any]] = []

for p in valores_top_p:
    res = consultar_con_top_p(prompt_datos_sinteticos, top_p=p, temperature=0.8, max_tokens=280)
    resultados_top_p.append(res)
    print("=" * 80)
    print(f"CONFIGURACIÓN: top_p = {p} | temperature = 0.8 | Vocabulario único: {res['vocabulario_unico']} palabras")
    print("-" * 80)
    print(res["respuesta"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Comparación de Repetición y Diversidad en Tabla Resumen
# MAGIC Además de controlar `top_p`, cuando necesitamos evitar que el modelo repita verbos o estructuras en generación de datos sintéticos, combinamos el muestreo por núcleo con una restricción explícita de no repetición léxica en el contexto.

# COMMAND ----------

df_resumen_top_p = pd.DataFrame([
    {
        "Top_P (Nucleus)": r["top_p"],
        "Temperature": r["temperature_fija"],
        "Palabras Totales": r["total_palabras"],
        "Vocabulario Único": r["vocabulario_unico"],
        "Tasa de Repetición": r["tasa_repeticion"],
        "Extracto Generado": r["respuesta"][:140].replace("\n", " | ")
    }
    for r in resultados_top_p
])

display(spark.createDataFrame(df_resumen_top_p))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Demostración del Peligro de Penalizar la Repetición en Código
# MAGIC Observemos por qué exigir "no repetir palabras" funciona de maravilla en texto narrativo o datos sintéticos, pero **destruye la consistencia de variables** si se aplica erróneamente a generación de código Python.

# COMMAND ----------

prompt_codigo_normal = """
Escribe un bloque corto de PySpark de 6 líneas que tome un DataFrame llamado `df_ventas`, filtre `monto > 0`, agregue la columna `impuesto` y seleccione las columnas finales. Devuelve solo código Python.
""".strip()

res_codigo_limpio = consultar_con_top_p(prompt_codigo_normal, top_p=0.1, temperature=0.0, max_tokens=200)
print("--- CÓDIGO CON PARÁMETROS CONSERVADORES (top_p=0.1, temp=0.0) ---")
print(res_codigo_limpio["respuesta"])

# COMMAND ----------

# Salida esperada:
# 1. Impresión en consola de 4 conjuntos de 5 reseñas sintéticas para top_p en [0.10, 0.50, 0.90, 0.98].
# 2. DataFrame de Spark mostrando cómo el Vocabulario Único aumenta al expandir el núcleo de top_p=0.10 a top_p=0.90.
# 3. Bloque de código PySpark limpio y consistente manteniendo el nombre de variable `df_ventas` cuando se usa top_p=0.1 y temperature=0.0.
