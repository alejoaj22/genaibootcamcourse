# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 1 (Starter): Experimentación con 5 Combinaciones de Temperatura y Top-P
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Implementar y evaluar 5 combinaciones de `temperature` y `top_p` en Databricks usando `ai_query()` para medir su impacto en determinismo, validez de esquema SQL y diversidad léxica.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Calibración empírica de hiperparámetros estocásticos e Ingeniería de Contexto aplicada.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Completar la Función de Invocación a `ai_query()`
# MAGIC Completa la sección marcada con `# TODO 1` para construir la consulta SQL que llama a `ai_query()` pasando `temperature`, `top_p` y `max_tokens` dentro de `named_struct`.

# COMMAND ----------

import time
from typing import Dict, Any, List
import pandas as pd

MODELO_LAB = "databricks-gpt-oss-20b"

def ejecutar_con_parametros(
    prompt: str,
    temperature: float,
    top_p: float,
    modelo: str = MODELO_LAB,
    max_tokens: int = 350
) -> Dict[str, Any]:
    """
    Ejecuta ai_query() en Databricks con los parámetros estocásticos indicados.
    """
    prompt_seguro = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
    
    # TODO 1: Construye la sentencia SQL invocando ai_query() con modelParameters => named_struct(...)
    # Pista: incluye 'temperature', CAST({float(temperature)} AS DOUBLE), 'top_p', CAST({float(top_p)} AS DOUBLE), 'max_tokens', CAST({int(max_tokens)} AS INT)
    sql_query = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_seguro}',
        modelParameters => named_struct(
            'temperature', CAST({float(temperature)} AS DOUBLE),
            'top_p', CAST({float(top_p)} AS DOUBLE),
            'max_tokens', CAST({int(max_tokens)} AS INT)
        )
    ) AS respuesta
    """
    
    try:
        resultado = spark.sql(sql_query).collect()[0]["respuesta"]
    except Exception:
        resultado = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_seguro}') AS respuesta").collect()[0]["respuesta"]
        
    latencia = round(time.time() - t0, 3)
    texto = resultado.strip() if resultado else ""
    return {
        "latencia_seg": latencia,
        "respuesta": texto
    }

print("Paso 1 listo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Diseñar el Contexto con Esquema DDL y Tarea Dual
# MAGIC Completa `# TODO 2` para ensamblar el prompt que incluye el esquema de la tabla `main.riesgos.transacciones_tarjeta` y solicita tanto la consulta SQL como las 2 hipótesis de fraude.

# COMMAND ----------

DDL_TABLA_TARJETAS = """
CREATE TABLE main.riesgos.transacciones_tarjeta (
    tx_id STRING,
    cliente_id STRING,
    fecha_tx DATE,
    monto_usd DECIMAL(12,2),
    comercio_categoria STRING,
    estado_tx STRING COMMENT 'Valores: APROBADA, RECHAZADA, REVERSADA'
) USING DELTA;
""".strip()

# TODO 2: Completa las instrucciones de TAREA y RESTRICCIONES dentro del contexto
PROMPT_AUDITORIA_DATOS = f"""
### ROL
Actúa como un Ingeniero Senior de Datos y Analista de Prevención de Fraude en Databricks.

### ESQUEMA TÉCNICO (DDL)
{DDL_TABLA_TARJETAS}

### TAREA
Realiza exactamente estas dos entregas:
1. [SQL]: Una consulta en Databricks SQL sobre `main.riesgos.transacciones_tarjeta` que obtenga los `cliente_id` y `fecha_tx` que tengan estrictamente más de 3 transacciones con `estado_tx = 'RECHAZADA'` en el mismo día, mostrando el conteo de rechazos y la suma de `monto_usd`.
2. [HIPOTESIS]: Propón 2 hipótesis breves y creativas de patrones de ataque o fraude que expliquen ese comportamiento.

### RESTRICCIONES
- Usa únicamente las columnas definidas en el DDL (`tx_id`, `cliente_id`, `fecha_tx`, `monto_usd`, `comercio_categoria`, `estado_tx`).
- Sé conciso (máximo 180 palabras en total).
""".strip()

print("Longitud del contexto ensamblado:", len(PROMPT_AUDITORIA_DATOS), "caracteres.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Definir las 5 Combinaciones y Ejecutar el Experimento
# MAGIC Completa `# TODO 3` agregando las 5 configuraciones (`C1` a `C5`) de la tabla del enunciado.

# COMMAND ----------

# TODO 3: Completa la lista con las 5 combinaciones experimentales (C1, C2, C3, C4 y C5)
COMBINACIONES_EXPERIMENTALES: List[Dict[str, Any]] = [
    {"id": "C1", "nombre": "Congelado Determinista", "temperature": 0.0, "top_p": 1.00},
    {"id": "C2", "nombre": "Técnico Conservador",    "temperature": 0.2, "top_p": 0.90},
    {"id": "C3", "nombre": "Balanceado Estándar",    "temperature": 0.5, "top_p": 0.85},
    {"id": "C4", "nombre": "Núcleo Estrecho",        "temperature": 0.8, "top_p": 0.20},
    {"id": "C5", "nombre": "Exploración Abierta",    "temperature": 1.0, "top_p": 0.98},
]

resultados_matriz = []

for cfg in COMBINACIONES_EXPERIMENTALES:
    c1 = ejecutar_con_parametros(PROMPT_AUDITORIA_DATOS, cfg["temperature"], cfg["top_p"])
    c2 = ejecutar_con_parametros(PROMPT_AUDITORIA_DATOS, cfg["temperature"], cfg["top_p"])
    
    identicas = (c1["respuesta"] == c2["respuesta"])
    usa_tabla_real = "main.riesgos.transacciones_tarjeta" in c1["respuesta"]
    
    palabras = (c1["respuesta"] + " " + c2["respuesta"]).lower().split()
    diversidad = round(len(set(palabras)) / max(len(palabras), 1), 3)
    
    resultados_matriz.append({
        "ID": cfg["id"],
        "Perfil": cfg["nombre"],
        "Temp": cfg["temperature"],
        "Top-P": cfg["top_p"],
        "¿Determinista (C1==C2)?": "SÍ" if identicas else "NO",
        "¿Fiel al DDL?": "SÍ" if usa_tabla_real else "NO",
        "Diversidad Léxica": diversidad,
        "Latencia Prom (s)": round((c1["latencia_seg"] + c2["latencia_seg"]) / 2, 3),
        "Muestra Respuesta": c1["respuesta"][:130].replace("\n", " ")
    })

df_experimento = pd.DataFrame(resultados_matriz)
display(spark.createDataFrame(df_experimento))

# COMMAND ----------

# Salida esperada:
# Tabla comparativa en Databricks con las filas C1 a C5 mostrando que C1 (Temp=0.0, Top-P=1.0) mantiene determinismo ("SÍ"), todas las configuraciones respetan la tabla `main.riesgos.transacciones_tarjeta` gracias a la Ingeniería de Contexto, y C5 alcanza la mayor diversidad léxica en las hipótesis de fraude.
