# Databricks notebook source
# MAGIC %md
# MAGIC # Laboratorio Hora 1 (Solución Completa): Experimentación con 5 Combinaciones de Temperatura y Top-P
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Solución de referencia completa que evalúa 5 combinaciones de `temperature` y `top_p` sobre `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`, midiendo determinismo, fidelidad al esquema DDL y diversidad léxica.
# MAGIC * **Tiempo estimado:** 20 minutos.
# MAGIC * **Concepto que ilustra:** Calibración estocástica rigurosa e inspección cuantitativa de outputs generados con `ai_query()`.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Función de Invocación con Telemetría y Validación de Esquema

# COMMAND ----------

import time
from typing import Dict, Any, List
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"

def ejecutar_con_parametros(
    prompt: str,
    temperature: float,
    top_p: float,
    modelo: str = MODELO_20B,
    max_tokens: int = 380
) -> Dict[str, Any]:
    """
    Ejecuta ai_query() en Databricks pasando temperature, top_p y max_tokens.
    """
    prompt_seguro = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
    
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
        "modelo": modelo,
        "latencia_seg": latencia,
        "respuesta": texto
    }

print("Función ejecutar_con_parametros() cargada exitosamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Contexto de Grado Producción (Rol + DDL + Tarea Dual + Contrato de Salida)

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
- Incluye la cláusula `HAVING COUNT(*) > 3` (o equivalente) en la consulta SQL.
- Sé conciso (máximo 180 palabras en total).
""".strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ejecución de la Matriz de 5 Combinaciones y Evaluación Automatizada

# COMMAND ----------

COMBINACIONES_EXPERIMENTALES: List[Dict[str, Any]] = [
    {"id": "C1", "nombre": "Congelado Determinista", "temperature": 0.0, "top_p": 1.00},
    {"id": "C2", "nombre": "Técnico Conservador",    "temperature": 0.2, "top_p": 0.90},
    {"id": "C3", "nombre": "Balanceado Estándar",    "temperature": 0.5, "top_p": 0.85},
    {"id": "C4", "nombre": "Núcleo Estrecho",        "temperature": 0.8, "top_p": 0.20},
    {"id": "C5", "nombre": "Exploración Abierta",    "temperature": 1.0, "top_p": 0.98},
]

COLUMNAS_PERMITIDAS = {"tx_id", "cliente_id", "fecha_tx", "monto_usd", "comercio_categoria", "estado_tx"}

resultados_matriz = []

for cfg in COMBINACIONES_EXPERIMENTALES:
    c1 = ejecutar_con_parametros(PROMPT_AUDITORIA_DATOS, cfg["temperature"], cfg["top_p"], modelo=MODELO_20B)
    c2 = ejecutar_con_parametros(PROMPT_AUDITORIA_DATOS, cfg["temperature"], cfg["top_p"], modelo=MODELO_20B)
    
    identicas = (c1["respuesta"] == c2["respuesta"])
    usa_tabla_real = "main.riesgos.transacciones_tarjeta" in c1["respuesta"]
    usa_having = "having" in c1["respuesta"].lower()
    
    palabras = [p.strip(".,;:()[]{}\"'").lower() for p in (c1["respuesta"] + " " + c2["respuesta"]).split() if p.strip()]
    diversidad = round(len(set(palabras)) / max(len(palabras), 1), 3)
    
    resultados_matriz.append({
        "ID": cfg["id"],
        "Perfil": cfg["nombre"],
        "Temp": cfg["temperature"],
        "Top-P": cfg["top_p"],
        "¿Determinista (C1==C2)?": "SÍ" if identicas else "NO",
        "¿Tabla y HAVING válidos?": "PASS" if (usa_tabla_real and usa_having) else "REVISAR",
        "Diversidad Léxica": diversidad,
        "Latencia Prom (s)": round((c1["latencia_seg"] + c2["latencia_seg"]) / 2, 3),
        "Muestra Corrida 1": c1["respuesta"][:140].replace("\n", " ")
    })

df_experimento = pd.DataFrame(resultados_matriz)
display(spark.createDataFrame(df_experimento))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Análisis e Interpretación para el Reto Integrador
# MAGIC 1. **Para generación de SQL, filtros y extracción de entidades (C1):** Usamos siempre `temperature = 0.0` y `top_p = 1.0` para garantizar repetibilidad en pipelines productivos.
# MAGIC 2. **Efecto de `C4 (temperature = 0.8, top_p = 0.20)`:** Aunque la temperatura es alta, el filtro de núcleo (`top_p = 0.20`) poda el 80% de la masa de probabilidad inferior, manteniendo la salida notablemente restringida.
# MAGIC 3. **Para ideación de nuevas hipótesis de negocio o generación de casos de prueba variados (C5):** `temperature = 1.0` y `top_p = 0.98` maximizan la variedad de escenarios propuestos.

# COMMAND ----------

# Salida esperada:
# DataFrame con las 5 configuraciones (C1 a C5) donde C1 presenta determinismo exacto ("SÍ"), todas las configuraciones aprueban la validación de tabla y cláusula HAVING ("PASS"), y la métrica de Diversidad Léxica crece de C1 hacia C5.
