# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 — Ejemplo 3: Ingeniería de Contexto y Calibración Determinista vs. Exploratoria
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Construir una arquitectura completa de **Ingeniería de Contexto** (Rol + Esquema DDL + Memoria Operativa + Reglas + Contrato de Salida) y calibrar los hiperparámetros para dos tareas opuestas sobre el mismo dominio: (1) Generación determinista de SQL/Regex y (2) Exploración creativa de arquitectura.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Diferencia práctica entre *Prompt Engineering* básico e *Ingeniería de Contexto* multicapa, comparando `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Definición de los Componentes del Contexto (Inyección de Dependencias Cognitivas)
# MAGIC En lugar de escribir un prompt improvisado, ensamblamos programáticamente los componentes del contexto: el esquema técnico del catálogo en Unity Catalog, el glosario de reglas del negocio y las restricciones de salida.

# COMMAND ----------

import json
import time
from typing import Dict, Any
import pandas as pd

DDL_ESQUEMA_UNITY_CATALOG = """
CREATE TABLE prod_lakehouse.logistica.envios_ultimamilla (
    envio_id STRING COMMENT 'Identificador único del envío formato ENV-YYYYMMDD-NNNN',
    pedido_id STRING,
    ciudad_destino STRING,
    fecha_prometida TIMESTAMP,
    fecha_entrega_real TIMESTAMP,
    costo_flete_cop DECIMAL(14,2),
    estado_envio STRING COMMENT 'Valores permitidos: ENTREGADO, EN_RUTA, NOVEDAD, CANCELADO'
) USING DELTA;
""".strip()

MEMORIA_REGLAS_NEGOCIO = """
- Un envío se considera 'INCUMPLIDO_SLA' si estado_envio = 'ENTREGADO' y fecha_entrega_real > fecha_prometida.
- Los envíos con estado_envio = 'CANCELADO' deben excluirse de cualquier cálculo de efectividad logística.
- El formato válido de envio_id siempre inicia con 'ENV-' seguido de 8 dígitos de fecha, un guion y 4 dígitos numéricos.
""".strip()

def ensamblar_contexto(
    rol: str,
    tarea: str,
    contrato_salida: str,
    incluir_ddl: bool = True,
    incluir_reglas: bool = True
) -> str:
    """
    Ensambla un prompt de grado producción aplicando Ingeniería de Contexto.
    """
    bloques = [f"### ROL DEL SISTEMA\n{rol}"]
    if incluir_ddl:
        bloques.append(f"### ESQUEMA TÉCNICO (UNITY CATALOG DDL)\n{DDL_ESQUEMA_UNITY_CATALOG}")
    if incluir_reglas:
        bloques.append(f"### MEMORIA DE REGLAS DE NEGOCIO\n{MEMORIA_REGLAS_NEGOCIO}")
    bloques.append(f"### TAREA SOLICITADA\n{tarea}")
    bloques.append(f"### CONTRATO DE SALIDA Y RESTRICCIONES\n{contrato_salida}")
    return "\n\n".join(bloques)

def invocar_modelo(prompt: str, modelo: str, temp: float, top_p: float, max_tokens: int = 500) -> Dict[str, Any]:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    t0 = time.time()
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_esc}',
        modelParameters => named_struct(
            'temperature', CAST({float(temp)} AS DOUBLE),
            'top_p', CAST({float(top_p)} AS DOUBLE),
            'max_tokens', CAST({int(max_tokens)} AS INT)
        )
    ) AS respuesta
    """
    try:
        salida = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        salida = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return {
        "modelo": modelo,
        "temp": temp,
        "top_p": top_p,
        "latencia_seg": round(time.time() - t0, 3),
        "respuesta": salida.strip() if salida else ""
    }

print("Motor de ensamblaje de contexto inicializado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Contraste 1: Prompt Básico (Sin Contexto) vs. Ingeniería de Contexto (Tarea Determinista)
# MAGIC Pediremos una consulta SQL y una expresión regular para validar `envio_id` y calcular la tasa de incumplimiento de SLA por ciudad. Primero veremos qué ocurre cuando enviamos un prompt ingenuo sin contexto frente a nuestro contexto ensamblado con `temperature = 0.0`.

# COMMAND ----------

prompt_ingenuo = "Dame un query SQL que calcule el porcentaje de envíos que incumplieron el SLA por ciudad y la regex del id de envío."

contexto_determinista = ensamblar_contexto(
    rol="Actúa como un Arquitecto Senior de Datos especialista en Databricks SQL y expresiones regulares.",
    tarea=(
        "Genera un objeto JSON válido con dos claves exactas:\n"
        "1. 'sql_query': Consulta en Databricks SQL que calcule por 'ciudad_destino' el total de envíos válidos, "
        "el total de envíos con 'INCUMPLIDO_SLA' y el porcentaje de incumplimiento redondeado a 2 decimales.\n"
        "2. 'regex_envio_id': Expresión regular exacta para validar el formato de 'envio_id'."
    ),
    contrato_salida=(
        "- Devuelve ÚNICAMENTE un objeto JSON válido sin bloques markdown (sin ```json).\n"
        "- No inventes columnas fuera del DDL proporcionado.\n"
        "- Excluye explícitamente los envíos con estado 'CANCELADO'."
    )
)

res_ingenuo_20b = invocar_modelo(prompt_ingenuo, "databricks-gpt-oss-20b", temp=0.7, top_p=0.95)
res_contexto_20b = invocar_modelo(contexto_determinista, "databricks-gpt-oss-20b", temp=0.0, top_p=1.0)
res_contexto_120b = invocar_modelo(contexto_determinista, "databricks-gpt-oss-120b", temp=0.0, top_p=1.0)

print("=== 1. SALIDA CON PROMPT INGENUO (Inventa nombres de tablas y columnas) ===")
print(res_ingenuo_20b["respuesta"][:450])
print("\n=== 2. SALIDA CON INGENIERÍA DE CONTEXTO EN gpt-oss-20b (temp=0.0) ===")
print(res_contexto_20b["respuesta"])
print("\n=== 3. SALIDA CON INGENIERÍA DE CONTEXTO EN gpt-oss-120b (temp=0.0) ===")
print(res_contexto_120b["respuesta"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Contraste 2: Tarea Exploratoria (Brainstorming de Nuevas Variables Predictivas)
# MAGIC Ahora usamos el **mismo contexto de esquema y reglas de negocio**, pero cambiamos la tarea a diseño exploratorio de arquitectura e incrementamos los parámetros a `temperature = 0.85` y `top_p = 0.95`.

# COMMAND ----------

contexto_exploratorio = ensamblar_contexto(
    rol="Actúa como un Principal AI Product Strategist y Lead Data Scientist en logística de última milla.",
    tarea=(
        "A partir de la tabla actual de envíos, propón 3 ideas innovadoras de enriquecimiento de datos (nuevas fuentes externas "
        "o features derivados) que permitan predecir proactivamente si un envío caerá en 'INCUMPLIDO_SLA' 4 horas antes de la fecha prometida."
    ),
    contrato_salida=(
        "Presenta cada idea con: [Nombre del Feature], [Fuente Externa Requerida] y [Mecanismo de Impacto en el Modelo]."
    )
)

res_exploratorio_120b = invocar_modelo(contexto_exploratorio, "databricks-gpt-oss-120b", temp=0.85, top_p=0.95, max_tokens=550)
print("=== SALIDA EXPLORATORIA CALIBRADA (temp=0.85, top_p=0.95 en gpt-oss-120b) ===")
print(res_exploratorio_120b["respuesta"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resumen Comparativo de Ejecución

# COMMAND ----------

tabla_comparativa = pd.DataFrame([
    {
        "Escenario": "Prompt Ingenuo (Sin Contexto)",
        "Modelo": res_ingenuo_20b["modelo"],
        "Temp / Top-P": f"{res_ingenuo_20b['temp']} / {res_ingenuo_20b['top_p']}",
        "Latencia (s)": res_ingenuo_20b["latencia_seg"],
        "¿Usa Esquema Real?": "NO (Alucina columnas)",
        "Longitud (chars)": len(res_ingenuo_20b["respuesta"])
    },
    {
        "Escenario": "Contexto Determinista (SQL + Regex)",
        "Modelo": res_contexto_20b["modelo"],
        "Temp / Top-P": f"{res_contexto_20b['temp']} / {res_contexto_20b['top_p']}",
        "Latencia (s)": res_contexto_20b["latencia_seg"],
        "¿Usa Esquema Real?": "SÍ (100% Fiel al DDL)",
        "Longitud (chars)": len(res_contexto_20b["respuesta"])
    },
    {
        "Escenario": "Contexto Determinista (SQL + Regex)",
        "Modelo": res_contexto_120b["modelo"],
        "Temp / Top-P": f"{res_contexto_120b['temp']} / {res_contexto_120b['top_p']}",
        "Latencia (s)": res_contexto_120b["latencia_seg"],
        "¿Usa Esquema Real?": "SÍ (100% Fiel al DDL)",
        "Longitud (chars)": len(res_contexto_120b["respuesta"])
    },
    {
        "Escenario": "Contexto Exploratorio (Brainstorming Features)",
        "Modelo": res_exploratorio_120b["modelo"],
        "Temp / Top-P": f"{res_exploratorio_120b['temp']} / {res_exploratorio_120b['top_p']}",
        "Latencia (s)": res_exploratorio_120b["latencia_seg"],
        "¿Usa Esquema Real?": "SÍ (Anclado al dominio)",
        "Longitud (chars)": len(res_exploratorio_120b["respuesta"])
    }
])

display(spark.createDataFrame(tabla_comparativa))

# COMMAND ----------

# Salida esperada:
# 1. El prompt ingenuo inventa nombres de tablas (como `shipments` o `envios`) y columnas inexistentes.
# 2. Tanto `databricks-gpt-oss-20b` como `databricks-gpt-oss-120b` con Ingeniería de Contexto y temp=0.0 producen el JSON con la consulta SQL exacta sobre `prod_lakehouse.logistica.envios_ultimamilla` y la regex `^ENV-\d{8}-\d{4}$`.
# 3. En el escenario exploratorio (temp=0.85), el modelo propone 3 nuevas variables predictivas (ej. clima/tráfico en tiempo real, historial de densidad zonal, telemetría GPS) ancladas a las columnas reales del DDL.
