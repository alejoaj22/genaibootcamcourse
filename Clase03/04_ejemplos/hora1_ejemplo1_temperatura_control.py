# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 — Ejemplo 1: Control Estocástico con Temperatura (0.0 vs 0.7 vs 1.0)
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Comprobar experimentalmente el efecto del parámetro `temperature` (0.0, 0.7 y 1.0) sobre el determinismo, la varianza léxica y la estabilidad sintáctica utilizando `ai_query()` en Databricks.
# MAGIC * **Tiempo estimado:** 8 minutos.
# MAGIC * **Concepto que ilustra:** Escalamiento de logits previo a la función Softmax ($P(w_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$) y comparación empírica entre tareas deterministas y tareas abiertas.
# MAGIC * **Requisitos previos:** Clúster o SQL Warehouse en Databricks con acceso a los endpoints `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración y Función Auxiliar de Invocación con `ai_query()`
# MAGIC Definimos una función en Python que construye la consulta SQL hacia `ai_query()` pasando `modelParameters => named_struct(...)`. Incluimos un mecanismo de respaldo (*fallback*) automático en caso de que el entorno SQL tenga restricciones de versión sobre `modelParameters`.

# COMMAND ----------

import time
from typing import Dict, Any, List
import pandas as pd

MODELO_RAPIDO = "databricks-gpt-oss-20b"
MODELO_POTENTE = "databricks-gpt-oss-120b"

def ejecutar_ai_query(
    prompt: str,
    modelo: str = MODELO_RAPIDO,
    temperature: float = 0.0,
    max_tokens: int = 300
) -> Dict[str, Any]:
    """
    Ejecuta ai_query() en Databricks controlando la temperatura y mide el tiempo de respuesta.
    """
    prompt_escapado = prompt.replace("\\", "\\\\").replace("'", "''")
    inicio = time.time()
    
    sql_con_parametros = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_escapado}',
        modelParameters => named_struct(
            'temperature', CAST({float(temperature)} AS DOUBLE),
            'max_tokens', CAST({int(max_tokens)} AS INT)
        )
    ) AS respuesta
    """
    
    try:
        fila = spark.sql(sql_con_parametros).collect()[0]
        texto = fila["respuesta"]
        modo = "modelParameters_nativo"
    except Exception as e:
        sql_basico = f"SELECT ai_query('{modelo}', '{prompt_escapado}') AS respuesta"
        fila = spark.sql(sql_basico).collect()[0]
        texto = fila["respuesta"]
        modo = f"fallback_basico ({type(e).__name__})"
        
    latencia = round(time.time() - inicio, 3)
    return {
        "modelo": modelo,
        "temperature": temperature,
        "latencia_seg": latencia,
        "longitud_chars": len(texto) if texto else 0,
        "modo_ejecucion": modo,
        "respuesta": texto.strip() if texto else ""
    }

print("Función ejecutar_ai_query() lista para la demostración.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experimento A: Determinismo en Generación de Código SQL (`0.0` vs `0.7` vs `1.0`)
# MAGIC Ejecutaremos **2 corridas independientes por cada nivel de temperatura** pidiendo una consulta SQL sobre un esquema fijo. Observaremos si las salidas entre la Corrida 1 y la Corrida 2 son idénticas o si sufren deriva sintáctica.

# COMMAND ----------

prompt_sql = """
### ROL
Actúa como un Ingeniero de Datos en Databricks SQL.

### ESQUEMA DE TABLA
Tabla: main.finanzas.pagos_clientes
Columnas:
- pago_id (STRING)
- cliente_id (STRING)
- fecha_pago (DATE)
- monto_usd (DECIMAL(12,2))
- metodo_pago (STRING: 'TARJETA', 'PSE', 'TRANSFERENCIA')
- estado (STRING: 'EXITOSO', 'RECHAZADO', 'PENDIENTE')

### TAREA
Escribe una consulta SQL que obtenga el total pagado y el conteo de transacciones exitosas por metodo_pago durante el año 2026, ordenado de mayor a menor monto total.

### RESTRICCIONES
Devuelve únicamente la sentencia SQL en texto plano, sin explicaciones ni bloques markdown.
""".strip()

temperaturas_prueba = [0.0, 0.7, 1.0]
registros_experimento: List[Dict[str, Any]] = []

for temp in temperaturas_prueba:
    corrida_1 = ejecutar_ai_query(prompt_sql, modelo=MODELO_RAPIDO, temperature=temp, max_tokens=250)
    corrida_2 = ejecutar_ai_query(prompt_sql, modelo=MODELO_RAPIDO, temperature=temp, max_tokens=250)
    
    son_identicas = (corrida_1["respuesta"] == corrida_2["respuesta"])
    
    registros_experimento.append({
        "Temperature": temp,
        "¿Corrida 1 == Corrida 2?": "SÍ (100% Determinista)" if son_identicas else "NO (Hay Variación)",
        "Latencia Promedio (s)": round((corrida_1["latencia_seg"] + corrida_2["latencia_seg"]) / 2, 3),
        "Muestra Corrida 1": corrida_1["respuesta"][:180].replace("\n", " "),
        "Muestra Corrida 2": corrida_2["respuesta"][:180].replace("\n", " ")
    })

df_resultados_sql = pd.DataFrame(registros_experimento)
display(spark.createDataFrame(df_resultados_sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Experimento B: Varianza Léxica en Tarea Creativa / Exploratoria
# MAGIC Ahora evaluamos una tarea donde **sí deseamos diversidad**: proponer 3 eslóganes y casos de uso innovadores para una plataforma de analítica predictiva en tiempo real. Calcularemos la **riqueza de vocabulario (tokens únicos)** generada en cada nivel de temperatura.

# COMMAND ----------

prompt_creativo = """
Propón 3 nombres creativos de productos de datos y un caso de uso disruptivo de IA Generativa para un banco digital latinoamericano. Sé conciso (máximo 90 palabras).
""".strip()

metricas_creativas = []

for temp in [0.0, 0.7, 1.0]:
    salidas = [
        ejecutar_ai_query(prompt_creativo, modelo=MODELO_POTENTE, temperature=temp, max_tokens=220)["respuesta"]
        for _ in range(2)
    ]
    texto_combinado = " ".join(salidas).lower().split()
    vocabulario_unico = len(set(texto_combinado))
    total_palabras = len(texto_combinado)
    tasa_diversidad = round(vocabulario_unico / max(total_palabras, 1), 3)
    
    metricas_creativas.append({
        "Modelo": MODELO_POTENTE,
        "Temperature": temp,
        "Palabras Totales (2 corridas)": total_palabras,
        "Palabras Únicas": vocabulario_unico,
        "Índice de Diversidad Léxica": tasa_diversidad,
        "Corrida_1_Preview": salidas[0][:140].replace("\n", " ")
    })

df_creativo = pd.DataFrame(metricas_creativas)
display(spark.createDataFrame(df_creativo))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Conclusión Pedagógica
# MAGIC * **`temperature = 0.0`:** Garantiza reproducibilidad entre ejecuciones. Es obligatorio en generación de SQL, extracción de JSON y pruebas automatizadas.
# MAGIC * **`temperature = 0.7`:** Ofrece fluidez natural y variación moderada, ideal para redacción de resúmenes ejecutivos o explicaciones a usuarios.
# MAGIC * **`temperature = 1.0`:** Maximiza el índice de diversidad léxica para sesiones de ideación (*brainstorming*), pero introduce variabilidad indeseada cuando se requiere sintaxis estricta.

# COMMAND ----------

# Salida esperada:
# 1. Tabla comparativa mostrando que con Temperature = 0.0 la Corrida 1 y la Corrida 2 producen exactamente la misma consulta SQL ("SÍ (100% Determinista)"), mientras que con 0.7 y 1.0 aparecen variaciones en alias o formato.
# 2. Tabla de diversidad léxica con databricks-gpt-oss-120b donde el Índice de Diversidad Léxica se incrementa progresivamente al pasar de Temperature 0.0 a 0.7 y 1.0.
