# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (starter) — Crear 2 funciones propias en Unity Catalog
# MAGIC **Objetivo:** completar una función SQL con parámetro y una función Python, registrarlas en Unity Catalog, probarlas y pasar la auditoría automática.
# MAGIC **Tiempo estimado:** 8–10 minutos
# MAGIC **Concepto que ilustra:** el nombre, la descripción y los parámetros de una función son la "interfaz" que lee el LLM.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup_02` (tablas `smartphones`, `inventory_stock`), permiso `CREATE FUNCTION`, cómputo serverless.
# MAGIC
# MAGIC Complete todo lo marcado con `TODO`. Mientras queden `TODO`, la auditoría mostrará ❌ (es lo esperado).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade unitycatalog-ai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

# Nombres de las funciones (puedes cambiarlos si inventas otras dos; deben coincidir con las celdas siguientes).
NOMBRE_FUNCION_SQL = "get_phones_under_price"
NOMBRE_FUNCION_PY = "estimate_delivery_date"
print("Esquema destino:", f"{course.catalog}.{course.schema}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- FUNCIÓN 1 (SQL con parámetro).
# MAGIC -- TODO 1a: escribe el COMMENT de la función (qué hace, cuándo usarla, qué devuelve; en inglés y sin la palabra TODO).
# MAGIC -- TODO 1b: escribe el COMMENT del parámetro max_price.
# MAGIC -- TODO 1c: usa el parámetro en el WHERE (price <= max_price).
# MAGIC CREATE OR REPLACE FUNCTION get_phones_under_price(max_price INT COMMENT 'TODO: describe el parametro')
# MAGIC RETURNS TABLE(phone_id INT, model STRING, color STRING, price INT)
# MAGIC COMMENT 'TODO: describe que hace, cuando usarla y que devuelve'
# MAGIC RETURN
# MAGIC   SELECT phone_id, model, color, price
# MAGIC   FROM smartphones
# MAGIC   WHERE price <= 0   -- TODO 1c: reemplaza 0 por el parametro max_price
# MAGIC   ORDER BY price DESC;

# COMMAND ----------

# FUNCIÓN 2 (Python).
# TODO 2a: escribe el docstring (descripción + cuándo usarla + Args + Returns).
# TODO 2b: implementa el cuerpo: valida business_days (1 a 30) y suma días hábiles (lunes a viernes) a partir de hoy.
# Recuerda: imports DENTRO de la función, sin parámetros por defecto.
def estimate_delivery_date(business_days: int) -> str:
    """
    TODO: describe que hace la funcion, cuando usarla y que devuelve.

    Args:
        business_days (int): TODO describe el parametro.

    Returns:
        str: TODO describe el retorno.
    """
    from datetime import date, timedelta
    # TODO 2b: si business_days está fuera de 1..30 devuelve "ERROR: ..."
    # TODO 2b: avanza día por día desde hoy contando solo lunes a viernes (date.weekday() < 5)
    return "TODO"

# COMMAND ----------

# Registro de la función Python en Unity Catalog.
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()
info = client.create_python_function(func=estimate_delivery_date,
                                     catalog=course.catalog,
                                     schema=course.schema,
                                     replace=True)
print("Registrada:", info.full_name)

# COMMAND ----------

# PRUEBAS de la función SQL (3 casos). Ajusta los valores si usaste otra función.
# TODO 3a: elige un presupuesto que devuelva al menos 1 fila, uno muy alto y uno que devuelva 0 filas.
precio_minimo = spark.sql("SELECT min(price) AS p FROM smartphones").first()["p"]
CASOS_SQL = {
    "normal (presupuesto = precio mínimo)": precio_minimo,        # TODO 3a: puedes cambiarlo
    "límite (presupuesto muy alto)": 10_000_000,                  # TODO 3a: puedes cambiarlo
    "inválido (presupuesto 0)": 0,                                # TODO 3a: puedes cambiarlo
}
for descripcion, presupuesto in CASOS_SQL.items():
    n = spark.sql(f"SELECT * FROM {NOMBRE_FUNCION_SQL}({presupuesto})").count()
    print(f"{NOMBRE_FUNCION_SQL}({presupuesto}) → {n} fila(s) · {descripcion}")

# COMMAND ----------

# PRUEBAS de la función Python (3 casos), ejecutadas de forma remota a través de Unity Catalog.
nombre_completo_py = f"{course.catalog}.{course.schema}.{NOMBRE_FUNCION_PY}"
# TODO 3b: elige un caso normal, uno límite y uno inválido.
CASOS_PY = {
    "normal (5 días hábiles)": 5,        # TODO 3b: puedes cambiarlo
    "límite (30 días hábiles)": 30,      # TODO 3b: puedes cambiarlo
    "inválido (0 días hábiles)": 0,      # TODO 3b: puedes cambiarlo
}
for descripcion, dias in CASOS_PY.items():
    r = client.execute_function(nombre_completo_py, {"business_days": dias})
    print(f"{NOMBRE_FUNCION_PY}({dias}) → {r.value if r.error is None else r.error} · {descripcion}")

# COMMAND ----------

# AUDITORÍA AUTOMÁTICA (no editar): verifica la calidad de los metadatos de cada función.
import pandas as pd

def auditar_funcion(nombre_corto):
    fi = client.get_function(f"{course.catalog}.{course.schema}.{nombre_corto}")
    comentario = fi.comment or ""
    params = fi.input_params.parameters if (fi.input_params and fi.input_params.parameters) else []
    return {
        "funcion": nombre_corto,
        "comentario_>=40_caracteres": len(comentario) >= 40,
        "sin_TODO": ("TODO" not in comentario) and all("TODO" not in (p.comment or "") for p in params),
        "parametros_descritos": all((p.comment or "").strip() != "" for p in params) if params else False,
        "dice_cuando_usarla": "use" in comentario.lower(),
    }

auditoria = pd.DataFrame([auditar_funcion(NOMBRE_FUNCION_SQL), auditar_funcion(NOMBRE_FUNCION_PY)])
display(auditoria)
aprobadas = int(auditoria.drop(columns="funcion").all(axis=1).sum())
print(f"Funciones aprobadas: {aprobadas} de 2 " + ("✅" if aprobadas == 2 else "❌ (completa los TODO)"))

# COMMAND ----------

# TODO 4: completa la frase con lo aprendido.
REFLEXION = "Una descripción ambigua podría hacer que el agente: TODO"
print(REFLEXION)

# COMMAND ----------

# Salida esperada:
# - Con los TODO sin completar: la función SQL devuelve 0 filas en todos los casos, la función Python devuelve "TODO" y la auditoría muestra False en comentario/sin_TODO/parametros → "Funciones aprobadas: 0 de 2 ❌ (completa los TODO)".
# - Con la solución completa: pruebas SQL → ≥1 fila (presupuesto mínimo), todas las filas (presupuesto muy alto) y 0 filas (presupuesto 0); pruebas Python → una fecha YYYY-MM-DD en día hábil para 5 y 30, y "ERROR: ..." para 0.
# - Auditoría con todas las columnas en True y "Funciones aprobadas: 2 de 2 ✅".
