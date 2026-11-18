# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 2 (solución) — Crear 2 funciones propias en Unity Catalog
# MAGIC **Objetivo:** solución completa y comentada: función SQL `get_phones_under_price`, función Python `estimate_delivery_date`, pruebas con casos normal/límite/inválido y auditoría en verde.
# MAGIC **Tiempo estimado:** 8–10 minutos
# MAGIC **Concepto que ilustra:** el nombre, la descripción y los parámetros de una función son la "interfaz" que lee el LLM.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup_02` (tablas `smartphones`, `inventory_stock`), permiso `CREATE FUNCTION`, cómputo serverless.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade unitycatalog-ai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

NOMBRE_FUNCION_SQL = "get_phones_under_price"
NOMBRE_FUNCION_PY = "estimate_delivery_date"
print("Esquema destino:", f"{course.catalog}.{course.schema}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- FUNCIÓN 1 (SQL con parámetro). Decisiones de diseño:
# MAGIC --  - Nombre verbo_sustantivo, específico: get_phones_under_price.
# MAGIC --  - El parámetro max_price tiene un nombre distinto al de la columna price (sin ambigüedad).
# MAGIC --  - El COMMENT dice qué hace, cuándo usarla y qué devuelve.
# MAGIC --  - ORDER BY price DESC: primero la mejor opción dentro del presupuesto.
# MAGIC CREATE OR REPLACE FUNCTION get_phones_under_price(
# MAGIC   max_price INT COMMENT 'Maximum budget in dollars (integer). Phones priced at or below this value are returned.'
# MAGIC )
# MAGIC RETURNS TABLE(phone_id INT, model STRING, color STRING, price INT)
# MAGIC COMMENT 'Lists the smartphones whose price is less than or equal to a budget, from the most expensive to the cheapest. Use it when the customer gives a maximum budget. Returns zero rows if no phone fits the budget.'
# MAGIC RETURN
# MAGIC   SELECT phone_id, model, color, price
# MAGIC   FROM smartphones
# MAGIC   WHERE price <= max_price
# MAGIC   ORDER BY price DESC;

# COMMAND ----------

# FUNCIÓN 2 (Python). Decisiones de diseño:
#  - Imports dentro de la función (se ejecuta aislada en el servidor).
#  - Validación de rango: devuelve "ERROR: ..." en lugar de lanzar una excepción.
#  - Docstring completo (qué hace, cuándo usarla, Args, Returns): es lo que lee el LLM.
def estimate_delivery_date(business_days: int) -> str:
    """
    Estimates the delivery date of an order counting only business days (Monday to Friday), starting from today.
    Use it when the customer asks when their phone will arrive.

    Args:
        business_days (int): Number of business days the shipping takes (between 1 and 30).

    Returns:
        str: Estimated delivery date in YYYY-MM-DD format, or an error message starting with ERROR.
    """
    from datetime import date, timedelta

    if business_days < 1 or business_days > 30:
        return "ERROR: business_days must be between 1 and 30."

    fecha = date.today()
    restantes = business_days
    while restantes > 0:
        fecha += timedelta(days=1)
        if fecha.weekday() < 5:        # 0 = lunes ... 4 = viernes; sábado y domingo no cuentan
            restantes -= 1
    return fecha.isoformat()

# Prueba local rápida (antes de registrar).
print("Prueba local (5 días hábiles):", estimate_delivery_date(5))

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

# PRUEBAS de la función SQL: normal, límite e inválido (comparadas contra un conteo directo de la tabla).
precio_minimo = spark.sql("SELECT min(price) AS p FROM smartphones").first()["p"]
total_telefonos = spark.sql("SELECT count(*) AS n FROM smartphones WHERE price IS NOT NULL").first()["n"]

CASOS_SQL = {
    "normal (presupuesto = precio mínimo)": (precio_minimo, None),
    "límite (presupuesto muy alto)": (10_000_000, total_telefonos),
    "inválido (presupuesto 0)": (0, 0),
}
resultados_sql = []
for descripcion, (presupuesto, esperado) in CASOS_SQL.items():
    n = spark.sql(f"SELECT * FROM {NOMBRE_FUNCION_SQL}({presupuesto})").count()
    directo = spark.sql(f"SELECT count(*) AS n FROM smartphones WHERE price <= {presupuesto}").first()["n"]
    ok = (n == directo) and (esperado is None or n == esperado) and (descripcion.startswith("normal") is False or n >= 1)
    resultados_sql.append(ok)
    print(f"{NOMBRE_FUNCION_SQL}({presupuesto}) → {n} fila(s) · {descripcion} · {'✅' if ok else '❌'}")
assert all(resultados_sql), "Alguna prueba SQL falló"

# COMMAND ----------

# PRUEBAS de la función Python (remotas, a través de UC): normal, límite e inválido.
from datetime import date

nombre_completo_py = f"{course.catalog}.{course.schema}.{NOMBRE_FUNCION_PY}"
resultados_py = []
for descripcion, dias in {"normal (5 días hábiles)": 5, "límite (30 días hábiles)": 30, "inválido (0 días hábiles)": 0}.items():
    r = client.execute_function(nombre_completo_py, {"business_days": dias})
    valor = r.value if r.error is None else r.error
    if dias == 0:
        ok = str(valor).startswith("ERROR")
    else:
        ok = (date.fromisoformat(str(valor)).weekday() < 5) and (date.fromisoformat(str(valor)) > date.today())
    resultados_py.append(ok)
    print(f"{NOMBRE_FUNCION_PY}({dias}) → {valor} · {descripcion} · {'✅' if ok else '❌'}")
assert all(resultados_py), "Alguna prueba Python falló"

# COMMAND ----------

# AUDITORÍA AUTOMÁTICA: verifica la calidad de los metadatos de cada función.
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
print(f"Funciones aprobadas: {aprobadas} de 2 " + ("✅" if aprobadas == 2 else "❌ (revisa los comentarios)"))

# COMMAND ----------

# Comprobamos que ambas son visibles como funciones del esquema (el agente de la Hora 3 las tomará por el comodín *).
funciones = [r[0].split(".")[-1] for r in spark.sql(f"SHOW USER FUNCTIONS IN {course.catalog}.{course.schema}").collect()]
print("Funciones del esquema:", funciones)
assert NOMBRE_FUNCION_SQL in funciones and NOMBRE_FUNCION_PY in funciones

REFLEXION = ("Una descripción ambigua podría hacer que el agente elija la herramienta equivocada, invente argumentos "
             "(por ejemplo un presupuesto en otra moneda) o no la use cuando debería.")
print(REFLEXION)

# COMMAND ----------

# Salida esperada:
# - Esquema destino: <catálogo>.genai_course y "Prueba local (5 días hábiles): <fecha YYYY-MM-DD en día hábil>".
# - "Registrada: <catálogo>.genai_course.estimate_delivery_date".
# - Pruebas SQL: normal → ≥1 fila ✅; límite → todas las filas con precio ✅; inválido (0) → 0 filas ✅.
# - Pruebas Python: dos fechas en día hábil (5 y 30 días) ✅ y "ERROR: business_days must be between 1 and 30." para 0 ✅.
# - Auditoría con todas las columnas en True y "Funciones aprobadas: 2 de 2 ✅".
# - La lista de funciones del esquema incluye get_phones_under_price y estimate_delivery_date; se imprime la reflexión final.
