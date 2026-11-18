# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 3 — Listar las herramientas del agente y ver lo que recibe el LLM
# MAGIC **Objetivo:** listar las funciones de Unity Catalog del esquema, convertirlas en *tool specs* con `UCFunctionToolkit`, auditar la calidad de sus descripciones y revisar permisos.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** el contrato entre el agente y sus herramientas (nombre, descripción, parámetros en JSON) y el costo en tokens de cada herramienta.
# MAGIC **Requisitos previos:** funciones del notebook 2.3 creadas (`get_latest_phone`, `check_inventory`, `get_current_user`, `generate_checkout_url`); opcional: índice `dphone_vs_index` (Clase 6).

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-openai unitycatalog-ai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

# 1) Listado de funciones del esquema con SQL (siempre disponible).
catalogo, esquema = course.catalog, course.schema
funciones = spark.sql(f"SHOW USER FUNCTIONS IN {catalogo}.{esquema}")
display(funciones)

# COMMAND ----------

# 2) Listado con el cliente de unitycatalog-ai (lo que usa el toolkit por debajo).
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()
try:
    lista = list(client.list_functions(catalog=catalogo, schema=esquema, max_results=50))
    for f in lista:
        print("-", f.full_name)
except Exception as e:
    print("No se pudo listar con el cliente (se usa solo el listado SQL):", str(e)[:200])

# COMMAND ----------

# 3) UCFunctionToolkit: convierte TODAS las funciones del esquema (comodín *) en tool specs.
from databricks_openai import UCFunctionToolkit

toolkit = UCFunctionToolkit(function_names=[f"{catalogo}.{esquema}.*"])
print(f"El agente recibiría {len(toolkit.tools)} herramientas de UC.")
for t in toolkit.tools:
    print("-", t["function"]["name"])

# Nota: el nombre usa "__" en lugar de "." porque los nombres de tools no admiten puntos.

# COMMAND ----------

# 4) La tool spec completa de UNA herramienta: esto (y solo esto) es lo que ve el LLM.
import json

spec = next((t for t in toolkit.tools if t["function"]["name"].endswith("check_inventory")), toolkit.tools[0])
print(json.dumps(spec, indent=2, ensure_ascii=False))

# COMMAND ----------

# 5) Auditoría de calidad de las descripciones (reglas simples y objetivas).
import pandas as pd

def auditar_tool(tool):
    f = tool["function"]
    descr = f.get("description") or ""
    props = f.get("parameters", {}).get("properties", {})
    params_sin_descr = [p for p, v in props.items() if not v.get("description")]
    return {
        "tool": f["name"].replace("__", "."),
        "desc_>=40_chars": len(descr) >= 40,
        "n_parametros": len(props),
        "params_sin_descripcion": ", ".join(params_sin_descr) or "-",
        "dice_cuando_usar": any(k in descr.lower() for k in ("use it when", "use when", "when the", "used for", "returns")),
        "tokens_aprox": len(json.dumps(tool)) // 4,   # regla práctica: ~4 caracteres por token
    }

auditoria = pd.DataFrame([auditar_tool(t) for t in toolkit.tools])
display(auditoria)
print("Tokens aprox. que consumen TODAS las herramientas en CADA llamada al LLM:", int(auditoria["tokens_aprox"].sum()))

# COMMAND ----------

# 6) Permisos: ¿quién puede ejecutar cada función? (puede fallar si no eres propietario; se captura el error).
for fila in funciones.collect()[:4]:
    nombre_funcion = fila[0]
    try:
        display(spark.sql(f"SHOW GRANTS ON FUNCTION {nombre_funcion}"))
    except Exception as e:
        print(f"{nombre_funcion}: no se pudieron listar permisos ({str(e)[:120]})")

# COMMAND ----------

# 7) Cómo se vería conceder la ejecución a un grupo (NO se ejecuta; solo para discutir el menor privilegio).
sentencia = f"GRANT EXECUTE ON FUNCTION {catalogo}.{esquema}.check_inventory TO `account users`"
print("Ejemplo (no ejecutado):")
print(sentencia)

# COMMAND ----------

# 8) La otra herramienta del agente: el índice vectorial como tool (si el índice existe).
from databricks_openai import VectorSearchRetrieverTool

try:
    vs_tool = VectorSearchRetrieverTool(
        index_name=f"{catalogo}.{esquema}.dphone_vs_index",
        tool_description="Searches dPhone manuals, datasheets and FAQs (in English): specs, screen, camera, battery, warranty.",
    )
    print(json.dumps(vs_tool.tool, indent=2, ensure_ascii=False))
except Exception as e:
    print("No se pudo crear la herramienta del índice (¿existe dphone_vs_index?):", str(e)[:200])

# COMMAND ----------

# Salida esperada:
# - SHOW USER FUNCTIONS: al menos get_latest_phone, check_inventory, get_current_user y generate_checkout_url (más las que hayas creado en otros ejemplos).
# - "El agente recibiría N herramientas de UC." con nombres del estilo <catálogo>__genai_course__check_inventory.
# - Una tool spec JSON con "type": "function", "name", "description" y "parameters" (con phone_id: integer y su descripción).
# - Tabla de auditoría: todas con descripción ≥ 40 caracteres y sin parámetros sin descripción; tokens_aprox por herramienta de ~100–250 y un total de ~500–1.500.
# - Permisos: tabla de privilegios por función o un aviso controlado si no eres propietario.
# - Tool spec del índice vectorial (nombre derivado del índice y la descripción que escribimos) o un aviso si el índice no existe.
