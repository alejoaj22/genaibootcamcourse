# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 3 — Comparar el mismo modelo CON y SIN herramientas
# MAGIC **Objetivo:** demostrar por qué un LLM necesita herramientas para responder con datos reales: comparar la respuesta, las llamadas a herramientas, la latencia y el acierto con y sin una herramienta de inventario.
# MAGIC **Tiempo estimado:** 8–10 minutos
# MAGIC **Concepto que ilustra:** tool calling (function calling) manual, alucinación sin datos externos y costo en latencia de usar herramientas.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup_02` (tabla `inventory_stock`), acceso a Foundation Model APIs (`databricks-gpt-oss-120b`), cómputo serverless o DBR 15.4+.
# MAGIC
# MAGIC > La herramienta de este ejemplo es una función **local** de Python (no una UC Function). En la Hora 2 la reemplazaremos por una función gobernada en Unity Catalog.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

# 1) Cliente OpenAI apuntando a los endpoints de Databricks y selección de modelos disponibles.
import json
import time
import pandas as pd
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
cliente = w.serving_endpoints.get_open_ai_client()

CANDIDATOS = ["databricks-gpt-oss-120b", "databricks-gpt-oss-20b", "databricks-meta-llama-3-3-70b-instruct"]
nombres_endpoints = {e.name for e in w.serving_endpoints.list()}
MODELOS = [m for m in CANDIDATOS if m in nombres_endpoints][:2]   # comparamos como máximo 2 para ahorrar tiempo

if not MODELOS:
    raise RuntimeError("No se encontró ningún endpoint candidato. Revisa Serving → Endpoints y ajusta CANDIDATOS.")
print("Modelos que compararemos:", MODELOS)

# COMMAND ----------

# 2) La herramienta: consulta REAL a la tabla inventory_stock (la misma que usará la UC Function check_inventory).
def consultar_inventario(phone_id: int) -> str:
    """Devuelve las unidades disponibles de un teléfono como texto JSON."""
    filas = spark.sql(f"SELECT item_id, quantity FROM inventory_stock WHERE item_id = {int(phone_id)}").collect()
    if not filas:
        return json.dumps({"item_id": int(phone_id), "quantity": None, "nota": "sin registro"})
    return json.dumps({"item_id": int(filas[0]["item_id"]), "quantity": int(filas[0]["quantity"])})

# Especificación que ve el LLM (formato OpenAI): nombre, descripción y parámetros. Nunca ve el código.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "consultar_inventario",
        "description": "Consulta las unidades disponibles en inventario de un teléfono dPhone dado su ID numérico.",
        "parameters": {
            "type": "object",
            "properties": {"phone_id": {"type": "integer", "description": "ID numérico del teléfono (por ejemplo 4)."}},
            "required": ["phone_id"],
        },
    },
}]

# COMMAND ----------

# 3) Preguntas de prueba con respuesta correcta conocida (tomada de la propia tabla).
muestra = spark.sql("SELECT item_id, quantity FROM inventory_stock ORDER BY item_id LIMIT 3").collect()
PREGUNTAS = [
    {"phone_id": int(f["item_id"]), "verdad": int(f["quantity"]),
     "texto": f"¿Cuántas unidades hay en inventario del teléfono dPhone con id {int(f['item_id'])}? Responde con el número."}
    for f in muestra
]
display(pd.DataFrame(PREGUNTAS))

# COMMAND ----------

# 4) Utilidades: extraer texto del mensaje (algunos modelos devuelven una lista de bloques) y medir tiempos.
def texto_de(mensaje):
    contenido = mensaje.content
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido
    # Lista de bloques: nos quedamos con los de tipo texto e ignoramos el razonamiento interno.
    return " ".join(b.get("text", "") for b in contenido if isinstance(b, dict) and b.get("type") == "text")

SISTEMA = ("Eres un asistente de la tienda dPhone. Responde de forma breve. "
           "Si no tienes un dato verificable, dilo claramente.")

def responder(modelo, pregunta, con_tools, max_iter=4):
    """Devuelve (texto, número de llamadas a herramienta, segundos)."""
    mensajes = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": pregunta}]
    llamadas = 0
    inicio = time.time()
    for _ in range(max_iter):
        args = dict(model=modelo, messages=mensajes, max_tokens=800)
        if con_tools:
            args["tools"] = TOOLS
        resp = cliente.chat.completions.create(**args)
        msg = resp.choices[0].message
        if not (con_tools and msg.tool_calls):
            return texto_de(msg).strip(), llamadas, time.time() - inicio
        # El modelo pidió herramientas: las ejecutamos nosotros y devolvemos el resultado (observación).
        mensajes.append({"role": "assistant", "content": texto_de(msg) or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            llamadas += 1
            try:
                resultado = consultar_inventario(**json.loads(tc.function.arguments))
            except Exception as e:
                resultado = f"ERROR: {e}"
            mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": resultado})
    return "Se alcanzó el máximo de iteraciones.", llamadas, time.time() - inicio

# COMMAND ----------

# 5) Ejecución: cada modelo × {sin herramientas, con herramientas} × cada pregunta.
resultados = []
for modelo in MODELOS:
    for pregunta in PREGUNTAS:
        for con_tools in (False, True):
            try:
                texto, llamadas, segundos = responder(modelo, pregunta["texto"], con_tools)
            except Exception as e:
                texto, llamadas, segundos = f"ERROR: {str(e)[:150]}", 0, 0.0
            acierto = str(pregunta["verdad"]) in texto
            resultados.append({
                "modelo": modelo, "modo": "CON herramienta" if con_tools else "SIN herramienta",
                "phone_id": pregunta["phone_id"], "verdad": pregunta["verdad"],
                "llamadas_tool": llamadas, "latencia_s": round(segundos, 1), "acierto": acierto,
                "respuesta": texto[:220].replace("\n", " ")})
tabla = pd.DataFrame(resultados)
display(tabla)

# COMMAND ----------

# 6) Resumen por modelo y modo: tasa de acierto y latencia media.
resumen = (tabla.groupby(["modelo", "modo"])
           .agg(acierto_pct=("acierto", lambda s: round(100 * s.mean(), 0)),
                latencia_media_s=("latencia_s", "mean"),
                llamadas_tool=("llamadas_tool", "sum"))
           .reset_index())
resumen["latencia_media_s"] = resumen["latencia_media_s"].round(1)
display(resumen)

# COMMAND ----------

# 7) Preguntas para la discusión en clase (se imprimen para proyectarlas).
print("1) Sin herramienta, ¿el modelo se negó a responder o inventó un número? Mira la columna 'respuesta'.")
print("2) Con herramienta, ¿qué ganó el modelo y qué costó en latencia (dos llamadas al LLM en lugar de una)?")
print("3) ¿Qué pasaría si la tabla cambiara mañana? ¿Habría que reentrenar el modelo? (pista: RAG/herramientas vs fine-tuning)")

# COMMAND ----------

# Salida esperada:
# - Lista de 1 o 2 modelos disponibles (normalmente databricks-gpt-oss-120b).
# - Tabla con 3 preguntas × 2 modos × N modelos. SIN herramienta: acierto casi siempre False (el modelo se niega o inventa); llamadas_tool = 0.
# - CON herramienta: llamadas_tool = 1 por pregunta y acierto True (el número coincide con la tabla inventory_stock).
# - La latencia CON herramienta suele ser mayor (dos llamadas al LLM). Los valores exactos varían según la carga del endpoint.
# - Resumen agregado por modelo y modo, con acierto_pct cercano a 0 % sin herramienta y 100 % con herramienta.
