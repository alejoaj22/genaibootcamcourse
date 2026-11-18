# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 2 — Streaming del agente: eventos en vivo y latencia percibida
# MAGIC **Objetivo:** usar `AGENT.predict_stream` para ver los eventos del agente (acciones, observaciones y texto) a medida que ocurren, y comparar el tiempo hasta el primer fragmento de texto con el tiempo total.
# MAGIC **Tiempo estimado:** 8–10 minutos
# MAGIC **Concepto que ilustra:** `predict` vs `predict_stream`, tipos de evento (`response.output_text.delta`, `response.output_item.done`) y latencia percibida.
# MAGIC **Requisitos previos:** haber ejecutado `hora3_ejemplo1_testear_agente.py` (crea el módulo `agent.py` en el volumen `_cache`).

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# 1) Cargamos el agente creado en el ejemplo 1.
import os
import sys
import time
import importlib
from collections import Counter

AGENT_DIR = f"/Volumes/{course.catalog}/{course.schema}/_cache/clase07_agent"
if not os.path.exists(f"{AGENT_DIR}/agent.py"):
    raise FileNotFoundError("No existe agent.py. Ejecuta primero hora3_ejemplo1_testear_agente.py")

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
import agent
importlib.reload(agent)
from utilidades import a_dict, nombre_corto, resumir_respuesta

AGENT = agent.AGENT
print("Agente cargado con", len(agent.TOOL_INFOS), "herramientas.")

# COMMAND ----------

# 2) Streaming de una consulta simple (una sola herramienta): imprimimos cada evento cuando llega.
def pedir(texto, sesion):
    return {"input": [{"role": "user", "content": texto}], "custom_inputs": {"session_id": sesion}}

def transmitir(texto, sesion, mostrar=True):
    """Ejecuta predict_stream y devuelve métricas de tiempo y un registro de eventos."""
    inicio = time.time()
    t_primer_evento = None
    t_primer_texto = None
    registro = []   # (segundos, tipo de evento)
    for evento in AGENT.predict_stream(pedir(texto, sesion)):
        t = time.time() - inicio
        if t_primer_evento is None:
            t_primer_evento = t
        registro.append((t, evento.type))

        if evento.type == "response.output_text.delta":
            if t_primer_texto is None:
                t_primer_texto = t
            if mostrar:
                print(evento.delta, end="", flush=True)   # el texto aparece "en vivo"
        elif evento.type == "response.output_item.done":
            item = a_dict(evento.item)
            if item.get("type") == "function_call" and mostrar:
                print(f"\n[ACCIÓN]      {nombre_corto(item.get('name', ''))}({item.get('arguments')})")
            elif item.get("type") == "function_call_output" and mostrar:
                print(f"[OBSERVACIÓN] {str(item.get('output'))[:160]}")
    total = time.time() - inicio
    return {"primer_evento_s": t_primer_evento, "primer_texto_s": t_primer_texto, "total_s": total}, registro

print("PREGUNTA: What is the screen size of dPhone D1 Fold?\n")
metricas1, registro1 = transmitir("What is the screen size of dPhone D1 Fold?", "clase07-ej2-a")
print("\n\nMétricas:", {k: (round(v, 1) if v is not None else None) for k, v in metricas1.items()})

# COMMAND ----------

# 3) Streaming de una consulta encadenada: se ven acción → observación → acción → observación → texto.
print("PREGUNTA: ¿Hay stock del último teléfono? ¿Cuántas unidades?\n")
metricas2, registro2 = transmitir("¿Hay stock del último teléfono? ¿Cuántas unidades?", "clase07-ej2-b")
print("\n\nMétricas:", {k: (round(v, 1) if v is not None else None) for k, v in metricas2.items()})

# COMMAND ----------

# 4) Conteo de eventos por tipo en cada consulta.
import pandas as pd

conteo = pd.DataFrame({
    "consulta simple": Counter(tipo for _, tipo in registro1),
    "consulta encadenada": Counter(tipo for _, tipo in registro2),
}).fillna(0).astype(int)
display(conteo)

# COMMAND ----------

# 5) Mismo agente con predict (sin streaming): el usuario no ve nada hasta el final.
inicio = time.time()
resp = AGENT.predict(pedir("¿Hay stock del último teléfono? ¿Cuántas unidades?", "clase07-ej2-c"))
total_predict = time.time() - inicio
herramientas, texto = resumir_respuesta(resp)

comparacion = pd.DataFrame([
    {"modo": "predict (bloqueante)", "primer_texto_visible_s": round(total_predict, 1), "total_s": round(total_predict, 1)},
    {"modo": "predict_stream", "primer_texto_visible_s": round(metricas2["primer_texto_s"] or metricas2["total_s"], 1),
     "total_s": round(metricas2["total_s"], 1)},
])
display(comparacion)
print("Herramientas (predict):", herramientas)
print("La diferencia de latencia percibida está en 'primer_texto_visible_s'; el total es similar.")

# COMMAND ----------

# 6) Línea de tiempo de eventos de la consulta encadenada (opcional, requiere matplotlib).
try:
    import matplotlib.pyplot as plt

    tipos = sorted({tipo for _, tipo in registro2})
    y = {tipo: i for i, tipo in enumerate(tipos)}
    plt.figure(figsize=(10, 3 + 0.4 * len(tipos)))
    plt.scatter([t for t, _ in registro2], [y[tipo] for _, tipo in registro2], s=18)
    plt.yticks(list(y.values()), list(y.keys()))
    plt.xlabel("Segundos desde el inicio")
    plt.title("Eventos del agente en el tiempo (consulta encadenada)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("No se pudo graficar la línea de tiempo:", e)

# COMMAND ----------

# Salida esperada:
# - Consulta simple: se imprime "[ACCIÓN] search_dphone_docs({...})", luego "[OBSERVACIÓN] ..." con fragmentos del documento y después el texto de la respuesta llegando por partes.
# - Consulta encadenada: dos pares ACCIÓN/OBSERVACIÓN (get_latest_phone → check_inventory) y después el texto final.
# - Métricas: primer_evento_s menor que primer_texto_s, y primer_texto_s menor o igual a total_s (valores típicos: primer evento 1–4 s; total 4–15 s).
# - Tabla de conteo de eventos con más eventos "response.output_text.delta" que cualquier otro tipo.
# - Tabla comparativa: en predict el texto es visible solo al final; en predict_stream aparece antes (misma duración total aproximada).
# - Gráfico de dispersión con los eventos en el tiempo.
