# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 1 — Demo conceptual del ciclo ReAct (sin LLM real)
# MAGIC **Objetivo:** ver, paso a paso y en una tabla, cómo un agente alterna Pensamiento → Acción → Observación hasta responder, y cómo `max_iter` evita un bucle infinito.
# MAGIC **Tiempo estimado:** 8–10 minutos
# MAGIC **Concepto que ilustra:** ciclo ReAct (Reasoning + Acting), tool calling, historial de mensajes y límite de iteraciones.
# MAGIC **Requisitos previos:** ninguno (usa Python puro y datos simulados; no llama a ningún modelo ni a Unity Catalog). Funciona en cualquier cómputo de Databricks.
# MAGIC
# MAGIC > Importante: aquí el "LLM" es una política con reglas fijas (simulada). En la Hora 3 el modelo real (`databricks-gpt-oss-120b`) ocupará su lugar, pero el bucle es el mismo.

# COMMAND ----------

# 1) Datos simulados (en la Hora 2 usaremos las tablas reales smartphones e inventory_stock).
import re
import time
import pandas as pd

TELEFONOS = {
    4: {"phone_id": 4, "model": "dPhone D1 Fold (simulado)", "color": "Black", "price": 1799},
    3: {"phone_id": 3, "model": "dPhone D1 (simulado)", "color": "Blue", "price": 999},
}
INVENTARIO = {4: 25, 3: 0}   # phone_id -> unidades disponibles

# COMMAND ----------

# 2) Herramientas: funciones normales de Python. El "agente" solo conoce su nombre y sus argumentos.
def get_latest_phone():
    """Devuelve el último teléfono lanzado (simulado)."""
    return TELEFONOS[4]

def check_inventory(phone_id: int):
    """Devuelve las unidades disponibles de un teléfono (simulado)."""
    return {"item_id": phone_id, "quantity": INVENTARIO.get(phone_id, 0)}

def get_current_user():
    """Devuelve el usuario de la sesión (simulado)."""
    return "estudiante@nodo.eafit.edu.co"

def generate_checkout_url(user_id: str, phone_id: int, valid_quantity: int):
    """Genera un enlace de compra (simulado, sin efectos reales)."""
    return f"https://tienda.ejemplo/checkout?i={phone_id}&q={valid_quantity}&u={user_id}"

HERRAMIENTAS = {
    "get_latest_phone": get_latest_phone,
    "check_inventory": check_inventory,
    "get_current_user": get_current_user,
    "generate_checkout_url": generate_checkout_url,
}

# COMMAND ----------

# 3) "LLM" simulado: decide la siguiente acción mirando la pregunta y lo que ya observó.
def politica_simulada(pregunta, historial):
    """Devuelve una acción {pensamiento, accion, args} o una respuesta final {respuesta}."""
    llamadas = [h["accion"] for h in historial]
    obs = {h["accion"]: h["observacion"] for h in historial}
    q = pregunta.lower()
    quiere_comprar = ("comprar" in q) or ("buy" in q)
    numeros = re.findall(r"\d+", pregunta)
    cantidad = int(numeros[0]) if numeros else 1

    # Paso 1: ¿cuál es el último teléfono? (el modelo no lo sabe de memoria)
    if "get_latest_phone" not in llamadas:
        return {"pensamiento": "No sé cuál es el último teléfono; necesito consultarlo.",
                "accion": "get_latest_phone", "args": {}}

    telefono = obs["get_latest_phone"]

    # Paso 2: ¿hay stock de ese teléfono?
    if "check_inventory" not in llamadas:
        return {"pensamiento": f"Ya sé que es {telefono['model']} (id {telefono['phone_id']}); ahora debo ver el stock.",
                "accion": "check_inventory", "args": {"phone_id": telefono["phone_id"]}}

    stock = obs["check_inventory"]["quantity"]

    # Si solo preguntaban por el stock, respondemos.
    if not quiere_comprar:
        return {"respuesta": f"Del {telefono['model']} hay {stock} unidades disponibles."}

    # Flujo de compra: validar cantidad, identificar usuario y generar el enlace.
    if cantidad > stock:
        return {"respuesta": f"Solo hay {stock} unidades; no puedo generar el enlace para {cantidad}."}
    if "get_current_user" not in llamadas:
        return {"pensamiento": "Hay stock suficiente. Necesito la identidad del usuario para el enlace.",
                "accion": "get_current_user", "args": {}}
    if "generate_checkout_url" not in llamadas:
        return {"pensamiento": "Tengo teléfono, cantidad validada y usuario: puedo generar el enlace.",
                "accion": "generate_checkout_url",
                "args": {"user_id": obs["get_current_user"], "phone_id": telefono["phone_id"],
                         "valid_quantity": cantidad}}
    return {"respuesta": f"Listo. Tu enlace de compra: {obs['generate_checkout_url']}"}

def politica_defectuosa(pregunta, historial):
    """Una política con error: nunca se decide a responder (simula un agente atascado)."""
    return {"pensamiento": "No estoy seguro; mejor consulto otra vez.",
            "accion": "get_latest_phone", "args": {}}

# COMMAND ----------

# 4) El bucle ReAct. Es muy parecido a call_and_run_tools del notebook 2.4.
def ciclo_react(pregunta, politica=politica_simulada, max_iter=10):
    historial = []   # memoria de corto plazo: lo que hemos hecho y observado
    filas = []       # para mostrar la traza en una tabla
    for paso in range(1, max_iter + 1):
        decision = politica(pregunta, historial)          # PENSAR (en la Hora 3: call_llm)
        if "respuesta" in decision:                       # el modelo decide terminar
            filas.append({"paso": paso, "pensamiento": "Ya tengo lo necesario.",
                          "accion": "responder", "argumentos": "", "observacion": decision["respuesta"]})
            return decision["respuesta"], pd.DataFrame(filas)
        resultado = HERRAMIENTAS[decision["accion"]](**decision["args"])   # ACTUAR + OBSERVAR (handle_tool_call)
        historial.append({"accion": decision["accion"], "args": decision["args"], "observacion": resultado})
        filas.append({"paso": paso, "pensamiento": decision["pensamiento"], "accion": decision["accion"],
                      "argumentos": str(decision["args"]), "observacion": str(resultado)})
    # Protección: se agotaron las iteraciones (equivale al mensaje "Max iterations reached. Stopping.")
    filas.append({"paso": max_iter + 1, "pensamiento": "-", "accion": "STOP", "argumentos": "",
                  "observacion": "Max iterations reached. Stopping."})
    return "Max iterations reached. Stopping.", pd.DataFrame(filas)

# COMMAND ----------

# 5) Caso 1: consulta que requiere encadenar DOS herramientas (la segunda depende de la primera).
respuesta, traza = ciclo_react("¿Hay stock del último teléfono?")
print("RESPUESTA FINAL:", respuesta)
display(traza)

# COMMAND ----------

# 6) Caso 2: flujo de compra con CUATRO herramientas encadenadas.
respuesta, traza = ciclo_react("Quiero comprar 2 unidades del último teléfono")
print("RESPUESTA FINAL:", respuesta)
display(traza)

# COMMAND ----------

# 7) Caso 3: cantidad superior al stock. El agente se detiene ANTES de la acción con efecto.
respuesta, traza = ciclo_react("Quiero comprar 40 unidades del último teléfono")
print("RESPUESTA FINAL:", respuesta)
display(traza)

# COMMAND ----------

# 8) Caso 4: agente atascado. Sin max_iter este bucle sería infinito (costo y latencia sin control).
inicio = time.time()
respuesta, traza = ciclo_react("Quiero comprar 2 unidades", politica=politica_defectuosa, max_iter=5)
print("RESPUESTA FINAL:", respuesta)
print(f"Iteraciones ejecutadas: {len(traza) - 1} (límite = 5) · Tiempo: {time.time() - inicio:.3f} s")
display(traza)

# COMMAND ----------

# 9) Resumen para la discusión: cuántas "llamadas al LLM" costó cada caso.
# En un agente real, cada fila de tipo "acción" implica una llamada al LLM + una ejecución de herramienta,
# y la fila final implica una llamada más al LLM.
casos = {
    "Stock del último teléfono": "¿Hay stock del último teléfono?",
    "Compra de 2 unidades": "Quiero comprar 2 unidades del último teléfono",
    "Compra de 40 unidades": "Quiero comprar 40 unidades del último teléfono",
}
resumen = []
for nombre, pregunta in casos.items():
    _, t = ciclo_react(pregunta)
    acciones = int((~t["accion"].isin(["responder", "STOP"])).sum())
    resumen.append({"caso": nombre, "herramientas_llamadas": acciones, "llamadas_llm_aprox": acciones + 1})
display(pd.DataFrame(resumen))

# COMMAND ----------

# Salida esperada:
# - Caso 1: 3 filas (get_latest_phone → check_inventory → responder). Respuesta: "Del dPhone D1 Fold (simulado) hay 25 unidades disponibles."
# - Caso 2: 5 filas (get_latest_phone → check_inventory → get_current_user → generate_checkout_url → responder) con un enlace de compra.
# - Caso 3: 3 filas; el agente informa que solo hay 25 unidades y NO genera el enlace.
# - Caso 4: 6 filas; la última es STOP con "Max iterations reached. Stopping." (5 iteraciones ejecutadas).
# - Resumen: 2, 4 y 2 herramientas llamadas, con ~3, ~5 y ~3 llamadas al LLM respectivamente.
