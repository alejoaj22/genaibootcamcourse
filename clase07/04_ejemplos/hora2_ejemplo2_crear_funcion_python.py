# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 2 — Crear una función Python en Unity Catalog y ejecutarla de forma remota
# MAGIC **Objetivo:** registrar `generate_checkout_url` (del notebook 2.3) y una función propia `calculate_installments` con `DatabricksFunctionClient`, ejecutarlas a través de UC y observar cómo se manejan los errores.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** los type hints y el docstring de una función Python se convierten en el esquema (parámetros y descripción) que lee el LLM.
# MAGIC **Requisitos previos:** `Includes/Lab_Setup_02`; permiso `CREATE FUNCTION`; cómputo **serverless** (o un SQL warehouse configurado para el cliente).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade unitycatalog-ai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

# 1) Función oficial del notebook 2.3 (la definimos de nuevo para que este ejemplo sea autónomo).
def generate_checkout_url(user_id: str, phone_id: int, valid_quantity: int) -> str:
    """
    Returns the checkout URL to fulfill the user's order for a phone with a given quantity.
    The URL is valid for 30 minutes.

    Args:
        user_id (str): Identifier of the current user/session.
        phone_id (int): ID of the phone to buy.
        valid_quantity (int): the requested quantity, validated against current stock

    Returns:
        str: Checkout URL string.
    """
    from datetime import datetime, timedelta
    from urllib.parse import urlencode

    expiry_time = datetime.now() + timedelta(minutes=30)
    params = {
        "i": phone_id,
        "q": valid_quantity,
        "u": user_id,
        "t": expiry_time
    }

    # Return checkout URL
    checkout_url = f"https://www.derar.cloud/checkout?{urlencode(params)}"
    return checkout_url

# Prueba local (se ejecuta en este cómputo, todavía NO en Unity Catalog).
print(generate_checkout_url("derar@derar.cloud", 4, 2))

# COMMAND ----------

# 2) Función propia: cuotas mensuales sin interés. Reglas para que sea una buena herramienta:
#    - Imports DENTRO de la función (se ejecuta aislada en el servidor).
#    - Type hints en todos los parámetros y en el retorno.
#    - Docstring con descripción, Args y Returns (es lo que lee el LLM).
#    - Sin parámetros por defecto, *args ni **kwargs.
#    - Validación de entradas: devuelve un mensaje claro en vez de lanzar una excepción.
def calculate_installments(price: float, months: int) -> str:
    """
    Calculates the monthly installment for a phone purchase paid in equal installments with no interest.
    Use it when the customer asks how much they would pay per month.

    Args:
        price (float): Total price of the phone in dollars.
        months (int): Number of monthly installments (between 1 and 36).

    Returns:
        str: A sentence with the number of installments and the amount of each one, or an error message.
    """
    if months < 1 or months > 36:
        return "ERROR: months must be between 1 and 36."
    if price <= 0:
        return "ERROR: price must be greater than zero."
    cuota = round(price / months, 2)
    return f"{months} installments of {cuota} dollars each (total {round(cuota * months, 2)})."

# Prueba local.
print(calculate_installments(1799.0, 12))
print(calculate_installments(1799.0, 0))

# COMMAND ----------

# 3) Registro de ambas funciones en Unity Catalog con DatabricksFunctionClient.
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

info_checkout = client.create_python_function(func=generate_checkout_url,
                                              catalog=course.catalog,
                                              schema=course.schema,
                                              replace=True)

info_cuotas = client.create_python_function(func=calculate_installments,
                                            catalog=course.catalog,
                                            schema=course.schema,
                                            replace=True)

print("Registradas:")
print(" -", info_checkout.full_name)
print(" -", info_cuotas.full_name)

# COMMAND ----------

# 4) Inspección: lo que quedó guardado en Unity Catalog (comentario, parámetros y tipos).
for nombre in ("generate_checkout_url", "calculate_installments"):
    fi = client.get_function(f"{course.catalog}.{course.schema}.{nombre}")
    print("=" * 70)
    print("Función:", fi.full_name)
    print("Comentario:", (fi.comment or "")[:200].replace("\n", " "))
    for p in (fi.input_params.parameters if fi.input_params else []):
        print(f"  - parámetro {p.name}: {p.type_text} · {p.comment}")

# COMMAND ----------

# 5) Ejecución REMOTA a través de UC (es lo que hará el agente con cada tool call).
nombre_cuotas = f"{course.catalog}.{course.schema}.calculate_installments"

resultado = client.execute_function(nombre_cuotas, {"price": 1799.0, "months": 12})
print("Caso válido →", resultado.value)

# COMMAND ----------

# 6) Casos límite: la validación interna responde con un mensaje; un tipo inválido produce un error del sistema.
resultado = client.execute_function(nombre_cuotas, {"price": 1799.0, "months": 0})
print("months = 0 →", resultado.value)

try:
    resultado = client.execute_function(nombre_cuotas, {"price": 1799.0, "months": "doce"})
    print("months = 'doce' → error:", resultado.error, "| valor:", resultado.value)
except Exception as e:
    print("months = 'doce' → excepción del cliente:", str(e)[:200])

# COMMAND ----------

# 7) Cadena típica del agente: el resultado de una herramienta alimenta a la siguiente.
nombre_checkout = f"{course.catalog}.{course.schema}.generate_checkout_url"
cuotas = client.execute_function(nombre_cuotas, {"price": 999.0, "months": 6}).value
enlace = client.execute_function(nombre_checkout, {"user_id": "estudiante@nodo.eafit.edu.co",
                                                    "phone_id": 4, "valid_quantity": 1}).value
print("Cuotas:", cuotas)
print("Enlace:", enlace)

# COMMAND ----------

# Salida esperada:
# - Prueba local: una URL https://www.derar.cloud/checkout?i=4&q=2&u=derar%40derar.cloud&t=... y "12 installments of 149.92 dollars each (total 1799.04)." / "ERROR: months must be between 1 and 36."
# - Registro: dos líneas con <catálogo>.genai_course.generate_checkout_url y <catálogo>.genai_course.calculate_installments.
# - Inspección: cada función muestra su comentario (primeras líneas del docstring) y sus parámetros con tipo.
# - Ejecución remota: "Caso válido → 12 installments of 149.92 dollars each (total 1799.04)."; months = 0 devuelve el mensaje ERROR; "doce" genera un error o excepción controlada.
# - Cadena final: "Cuotas: 6 installments of 166.5 dollars each (total 999.0)." y un enlace de compra con el id del usuario.
