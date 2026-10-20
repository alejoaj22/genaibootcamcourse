# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejemplo 3: "Código como Conversación" — Depuración de Causa Raíz (RCA) y Generación de Tests
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Aplicar el flujo **"Código como Conversación"** con `ai_query()` para diagnosticar la causa raíz (RCA) de una función Python de ingeniería de datos con 3 bugs sutiles, refactorizarla con tipado estricto y generar automáticamente una suite de pruebas unitarias verificables en vivo.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Prompting para depuración estructurada (Código + Traceback/Síntoma + Caso Reproductor $\to$ RCA + Refactorización + Tests de Regresión).
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. El Escenario Real: Una Función de Liquidación de Comisiones con 3 Bugs Ocultos
# MAGIC Observemos la siguiente función escrita para procesar lotes de ventas de vendedores. Contiene **3 defectos clásicos y peligrosos** en producción:
# MAGIC 1. **Bug 1 (Argumento mutable por defecto):** `alertas=[]` en la firma de la función hace que las alertas de llamadas anteriores se acumulen entre vendedores distintos.
# MAGIC 2. **Bug 2 (`ZeroDivisionError` / datos vacíos):** Si un vendedor tiene `ventas_validas` vacía (por ejemplo, todas sus ventas fueron anuladas), `sum(...) / len(...)` lanza `ZeroDivisionError`.
# MAGIC 3. **Bug 3 (Error lógico en condicional escalonado):** Evalúa `if total >= 1000:` antes de `elif total >= 5000:`, haciendo que el tramo VIP de `$5,000+` sea **código inalcanzable (*dead code*)**.

# COMMAND ----------

CODIGO_CON_BUGS = '''
def liquidar_comision_vendedor(vendedor_id, transacciones, alertas=[]):
    ventas_validas = [t["monto"] for t in transacciones if t["estado"] == "APROBADA"]
    total = sum(ventas_validas)
    ticket_promedio = total / len(ventas_validas)
    
    if total >= 1000:
        tasa = 0.05
        nivel = "ESTANDAR"
    elif total >= 5000:
        tasa = 0.12
        nivel = "VIP_ORO"
    else:
        tasa = 0.02
        nivel = "INICIAL"
        
    if ticket_promedio < 50:
        alertas.append(f"Ticket bajo para {vendedor_id}")
        
    return {
        "vendedor_id": vendedor_id,
        "total_vendido": total,
        "ticket_promedio": round(ticket_promedio, 2),
        "nivel": nivel,
        "comision_usd": round(total * tasa, 2),
        "alertas": alertas
    }
'''.strip()

print("Código defectuoso cargado para análisis:")
print(CODIGO_CON_BUGS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Demostración del Fallo en Vivo Antes de Depurar
# MAGIC Ejecutamos la función defectuosa con dos vendedores para evidenciar cómo contamina la lista `alertas`, asigna mal el nivel `VIP_ORO` y explota cuando no hay transacciones aprobadas.

# COMMAND ----------

exec(CODIGO_CON_BUGS)

res_v1 = liquidar_comision_vendedor("VEND-01", [{"monto": 30, "estado": "APROBADA"}])
res_v2 = liquidar_comision_vendedor("VEND-02", [{"monto": 6000, "estado": "APROBADA"}])

print("Resultado VEND-01 (ticket de $30):", res_v1)
print("Resultado VEND-02 (vendió $6000 -> debería ser VIP_ORO sin alertas de VEND-01):", res_v2)

try:
    res_v3 = liquidar_comision_vendedor("VEND-03", [{"monto": 500, "estado": "ANULADA"}])
except Exception as err:
    print(f"Excepción crítica en VEND-03 (sin ventas aprobadas): {type(err).__name__}: {err}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Turno 1 de la Conversación con IA: Diagnóstico de Causa Raíz (RCA) y Refactorización
# MAGIC Enviamos el código defectuoso junto con los síntomas observados a `databricks-gpt-oss-120b` con `temperature = 0.0`, pidiendo primero el análisis `[CAUSA_RAIZ]` y luego el bloque `[CODIGO_CORREGIDO]`.

# COMMAND ----------

import re

MODELO_DEBUGGER = "databricks-gpt-oss-120b"

def consultar_debugger(prompt: str, max_tokens: int = 900) -> str:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    sql = f"""
    SELECT ai_query(
        '{MODELO_DEBUGGER}',
        '{prompt_esc}',
        modelParameters => named_struct('temperature', CAST(0.0 AS DOUBLE), 'max_tokens', CAST({int(max_tokens)} AS INT))
    ) AS respuesta
    """
    try:
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{MODELO_DEBUGGER}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return res.strip() if res else ""

prompt_rca_y_refactor = f"""
### ROL
Actúa como un Staff Software Engineer especialista en Python, calidad de código y depuración de causa raíz (RCA).

### CÓDIGO FUENTE CON DEFECTOS
{CODIGO_CON_BUGS}

### SÍNTOMAS Y FALLOS REPORTADOS EN PRODUCCIÓN
1. Las alertas generadas para un vendedor aparecen pegadas en la respuesta de los siguientes vendedores.
2. Un vendedor con ventas aprobadas por $6,000 USD recibe nivel 'ESTANDAR' (tasa 0.05) en lugar de 'VIP_ORO' (tasa 0.12).
3. Cuando un vendedor solo tiene transacciones con estado 'ANULADA', la función colapsa con `ZeroDivisionError: division by zero`.

### TAREA
Responde utilizando estrictamente estas dos secciones delimitadas:

[CAUSA_RAIZ]
Explica técnicamente la causa exacta de cada uno de los 3 bugs indicando la línea o instrucción responsable.
[/CAUSA_RAIZ]

[CODIGO_CORREGIDO]
Escribe únicamente el código Python corregido de la función `liquidar_comision_vendedor` con:
- Type hints de Python (`Optional`, `List`, `Dict`, `Any`).
- Inicialización segura de `alertas: Optional[List[str]] = None`.
- Manejo seguro cuando `ventas_validas` está vacía (`ticket_promedio = 0.0`, sin agregar alerta de ticket bajo si no hubo ventas).
- Orden correcto de evaluación condicional (`total >= 5000` antes de `total >= 1000`).
- Sin bloques markdown ```python dentro de esta sección.
[/CODIGO_CORREGIDO]
""".strip()

respuesta_rca = consultar_debugger(prompt_rca_y_refactor)

match_rca = re.search(r"\[CAUSA_RAIZ\](.*?)\[/CAUSA_RAIZ\]", respuesta_rca, re.DOTALL)
match_cod = re.search(r"\[CODIGO_CORREGIDO\](.*?)\[/CODIGO_CORREGIDO\]", respuesta_rca, re.DOTALL)

diagnostico_texto = match_rca.group(1).strip() if match_rca else respuesta_rca
codigo_limpio = match_cod.group(1).strip() if match_cod else ""
codigo_limpio = re.sub(r"^```(?:python)?|```$", "", codigo_limpio, flags=re.MULTILINE).strip()

print("=== 1. DIAGNÓSTICO DE CAUSA RAÍZ (RCA) ===")
print(diagnostico_texto)
print("\n=== 2. CÓDIGO REFACTORIZADO POR EL MODELO ===")
print(codigo_limpio)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Turno 2 ("Código como Conversación"): Generación y Ejecución de Tests Unitarios de Regresión
# MAGIC Ahora encadenamos la versión corregida de la función hacia un segundo turno para que el modelo escriba una función `ejecutar_suite_pruebas()` con sentencias `assert` que verifiquen que los 3 bugs quedaron erradicados.

# COMMAND ----------

prompt_generar_tests = f"""
### ROL
Actúa como un Ingeniero Senior de QA y Testing en Python.

### FUNCIÓN REFACTORIZADA A PROBAR
{codigo_limpio}

### TAREA
Escribe una función en Python llamada `ejecutar_suite_pruebas()` que ejecute 4 pruebas con `assert` sobre `liquidar_comision_vendedor`:
1. Caso aislamiento de estado: Llamar a un vendedor con ticket < 50 y luego a otro con ticket >= 50 verificando que la lista `alertas` del segundo esté vacía (`[]`).
2. Caso tramo VIP_ORO: Verificar que con una venta aprobada de 6000 el `nivel` sea `'VIP_ORO'` y `comision_usd` sea `720.0`.
3. Caso lista sin ventas aprobadas: Verificar que si todas las ventas están `'ANULADA'`, no lanza `ZeroDivisionError`, `total_vendido == 0` y `ticket_promedio == 0.0`.
4. Al final de `ejecutar_suite_pruebas()`, imprimir `"TODOS LOS TESTS DE REGRESIÓN PASARON EXITOSAMENTE"`.

RESTRICCIÓN: Devuelve únicamente código Python ejecutable, sin bloques markdown ```python ni texto adicional.
""".strip()

codigo_tests = consultar_debugger(prompt_generar_tests, max_tokens=600)
codigo_tests_limpio = re.sub(r"^```(?:python)?|```$", "", codigo_tests, flags=re.MULTILINE).strip()

print("=== SUITE DE TESTS GENERADA POR IA ===")
print(codigo_tests_limpio)

print("\n=== EJECUTANDO FUNCIÓN CORREGIDA + SUITE DE TESTS EN DATABRICKS ===")
if codigo_limpio and "def liquidar_comision_vendedor" in codigo_limpio:
    exec(codigo_limpio, globals())
if codigo_tests_limpio and "def ejecutar_suite_pruebas" in codigo_tests_limpio:
    exec(codigo_tests_limpio, globals())
    ejecutar_suite_pruebas()

# COMMAND ----------

# Salida esperada:
# 1. Evidencia inicial de los 3 bugs en VEND-01, VEND-02 y VEND-03.
# 2. Diagnóstico `[CAUSA_RAIZ]` explicando el argumento mutable `alertas=[]`, el orden invertido del `if/elif` y la división por `len(ventas_validas)` cuando es 0.
# 3. Código refactorizado tipado y ejecución exitosa de `ejecutar_suite_pruebas()` imprimiendo: "TODOS LOS TESTS DE REGRESIÓN PASARON EXITOSAMENTE".
