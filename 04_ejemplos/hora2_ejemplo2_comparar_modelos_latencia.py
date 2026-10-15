# Databricks notebook source
# MAGIC %md
# MAGIC # ⚖️ Hora 2 — Ejemplo 2: Comparación de Modelos (`databricks-gpt-oss-20b` vs `databricks-gpt-oss-120b`) y Latencia en Python
# MAGIC
# MAGIC - **Objetivo:** Comparar programáticamente desde Python dos modelos de diferente escala y capacidad servidos en Databricks Foundation Model APIs (`databricks-gpt-oss-20b` vs `databricks-gpt-oss-120b`), midiendo su latencia en segundos, consumo de tokens e inspeccionando los bloques de razonamiento (`reasoning`) frente al texto final (`text`).
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Diferencias entre modelos rápidos de instrucciones (20B) y modelos de razonamiento profundo (120B); trade-off entre calidad, latencia y costo; uso del cliente OpenAI nativo de Databricks (`WorkspaceClient().serving_endpoints.get_open_ai_client()`).
# MAGIC - **Requisitos previos:** Cómputo conectado en Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Instalación/actualización de `databricks-sdk` y `openai`
# MAGIC
# MAGIC En Databricks Runtime ML las librerías suelen venir preinstaladas, pero aseguramos que tanto `databricks-sdk` como `openai` estén listos para usar `w.serving_endpoints.get_open_ai_client()`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inicialización del cliente OpenAI autenticado con Databricks y parser de respuesta
# MAGIC
# MAGIC ### Explicación técnica crítica (¡Atención Desarrolladores y Data Engineers!):
# MAGIC Cuando consultas modelos de razonamiento como `databricks-gpt-oss-120b` (o `20b` con razonamiento activo) a través del endpoint compatible con OpenAI en Databricks, el campo `response.choices[0].message.content` **puede no ser un simple `str`**, sino una **lista de bloques tipados**:
# MAGIC 1. Un bloque `{"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]}` con la cadena de pensamiento interna del modelo.
# MAGIC 2. Un bloque `{"type": "text", "text": "..."}` con la respuesta final al usuario.
# MAGIC
# MAGIC Nuestra función `extraer_razonamiento_y_respuesta(message)` maneja ambos formatos (string clásico y lista de bloques) de forma transparente y segura.

# COMMAND ----------

import time
import pandas as pd
from databricks.sdk import WorkspaceClient

# 1. Inicializamos el cliente del Workspace de Databricks (usa la autenticación automática del notebook)
w = WorkspaceClient()

# 2. Obtenemos un cliente estándar de OpenAI que apunta directamente a Databricks Model Serving
client = w.serving_endpoints.get_open_ai_client()

MODELO_RAPIDO = "databricks-gpt-oss-20b"
MODELO_GRANDE = "databricks-gpt-oss-120b"

def extraer_razonamiento_y_respuesta(message) -> tuple[str, str]:
    """
    Extrae de forma segura tanto el pensamiento interno ('reasoning') como el texto final ('text')
    de un ChatCompletionMessage devuelto por Databricks Foundation Model APIs.
    """
    contenido = message.content
    # Caso A: El modelo devolvió un string simple tradicional
    if isinstance(contenido, str):
        razonamiento = getattr(message, "reasoning_content", "") or ""
        return razonamiento.strip(), contenido.strip()
    
    # Caso B: El modelo (ej. gpt-oss-120b) devolvió una lista de bloques [{'type': 'reasoning', ...}, {'type': 'text', ...}]
    partes_razonamiento = []
    partes_texto = []
    if isinstance(contenido, list):
        for bloque in contenido:
            b_type = bloque.get("type") if isinstance(bloque, dict) else getattr(bloque, "type", None)
            if b_type == "reasoning":
                resumen = bloque.get("summary", []) if isinstance(bloque, dict) else getattr(bloque, "summary", [])
                for item in resumen:
                    txt = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    if txt:
                        partes_razonamiento.append(txt)
            elif b_type == "text":
                txt = bloque.get("text", "") if isinstance(bloque, dict) else getattr(bloque, "text", "")
                if txt:
                    partes_texto.append(txt)
    return "\n".join(partes_razonamiento).strip(), "\n".join(partes_texto).strip()

print(f"✅ Cliente OpenAI de Databricks configurado. Compararemos '{MODELO_RAPIDO}' vs '{MODELO_GRANDE}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Función de benchmarking con medición de latencia y tokens
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `t_inicio = time.perf_counter()`: Reloj de alta precisión en Python para medir segundos transcurridos.
# MAGIC - `client.chat.completions.create(...)`: Llama al endpoint pasando el `model`, los `messages` (`system` y `user`) y `temperature=0.1`.
# MAGIC - `uso = response.usage`: Extrae `prompt_tokens`, `completion_tokens` y `total_tokens` reportados por el servidor.

# COMMAND ----------

def consultar_y_medir(modelo: str, prompt_sistema: str, prompt_usuario: str, max_tokens: int = 700) -> dict:
    """Envía el mismo prompt a un endpoint de Databricks y registra latencia, tokens, razonamiento y respuesta."""
    t_inicio = time.perf_counter()
    
    response = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )
    
    latencia_seg = time.perf_counter() - t_inicio
    mensaje = response.choices[0].message
    razonamiento, texto_final = extraer_razonamiento_y_respuesta(mensaje)
    
    uso = response.usage
    prompt_toks = uso.prompt_tokens if uso else 0
    comp_toks = uso.completion_tokens if uso else 0
    total_toks = uso.total_tokens if uso else 0
    
    return {
        "modelo": modelo,
        "latencia_seg": round(latencia_seg, 3),
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "total_tokens": total_toks,
        "tokens_por_seg": round(comp_toks / latencia_seg, 1) if latencia_seg > 0 else 0.0,
        "caracteres_razonamiento": len(razonamiento),
        "razonamiento": razonamiento,
        "respuesta_final": texto_final
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ejecución de la prueba comparativa: Lógica de Negocio + SQL de DerarTel
# MAGIC
# MAGIC Planteamos un problema con una **trampa realista de datos**: el cliente pregunta si puede comprar 3 unidades del `dPhone D1 Fold` en la tienda de **Medellín**, pero le entregamos el esquema exacto de nuestras tablas (`smartphones` e `inventory_stock`), donde **NO existe columna de ciudad**. Veamos cómo responde y cuánto tarda cada modelo.

# COMMAND ----------

prompt_sistema = (
    "Eres un Arquitecto Senior de Datos y Backend en DerarTel. "
    "Responde en español de forma rigurosa y concisa."
)

prompt_usuario = """
Tenemos únicamente estas dos tablas en Unity Catalog (esquema genai_course):
1. Tabla `smartphones` (phone_id INT, model STRING, color STRING, price INT, release_date DATE)
2. Tabla `inventory_stock` (item_id INT, quantity INT, update_date DATE)

Un cliente escribe: "Quiero comprar 3 unidades del dPhone más reciente pero necesito recogerlas hoy mismo en la tienda de Medellín".

Responde en 3 puntos breves:
1) Escribe la consulta SQL exacta para saber cuál es el modelo más reciente y si el stock global alcanza para 3 unidades.
2) ¿Qué limitación tiene nuestro esquema actual frente a la pregunta sobre Medellín y cómo no debemos engañar al cliente?
3) ¿Qué columna o tabla propondrías agregar al modelo de datos para soportar inventario por sucursal?
"""

resultados = []
for mod in [MODELO_RAPIDO, MODELO_GRANDE]:
    print(f"Consultando {mod} ...")
    res = consultar_y_medir(mod, prompt_sistema, prompt_usuario)
    resultados.append(res)
    print(f"  ⏱️ Latencia: {res['latencia_seg']} s | Tokens totales: {res['total_tokens']} ({res['tokens_por_seg']} tok/s)\n")

# Mostramos la tabla comparativa de métricas
df_metricas = pd.DataFrame(resultados)[[
    "modelo", "latencia_seg", "prompt_tokens", "completion_tokens", 
    "total_tokens", "tokens_por_seg", "caracteres_razonamiento"
]]
display(df_metricas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspección detallada: Cadena de Pensamiento (`reasoning`) vs Respuesta Final (`text`)

# COMMAND ----------

for r in resultados:
    print("=" * 95)
    print(f"🤖 MODELO: {r['modelo']}  (Latencia: {r['latencia_seg']} s | Tokens salida: {r['completion_tokens']})")
    print("=" * 95)
    if r["razonamiento"]:
        print("🧠 [CADENA DE RAZONAMIENTO INTERNA (REASONING)]:")
        print(r["razonamiento"][:700] + ("..." if len(r["razonamiento"]) > 700 else ""))
        print("-" * 95)
    print("💬 [RESPUESTA FINAL AL USUARIO (TEXT)]:")
    print(r["respuesta_final"])
    print("\n")

# Salida esperada:
# - Tabla comparativa mostrando a databricks-gpt-oss-20b y databricks-gpt-oss-120b con sus latencias en segundos, prompt_tokens (~190), completion_tokens y velocidad en tokens/seg.
# - Ambos modelos generan el SQL cruzando smartphones (s.phone_id = i.item_id) ordenando por release_date DESC LIMIT 1 y verificando quantity >= 3.
# - Ambos detectan que inventory_stock NO tiene columna de ciudad/sucursal (Medellín) y proponen agregar una tabla de sucursales/bodegas (store_id, city) para evitar alucinar disponibilidad local.
