# Databricks notebook source
# MAGIC %md
# MAGIC # 🛡️ Hora 3 — Ejemplo 2: Alucinación en Frío vs Grounding con Datos Reales del dPhone
# MAGIC
# MAGIC - **Objetivo:** Demostrar en vivo por qué un LLM alucina o falla al responder la pregunta conductora de la Clase 1 sobre el **dPhone** cuando se invoca sin contexto, y cómo se elimina la alucinación aplicando **Grounding (anclaje en datos)** al inyectar las filas reales de `smartphones`, `inventory_stock` y las políticas oficiales de garantía de DerarTel.
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Naturaleza probabilística de las alucinaciones, fecha de corte de conocimiento, ausencia de acceso a datos privados y patrón de **Grounding en Ventana de Contexto** (precursor directo de RAG y Agentes).
# MAGIC - **Requisitos previos:** Tener las tablas `smartphones` e `inventory_stock` creadas en `genai_course` y el endpoint `databricks-gpt-oss-20b` activo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preparación de la pregunta del cliente (la misma de la Clase 1)
# MAGIC
# MAGIC Recordemos la pregunta exacta del cliente de DerarTel con la que abrimos la Clase 1:
# MAGIC > *"¿Cuál es el dPhone más reciente de DerarTel, cuánto cuesta, cuántas unidades hay en stock (y si saben cuántas hay en Medellín) y qué cubre la garantía si se moja?"*

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import pandas as pd

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()
MODELO = "databricks-gpt-oss-20b"

def extraer_texto(mensaje) -> str:
    c = mensaje.content
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return "\n".join(
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for b in c
            if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "text"
        ).strip()
    return str(c)

pregunta_cliente = (
    "Hola DerarTel, quiero saber: "
    "1) ¿Cuál es el modelo de dPhone más reciente y cuál es su precio oficial en dólares? "
    "2) ¿Cuántas unidades hay hoy en stock en total y cuántas en la tienda de Medellín? "
    "3) ¿Qué pasa con la garantía si el teléfono sufre daño por inmersión en agua?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experimento 1: Llamada en frío (SIN Grounding)
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Enviamos la pregunta directamente al modelo con `temperature=0.0`, pero **sin adjuntar ninguna tabla ni manual**.
# MAGIC - Incluso con temperatura `0.0`, el modelo no tiene forma de conocer las tablas internas de DerarTel ni el lanzamiento del `dPhone D1 Fold` del 20 de marzo de 2026.

# COMMAND ----------

prompt_sin_grounding = (
    "Eres el asistente virtual de la empresa DerarTel, fabricante del smartphone dPhone. "
    "Responde de forma amable y detallada a las 3 preguntas del cliente."
)

resp_frio = client.chat.completions.create(
    model=MODELO,
    messages=[
        {"role": "system", "content": prompt_sin_grounding},
        {"role": "user", "content": pregunta_cliente}
    ],
    temperature=0.0,
    max_tokens=450
)

texto_sin_grounding = extraer_texto(resp_frio.choices[0].message)
print("❌ === RESPUESTA DEL MODELO SIN GROUNDING (EN FRÍO) ===")
print(texto_sin_grounding)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Construcción dinámica del contexto verificable desde Unity Catalog
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Consultamos en vivo con `spark.sql(...)` el cruce real entre `smartphones` e `inventory_stock` dentro del esquema `genai_course`.
# MAGIC - Formateamos las filas reales como una tabla Markdown legible por el LLM e incluimos el extracto normativo del manual de usuario (`08_User_Manual.pdf`).

# COMMAND ----------

catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("USE SCHEMA genai_course")

# Consultamos los datos reales y actualizados desde nuestras tablas Delta en Unity Catalog
df_catalogo_stock = spark.sql("""
    SELECT 
        s.phone_id,
        s.model,
        s.color,
        s.price AS price_usd,
        CAST(s.release_date AS STRING) AS release_date,
        i.quantity AS global_stock_units,
        CAST(i.update_date AS STRING) AS stock_update_date
    FROM smartphones s
    INNER JOIN inventory_stock i ON s.phone_id = i.item_id
    ORDER BY s.release_date DESC
""").toPandas()

tabla_contexto_md = df_catalogo_stock.to_markdown(index=False)

extracto_manual_garantia = """
EXTRACTO OFICIAL - MANUAL DE USUARIO Y GARANTÍA DERARTEL (08_User_Manual.pdf):
- Cobertura estándar: 12 meses por defectos de fabricación en hardware, pantalla y batería bajo uso normal.
- Exclusión por líquidos: Los daños ocasionados por inmersión en agua, contacto con líquidos o corrosión interna activan los indicadores internos de contacto líquido (LCI) y NO están cubiertos por la garantía limitada estándar, salvo que el cliente haya adquirido el plan adicional DerarCare+ con cobertura de daño accidental.
- Nota de inventario en base de datos: La tabla `inventory_stock` registra únicamente el stock global consolidado por `item_id`; no desglosa inventario por ciudad ni por tienda física (como Medellín o Bogotá).
"""

print("📋 === CONTEXTO VERIFICABLE EXTRAÍDO DE UNITY CATALOG ===")
print(tabla_contexto_md)
print(extracto_manual_garantia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Experimento 2: Llamada CON Grounding y Regla Estricta de Abstención
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Inyectamos `tabla_contexto_md` y `extracto_manual_garantia` dentro del prompt de sistema.
# MAGIC - Agregamos una **regla de abstención explícita**: prohibido inventar precios, stock por ciudad o coberturas que no estén en el bloque `<contexto_oficial>`.

# COMMAND ----------

prompt_con_grounding = f"""
Eres el asistente oficial de DerarTel.
Debes responder al cliente basándote EXCLUSIVAMENTE en la información contenida dentro de las etiquetas <contexto_oficial>.
REGLAS ESTRICTAS ANTI-ALUCINACIÓN:
1. No inventes ni supongas ningún dato que no aparezca explícitamente en <contexto_oficial>.
2. Si el cliente pregunta por un desglose que la tabla no tiene (por ejemplo, stock específico en una ciudad como Medellín), informa el stock global exacto y aclara con transparencia que el sistema solo registra inventario global consolidado y no por ciudad.
3. Cita fechas de lanzamiento, precios y unidades exactas según la tabla.

<contexto_oficial>
TABLA OFICIAL DE PRODUCTOS Y STOCK GLOBAL (Unity Catalog: genai_course.smartphones + inventory_stock):
{tabla_contexto_md}

{extracto_manual_garantia}
</contexto_oficial>
"""

resp_grounding = client.chat.completions.create(
    model=MODELO,
    messages=[
        {"role": "system", "content": prompt_con_grounding},
        {"role": "user", "content": pregunta_cliente}
    ],
    temperature=0.0,
    max_tokens=450
)

texto_con_grounding = extraer_texto(resp_grounding.choices[0].message)
print("✅ === RESPUESTA DEL MODELO CON GROUNDING (DATOS DE UNITY CATALOG) ===")
print(texto_con_grounding)

# Salida esperada:
# 1. SIN Grounding: El modelo o bien declara que no tiene acceso al catálogo en tiempo real de DerarTel o inventa precios/modelos inexistentes.
# 2. CON Grounding:
#    - Identifica con precisión total que el modelo más reciente es el **dPhone D1 Fold** (phone_id=4, color Silver), lanzado el **2026-03-20**, con precio oficial de **$1,200 USD**.
#    - Informa que hay exactamente **10 unidades** en stock global (actualizado al 2026-03-25) y aclara honestamente que la base de datos `inventory_stock` no desglosa stock por ciudad (Medellín).
#    - Explica que el daño por inmersión en agua activa los sensores LCI y NO está cubierto por la garantía estándar de 12 meses (salvo plan adicional DerarCare+).
