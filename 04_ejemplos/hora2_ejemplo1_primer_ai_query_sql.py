# Databricks notebook source
# MAGIC %md
# MAGIC # ⚡ Hora 2 — Ejemplo 1: Nuestro Primer `ai_query()` en SQL sobre Datos de DerarTel
# MAGIC
# MAGIC - **Objetivo:** Ejecutar nuestras primeras inferencias generativas directamente desde Databricks SQL utilizando la función nativa `ai_query()` con el endpoint `databricks-gpt-oss-20b`, primero en una consulta simple y luego fila por fila sobre las tablas `smartphones` y `customer_support_tickets`.
# MAGIC - **Tiempo estimado:** 8 minutos.
# MAGIC - **Concepto que ilustra:** Herramientas de IA nativas para datos (Databricks SQL AI Functions) y procesamiento generativo por lotes (*Batch LLM Inference*) dentro del Lakehouse.
# MAGIC - **Requisitos previos:** Haber ejecutado los notebooks de la Hora 1 (tener creadas las tablas `smartphones` y `customer_support_tickets` en el esquema `genai_course`) y contar con el endpoint `databricks-gpt-oss-20b` activo en Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Posicionarnos en el esquema `genai_course`
# MAGIC
# MAGIC Antes de lanzar consultas SQL sobre nuestras tablas, activamos el esquema `genai_course` dentro de nuestro catálogo actual.

# COMMAND ----------

# Activamos el catálogo actual y el esquema genai_course
catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql("USE SCHEMA genai_course")
print(f"Contexto listo: {catalog_name}.genai_course")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Llamada escalar mínima con `ai_query()`
# MAGIC
# MAGIC La firma básica de la función es:
# MAGIC ```sql
# MAGIC ai_query(endpoint_name, request)
# MAGIC ```
# MAGIC Donde `endpoint_name` es el nombre del modelo servido en **Databricks Model Serving** (por ejemplo `'databricks-gpt-oss-20b'`) y `request` es el texto del prompt.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `SELECT ai_query('databricks-gpt-oss-20b', '...') AS respuesta_llm;`: Envía un único prompt al modelo desde SQL sin necesidad de consultar ninguna tabla todavía.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primera llamada "Hola Mundo" a un LLM directamente desde una sentencia SELECT de SQL
# MAGIC SELECT 
# MAGIC     'databricks-gpt-oss-20b' AS modelo_utilizado,
# MAGIC     ai_query(
# MAGIC         'databricks-gpt-oss-20b',
# MAGIC         'Explica en español y en una sola frase de máximo 25 palabras por qué es útil poder llamar a un LLM dentro de una consulta SQL.'
# MAGIC     ) AS respuesta_llm;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inferencia fila por fila sobre la tabla `smartphones`
# MAGIC
# MAGIC Ahora combinamos columnas estructuradas (`model`, `color`, `price`, `release_date`) con la función `CONCAT()` para que el modelo redacte un mensaje comercial en español para cada uno de los 4 teléfonos de DerarTel.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `CONCAT('Actúa como asesor de DerarTel...', model, ...)`: Construye dinámicamente un prompt distinto para cada fila de la tabla `smartphones` inyectando sus valores reales.
# MAGIC - `LIMIT 4`: Buena práctica obligatoria en consultas con `ai_query()` para acotar exactamente cuántas llamadas al LLM se realizarán.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Generamos un eslogan comercial en español para cada teléfono de la tabla smartphones
# MAGIC SELECT 
# MAGIC     phone_id,
# MAGIC     model,
# MAGIC     color,
# MAGIC     price,
# MAGIC     release_date,
# MAGIC     ai_query(
# MAGIC         'databricks-gpt-oss-20b',
# MAGIC         CONCAT(
# MAGIC             'Actúa como redactor comercial de DerarTel. Escribe en español un eslogan atractivo de máximo 20 palabras ',
# MAGIC             'para promocionar el smartphone "', model, '" en color ', color, 
# MAGIC             ', lanzado el ', CAST(release_date AS STRING), 
# MAGIC             ' con un precio oficial de $', CAST(price AS STRING), ' USD. No inventes especificaciones técnicas.'
# MAGIC         )
# MAGIC     ) AS eslogan_comercial_es
# MAGIC FROM smartphones
# MAGIC ORDER BY phone_id
# MAGIC LIMIT 4;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Triaje automático y traducción de tickets en `customer_support_tickets`
# MAGIC
# MAGIC Los tickets originales de `support_tickets.csv` vienen en inglés y en texto libre. Con una sola consulta SQL podemos traducirlos al español, clasificar su intención (`VENTA` vs `SOPORTE_TECNICO`) y extraer qué producto mencionan.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Pedimos al modelo que devuelva un objeto JSON compacto con las claves `intencion`, `modelo_mencionado`, `cantidad_solicitada` y `resumen_es`.
# MAGIC - Esto demuestra cómo un perfil de datos puede transformar texto desestructurado en columnas analíticas sin salir de SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analizamos y estructuramos los tickets de soporte usando ai_query()
# MAGIC SELECT 
# MAGIC     ticket_id,
# MAGIC     customer_email,
# MAGIC     subject,
# MAGIC     description,
# MAGIC     ai_query(
# MAGIC         'databricks-gpt-oss-20b',
# MAGIC         CONCAT(
# MAGIC             'Analiza el siguiente ticket de soporte de DerarTel y responde ÚNICAMENTE con un objeto JSON válido ',
# MAGIC             'con las claves: "intencion" ("COMPRA" o "INFORMACION_TECNICA"), ',
# MAGIC             '"modelo_mencionado" (nombre del teléfono o "ULTIMO_MODELO"), ',
# MAGIC             '"cantidad" (número entero, 1 por defecto si no se especifica) y ',
# MAGIC             '"resumen_espanol" (resumen en español en máximo 15 palabras). ',
# MAGIC             'Asunto: ', subject, ' | Descripción: ', description
# MAGIC         )
# MAGIC     ) AS triaje_estructurado_json
# MAGIC FROM customer_support_tickets
# MAGIC ORDER BY ticket_id
# MAGIC LIMIT 2;

# COMMAND ----------

# Salida esperada:
# 1. En la consulta escalar: una fila con la explicación en español de max 25 palabras.
# 2. En smartphones (4 filas): cada modelo (dPhone D1, dPhone D1 Lite, dPhone D1 Pro, dPhone D1 Fold) acompañado de su eslogan en español respetando el color y precio exactos ($800, $650, $1000, $1200).
# 3. En customer_support_tickets (2 filas):
#    - Ticket 1 (john.doe@example.com): {"intencion": "INFORMACION_TECNICA", "modelo_mencionado": "dPhone D1 Fold", "cantidad": 1, "resumen_espanol": "Solicita información y especificaciones técnicas del nuevo dPhone D1 Fold."}
#    - Ticket 2 (sarah.smith@example.com): {"intencion": "COMPRA", "modelo_mencionado": "ULTIMO_MODELO", "cantidad": 3, "resumen_espanol": "Interesada en comprar 3 unidades del modelo de dPhone más reciente."}
