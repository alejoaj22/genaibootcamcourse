# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 1 — Crear y auditar funciones SQL en Unity Catalog
# MAGIC **Objetivo:** crear (de forma idempotente) las tres funciones SQL del notebook 2.3 y una cuarta propia (`get_phones_by_color`), inspeccionar sus metadatos y probar casos límite.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** una UC Function SQL es una herramienta: el nombre, el `COMMENT` y los parámetros son lo único que "ve" el LLM.
# MAGIC **Requisitos previos:** permiso `CREATE FUNCTION` en el esquema `genai_course`; tablas `smartphones` e `inventory_stock` (`Includes/Lab_Setup_02`).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade unitycatalog-ai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup_02

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 1) ¿En qué catálogo y esquema se crearán las funciones? (Lab_Setup ya ejecutó USE CATALOG / USE SCHEMA)
# MAGIC SELECT current_catalog() AS catalogo, current_schema() AS esquema

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2) Función 1 (idéntica a la del notebook 2.3): sin parámetros. El COMMENT advierte "ZERO parameters".
# MAGIC CREATE OR REPLACE FUNCTION get_latest_phone()
# MAGIC RETURNS TABLE(phone_id INT, model STRING, color STRING, price INT)
# MAGIC COMMENT 'Returns the latest phone, ordered by release_date descending. IMPORTANT: This function accepts ZERO parameters. Do not pass any arguments.'
# MAGIC RETURN
# MAGIC   SELECT phone_id, model, color, price
# MAGIC   FROM smartphones
# MAGIC   ORDER BY release_date DESC
# MAGIC   LIMIT 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3) Función 2 (idéntica a la del notebook 2.3): un parámetro con su propio COMMENT.
# MAGIC CREATE OR REPLACE FUNCTION check_inventory(phone_id INT COMMENT 'ID of the phone to check (integer)')
# MAGIC RETURNS TABLE (
# MAGIC   item_id INT,
# MAGIC   quantity INT
# MAGIC )
# MAGIC COMMENT 'Checks the stock quantity for a given phone in the inventory.'
# MAGIC RETURN
# MAGIC   SELECT item_id, quantity
# MAGIC   FROM inventory_stock
# MAGIC   WHERE item_id = phone_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 4) Función 3 (idéntica a la del notebook 2.3): identidad del usuario de la sesión.
# MAGIC CREATE OR REPLACE FUNCTION get_current_user()
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the current session user, used for identifying the active user. IMPORTANT: This function accepts ZERO parameters. Do not pass any arguments.'
# MAGIC RETURN current_user();

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 5) Pruebas rápidas de las tres funciones.
# MAGIC SELECT * FROM get_latest_phone();

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM check_inventory(4);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT get_current_user() AS usuario;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 6) ¿Qué colores existen? (para diseñar la nueva función y sus pruebas)
# MAGIC SELECT DISTINCT color FROM smartphones ORDER BY color;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 7) Función 4 (propia): un parámetro de texto. Buenas prácticas aplicadas:
# MAGIC --    - Parámetro con nombre distinto al de la columna (color_name vs color) para evitar ambigüedad.
# MAGIC --    - COMMENT que dice QUÉ hace, CUÁNDO usarla y QUÉ devuelve.
# MAGIC --    - Comparación sin distinguir mayúsculas (los LLM no siempre respetan el formato).
# MAGIC CREATE OR REPLACE FUNCTION get_phones_by_color(
# MAGIC   color_name STRING COMMENT 'Phone color in English, for example Black, Blue or Silver. Not case sensitive.'
# MAGIC )
# MAGIC RETURNS TABLE(phone_id INT, model STRING, color STRING, price INT)
# MAGIC COMMENT 'Lists the smartphones available in a given color, ordered by price ascending. Use it when the customer asks for a specific color. Returns zero rows if the color does not exist.'
# MAGIC RETURN
# MAGIC   SELECT phone_id, model, color, price
# MAGIC   FROM smartphones
# MAGIC   WHERE lower(color) = lower(color_name)
# MAGIC   ORDER BY price ASC;

# COMMAND ----------

# 8) Pruebas con casos límite: color existente, el mismo en MAYÚSCULAS, y un color que no existe.
primer_color = spark.sql("SELECT color FROM smartphones WHERE color IS NOT NULL ORDER BY color LIMIT 1").first()["color"]
casos = [primer_color, primer_color.upper(), "Turquesa-inexistente"]
for c in casos:
    n = spark.sql(f"SELECT * FROM get_phones_by_color('{c}')").count()
    print(f"get_phones_by_color('{c}') → {n} fila(s)")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 9) Lo que "ve" el LLM: descripción completa de la función (firma, parámetros, comentario, cuerpo).
# MAGIC DESCRIBE FUNCTION EXTENDED get_phones_by_color

# COMMAND ----------

# 10) Auditoría rápida de metadatos con information_schema: comentarios de todas las funciones del esquema.
cat, sch = course.catalog, course.schema
display(spark.sql(f"""
    SELECT routine_name, routine_type, data_type AS tipo_retorno,
           comment IS NOT NULL AND length(comment) > 20 AS tiene_comentario_util,
           left(comment, 90) AS comentario
    FROM {cat}.information_schema.routines
    WHERE routine_schema = '{sch}'
    ORDER BY routine_name
"""))

# COMMAND ----------

# 11) Parámetros descritos: ¿cada parámetro tiene COMMENT? (un parámetro sin descripción = el LLM adivina).
display(spark.sql(f"""
    SELECT specific_name AS funcion, parameter_name, data_type, comment IS NOT NULL AS tiene_comentario
    FROM {cat}.information_schema.parameters
    WHERE specific_schema = '{sch}'
    ORDER BY specific_name, ordinal_position
"""))

# COMMAND ----------

# Salida esperada:
# - Celda 1: catálogo (p. ej. workspace) y esquema genai_course.
# - get_latest_phone() devuelve 1 fila (phone_id, model, color, price); check_inventory(4) devuelve (item_id=4, quantity=<n>); get_current_user() devuelve tu correo.
# - Celda de colores: lista de colores distintos de smartphones.
# - Pruebas de get_phones_by_color: ≥1 fila para el color real, el mismo número de filas con MAYÚSCULAS y 0 filas para "Turquesa-inexistente".
# - DESCRIBE FUNCTION EXTENDED muestra la firma, el comentario y el cuerpo de get_phones_by_color.
# - information_schema: las 4 funciones SQL con comentario útil; get_phones_by_color y check_inventory con parámetros descritos (tiene_comentario = true).
