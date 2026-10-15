# Databricks notebook source
# MAGIC %md
# MAGIC # 📱 Hora 1 — Ejemplo 3: Explorar a Fondo el Dataset de DerarTel / dPhone
# MAGIC
# MAGIC - **Objetivo:** Materializar las tres tablas Delta del caso DerarTel (`smartphones`, `inventory_stock` y `customer_support_tickets`) a partir de los archivos CSV de `raw/`, explorar su contenido y esquema, e inspeccionar los metadatos binarios de los 8 documentos de `docs/`.
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Diferencia práctica entre datos estructurados (Tablas Delta para consultas exactas y herramientas de agentes) y datos no estructurados (documentos PDF/DOCX/PNG en Volúmenes para pipelines de RAG).
# MAGIC - **Requisitos previos:** Haber ejecutado `Includes/Lab_Setup.ipynb` o el Ejemplo 2 (`hora1_ejemplo2_revisar_lab_setup.py`). Incluye un mecanismo de contingencia (Plan B en memoria) por si el bucket S3 externo no estuviera accesible.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Conectar con el esquema `genai_course` y materializar las 3 tablas Delta
# MAGIC
# MAGIC En `Includes/Lab_Setup.ipynb`, los métodos `create_agent_tables()` y `create_batch_tables()` están definidos en la clase `CourseSetup`, pero no se llaman en la última celda de ese notebook. Aquí los ejecutamos explícitamente para crear:
# MAGIC 1. `smartphones` (desde `raw/smartphones.csv`)
# MAGIC 2. `inventory_stock` (desde `raw/inventory_stock.csv`)
# MAGIC 3. `customer_support_tickets` (desde `raw/support_tickets.csv`)
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `spark.sql(f"USE CATALOG {catalog_name}")` y `USE SCHEMA genai_course`: Asegura que trabajamos en el esquema del curso.
# MAGIC - `read_files('{dataset_volume}/raw/smartphones.csv', format => 'csv', schemaEvolutionMode => 'none')`: Función de tabla (TVF) nativa de Databricks SQL que infiere las columnas del CSV y lo convierte en una tabla Delta gobernada.
# MAGIC - Bloque `except Exception`: Si por contingencia de red la carpeta `raw/` no se hubiera descargado, crea exactamente las mismas 3 tablas Delta con los datos oficiales de DerarTel para que nadie se quede atrás.

# COMMAND ----------

# Aseguramos que estamos posicionados en nuestro catálogo y en el esquema genai_course
catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = "genai_course"
dataset_volume = f"/Volumes/{catalog_name}/{schema_name}/dataset"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

try:
    # Creamos las 3 tablas Delta leyendo directamente los CSV del Volumen en Unity Catalog
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS smartphones
        AS SELECT * FROM read_files(
            '{dataset_volume}/raw/smartphones.csv',
            format => 'csv',
            schemaEvolutionMode => 'none'
        )
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS inventory_stock
        AS SELECT * FROM read_files(
            '{dataset_volume}/raw/inventory_stock.csv',
            format => 'csv',
            schemaEvolutionMode => 'none'
        )
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS customer_support_tickets
        AS SELECT * FROM read_files(
            '{dataset_volume}/raw/support_tickets.csv',
            format => 'csv',
            schemaEvolutionMode => 'none'
        )
    """)
    print("✅ Las 3 tablas Delta se crearon/verificaron exitosamente desde los archivos CSV del Volumen.")
except Exception as e:
    print(f"⚠️ Activando Plan B en memoria para crear las tablas oficiales de DerarTel ({e})...")
    spark.sql("""
        CREATE OR REPLACE TABLE smartphones AS
        SELECT * FROM VALUES
            (1, 'dPhone D1',      'Silver', 800,  DATE'2026-01-30'),
            (2, 'dPhone D1 Lite', 'Blue',   650,  DATE'2026-01-30'),
            (3, 'dPhone D1 Pro',  'Green',  1000, DATE'2026-02-15'),
            (4, 'dPhone D1 Fold', 'Silver', 1200, DATE'2026-03-20')
        AS t(phone_id, model, color, price, release_date)
    """)
    spark.sql("""
        CREATE OR REPLACE TABLE inventory_stock AS
        SELECT * FROM VALUES
            (1, 8,  DATE'2026-03-22'),
            (2, 5,  DATE'2026-03-22'),
            (3, 15, DATE'2026-03-23'),
            (4, 10, DATE'2026-03-25')
        AS t(item_id, quantity, update_date)
    """)
    spark.sql("""
        CREATE OR REPLACE TABLE customer_support_tickets AS
        SELECT * FROM VALUES
            (1, 'john.doe@example.com',    'Question about dPhone D1 Fold', 'Hi, I would like to get more information about the new dPhone D1 Fold. Can you share the specifications?', 'open', 'medium', '2026-03-29T01:01:29.026Z', CAST(NULL AS STRING), CAST(NULL AS STRING), CAST(NULL AS STRING)),
            (2, 'sarah.smith@example.com', 'Purchase latest phone',         'Hello, I am interested in buying 3 of latest dPhone model. Please let me know how I can proceed with the purchase.', 'open', 'medium', '2026-03-29T01:01:29.026Z', CAST(NULL AS STRING), CAST(NULL AS STRING), CAST(NULL AS STRING))
        AS t(ticket_id, customer_email, subject, description, status, priority, creation_time, reply_message, reply_agent, reply_time)
    """)
    print("✅ Plan B completado: las 3 tablas Delta están listas en genai_course.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploración de las tablas estructuradas: `smartphones` + `inventory_stock`
# MAGIC
# MAGIC Cruzamos (`INNER JOIN`) el catálogo de teléfonos con el inventario de bodega para responder con datos reales la primera mitad de la pregunta de la Clase 1: *¿cuál es el dPhone más reciente y cuánto stock hay?*
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `SELECT s.phone_id, s.model, ...`: Selecciona los atributos del teléfono y la cantidad en inventario.
# MAGIC - `(s.price * i.quantity) AS valor_inventario_usd`: Calcula el valor financiero del stock disponible por cada modelo.
# MAGIC - `INNER JOIN inventory_stock i ON s.phone_id = i.item_id`: Une ambas tablas por el identificador único del teléfono.
# MAGIC - `ORDER BY s.release_date DESC`: Ordena del lanzamiento más reciente (`2026-03-20`, `dPhone D1 Fold`) al más antiguo.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Cruzamos smartphones con inventory_stock para ver el catálogo completo y su stock actual
# MAGIC SELECT 
# MAGIC     s.phone_id,
# MAGIC     s.model,
# MAGIC     s.color,
# MAGIC     s.price                       AS precio_unitario_usd,
# MAGIC     s.release_date                AS fecha_lanzamiento,
# MAGIC     i.quantity                    AS unidades_en_stock,
# MAGIC     i.update_date                 AS ultima_actualizacion_stock,
# MAGIC     (s.price * i.quantity)        AS valor_inventario_usd
# MAGIC FROM smartphones s
# MAGIC INNER JOIN inventory_stock i
# MAGIC     ON s.phone_id = i.item_id
# MAGIC ORDER BY s.release_date DESC, s.price DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploración de los tickets de soporte: `customer_support_tickets`
# MAGIC
# MAGIC Revisamos qué están preguntando los clientes reales de DerarTel en este momento.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Consultamos las columnas principales de `customer_support_tickets` para observar que **Ticket 1** pide especificaciones técnicas del `dPhone D1 Fold` (requiere buscar en los PDFs de `docs/`) y **Ticket 2** pide comprar 3 unidades del modelo más reciente (requiere consultar `smartphones` e `inventory_stock`).

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Inspeccionamos los tickets de soporte abiertos por clientes de DerarTel
# MAGIC SELECT 
# MAGIC     ticket_id,
# MAGIC     customer_email,
# MAGIC     subject,
# MAGIC     description,
# MAGIC     status,
# MAGIC     priority,
# MAGIC     creation_time
# MAGIC FROM customer_support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploración de los 8 documentos no estructurados en `docs/` con PySpark (`binaryFile`)
# MAGIC
# MAGIC Así como en el notebook `1.1A- Data Prep (Bronze)` de la Semana 5 leeremos los documentos usando el formato `binaryfile` de Spark, aquí inspeccionamos sus metadatos sin modificarlos.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `spark.read.format("binaryFile").load(f"{dataset_volume}/docs")`: Lee cada archivo de la carpeta `docs/` como un registro con `path`, `modificationTime`, `length` (tamaño en bytes) y `content` (bytes crudos del archivo).
# MAGIC - `.select("path", "modificationTime", "length")`: Omitimos la columna `content` (binaria) en la vista rápida para obtener un catálogo limpio de documentos ordenado por nombre.

# COMMAND ----------

from pyspark.sql import functions as F

ruta_docs = f"{dataset_volume}/docs"

try:
    docs_meta_df = (
        spark.read.format("binaryFile")
        .load(ruta_docs)
        .withColumn("nombre_archivo", F.element_at(F.split(F.col("path"), "/"), -1))
        .withColumn("extension", F.upper(F.element_at(F.split(F.col("path"), "\\."), -1)))
        .withColumn("tamano_kb", F.round(F.col("length") / 1024.0, 2))
        .select("nombre_archivo", "extension", "tamano_kb", "modificationTime", "path")
        .orderBy("nombre_archivo")
    )
    print(f"Total de documentos encontrados en {ruta_docs}: {docs_meta_df.count()}")
    display(docs_meta_df)
except Exception as e:
    print(f"No se pudo leer la carpeta {ruta_docs}: {e}")

# Salida esperada:
# ✅ Las 3 tablas Delta se crearon/verificaron exitosamente desde los archivos CSV del Volumen.
# Resultado del JOIN (4 filas):
#   phone_id=4 | dPhone D1 Fold | Silver | $1200 | 2026-03-20 | stock=10 | valor=$12,000
#   phone_id=3 | dPhone D1 Pro  | Green  | $1000 | 2026-02-15 | stock=15 | valor=$15,000
#   phone_id=1 | dPhone D1      | Silver | $800  | 2026-01-30 | stock=8  | valor=$6,400
#   phone_id=2 | dPhone D1 Lite | Blue   | $650  | 2026-01-30 | stock=5  | valor=$3,250
# Total de documentos encontrados en /Volumes/<catalogo>/genai_course/dataset/docs: 8
# (4 archivos PDF de especificaciones, 1 PDF de perfil corporativo, 1 DOCX de privacidad, 1 PNG de diagrama técnico y 1 PDF de manual de usuario).
