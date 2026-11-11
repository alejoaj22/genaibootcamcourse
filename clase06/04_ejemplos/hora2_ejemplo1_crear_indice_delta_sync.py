# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 1 — Crear (o reutilizar) el Delta Sync Index y verificar la sincronización incremental
# MAGIC **Objetivo:** crear de forma idempotente el endpoint y el índice `dphone_vs_index`, inspeccionar su estado y demostrar cómo el Change Data Feed propaga una inserción y un borrado.
# MAGIC **Tiempo estimado:** 10 minutos (más ~5 minutos si el índice se crea por primera vez)
# MAGIC **Concepto que ilustra:** arquitectura endpoint → índice, Delta Sync con embeddings gestionados, CDF y `pipeline_type="TRIGGERED"`.
# MAGIC **Requisitos previos:** tabla `gold_docs_chunked` (notebook 1.1C); permisos para usar Vector Search y el endpoint `databricks-gte-large-en`.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

# 1) Cliente y endpoint compartido (mismo nombre que el notebook 1.2).
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)
vs_endpoint_name = "vs_endpoint"

if vs_client.endpoint_exists(vs_endpoint_name):
    print("El endpoint ya existe; esperando a que esté ONLINE...")
    vs_client.wait_for_endpoint(vs_endpoint_name)
else:
    print("Creando endpoint STANDARD (puede tardar 5-15 minutos)...")
    vs_client.create_endpoint_and_wait(vs_endpoint_name, endpoint_type="STANDARD")

# Información del endpoint.
info = vs_client.get_endpoint(vs_endpoint_name)
print("Tipo:", info.get("endpoint_type"), "| Estado:", info.get("endpoint_status", {}).get("state"))

# COMMAND ----------

# 2) Change Data Feed en la tabla fuente (requisito del Delta Sync Index).
source_table_fullname = f"{course.catalog}.{course.schema}.gold_docs_chunked"
spark.sql(f"ALTER TABLE {source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

props = spark.sql(f"SHOW TBLPROPERTIES {source_table_fullname}").where("key = 'delta.enableChangeDataFeed'")
display(props)

# COMMAND ----------

# 3) Índice Delta Sync con embeddings gestionados. Si ya existe, solo esperamos a que esté listo.
vs_index_fullname = f"{course.catalog}.{course.schema}.dphone_vs_index"

if not vs_client.index_exists(vs_endpoint_name, vs_index_fullname):
    print("Creando índice (≈5 minutos)...")
    vs_index = vs_client.create_delta_sync_index_and_wait(
        source_table_name=source_table_fullname,
        index_name=vs_index_fullname,
        primary_key="id",
        embedding_source_column="chunk",
        embedding_model_endpoint_name="databricks-gte-large-en",
        endpoint_name=vs_endpoint_name,
        pipeline_type="TRIGGERED",
    )
else:
    print("El índice ya existe; esperando a que termine cualquier actualización...")
    vs_index = vs_client.get_index(vs_endpoint_name, vs_index_fullname)
    vs_index.wait_until_ready(wait_for_updates=True)

# COMMAND ----------

# 4) Inspección del índice: estado, filas indexadas y configuración.
desc = vs_index.describe()
estado = desc.get("status", {})
print("Listo:", estado.get("ready"))
print("Estado detallado:", estado.get("detailed_state"))
print("Filas indexadas:", estado.get("indexed_row_count"))
print("Filas en la tabla Gold:", spark.read.table(source_table_fullname).count())

spec = desc.get("delta_sync_index_spec", {})
print("Pipeline:", spec.get("pipeline_type"))
print("Columna de embeddings:", spec.get("embedding_source_columns"))

# COMMAND ----------

# Índices disponibles en el endpoint (útil para ver los de otros compañeros si se comparte endpoint).
indices = vs_client.list_indexes(vs_endpoint_name).get("vector_indexes", [])
for ix in indices:
    print("-", ix.get("name"))

# COMMAND ----------

# Utilidad para convertir resultados de similarity_search en un DataFrame de pandas.
import pandas as pd

def a_df(resultados):
    columnas = [c["name"] for c in resultados.get("manifest", {}).get("columns", [])]
    filas = resultados.get("result", {}).get("data_array", []) or []
    return pd.DataFrame(filas, columns=columnas)

# COMMAND ----------

# 5) Demostración del CDF: insertamos un chunk de prueba con un id negativo (no choca con los reales).
TEXTO_PRUEBA = ("The dPhone Zeta concept is a foldable prototype with a transparent graphene display "
                "and a solar-charging back panel.")
# Si una ejecución anterior se interrumpió, eliminamos el residuo para que el ejemplo sea repetible.
spark.sql(f"DELETE FROM {source_table_fullname} WHERE id = -1")
spark.sql(f"""
    INSERT INTO {source_table_fullname} (path, chunk, id)
    VALUES ('test://chunk_de_prueba', '{TEXTO_PRUEBA}', -1)
""")
# Guardamos la versión de la tabla creada por el INSERT (la usaremos para leer el CDF).
version_insert = spark.sql(f"DESCRIBE HISTORY {source_table_fullname} LIMIT 1").first()["version"]
print("Versión de la tabla tras el INSERT:", version_insert)

# Con TRIGGERED, el índice no se entera hasta que llamamos sync().
vs_index.sync()
vs_index.wait_until_ready(wait_for_updates=True)

res = vs_index.similarity_search(query_text="foldable phone with solar charging",
                                 columns=["id", "path", "chunk"], num_results=3)
display(a_df(res))

# COMMAND ----------

# Vemos el cambio registrado en la bitácora (Change Data Feed) de la tabla.
# Importante: solo se puede leer el CDF desde versiones posteriores a su activación,
# por eso consultamos exactamente la versión del INSERT (no la versión 0).
display(spark.sql(f"""
    SELECT _change_type, _commit_version, id, path
    FROM table_changes('{source_table_fullname}', {version_insert}, {version_insert})
    WHERE id = -1
"""))

# COMMAND ----------

# 6) Limpieza: borramos el chunk de prueba y volvemos a sincronizar (el índice también lo elimina).
spark.sql(f"DELETE FROM {source_table_fullname} WHERE id = -1")
vs_index.sync()
vs_index.wait_until_ready(wait_for_updates=True)

res = vs_index.similarity_search(query_text="foldable phone with solar charging",
                                 columns=["id", "path", "chunk"], num_results=3)
print("¿Sigue apareciendo el chunk de prueba?", (-1 in a_df(res)["id"].astype(int).tolist()))

# COMMAND ----------

# Salida esperada:
# - Endpoint "vs_endpoint" tipo STANDARD en estado ONLINE.
# - delta.enableChangeDataFeed = true en la tabla Gold.
# - Índice listo con indexed_row_count igual al número de filas de gold_docs_chunked, pipeline TRIGGERED.
# - Tras insertar y sincronizar, el primer resultado es el chunk de prueba (id = -1, path test://chunk_de_prueba).
# - table_changes muestra una fila con _change_type = 'insert' para id = -1.
# - Tras borrar y sincronizar: "¿Sigue apareciendo el chunk de prueba? False".
