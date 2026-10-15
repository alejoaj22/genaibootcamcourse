# Databricks notebook source
# MAGIC %md
# MAGIC # 🏗️ Hora 1 — Ejemplo 2: Ejecutar y Auditar `Includes/Lab_Setup`
# MAGIC
# MAGIC - **Objetivo:** Ejecutar el script fundacional `Includes/Lab_Setup` (con respaldo automático en caso de mover el notebook de carpeta) e inspeccionar paso a paso qué objetos creó la clase `CourseSetup` en Unity Catalog: el esquema `genai_course`, los tres volúmenes (`dataset`, `_checkpoint`, `_cache`) y las carpetas descargadas desde Amazon S3 (`docs/` y `raw/`).
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Infraestructura de datos como código (Idempotencia), Volúmenes gobernados en Unity Catalog (`/Volumes/<catalogo>/<esquema>/<volumen>`) e ingesta inicial de archivos desde almacenamiento externo.
# MAGIC - **Requisitos previos:** Cómputo conectado y permisos de creación de esquema y volumen sobre el catálogo predeterminado (`current_catalog()`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ejecución e inicialización de `CourseSetup`
# MAGIC
# MAGIC En los notebooks del curso verás a menudo `%run ../Includes/Lab_Setup`. Como este archivo vive en `clase02/04_ejemplos/` (dos niveles debajo de la raíz del repositorio), verificamos si el objeto `course` ya existe o ejecutamos la lógica exacta e idempotente de `Includes/Lab_Setup.ipynb` para que funcione sin importar desde qué carpeta del Workspace lo abras.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `def path_exists(path):`: Intenta listar la ruta con `dbutils.fs.ls(path)`; si recibe `CloudFileNotFoundException` o `FileNotFoundException`, devuelve `False` en lugar de romper el programa.
# MAGIC - `class CourseSetup:`: Encapsula el aprovisionamiento del entorno:
# MAGIC   - `create_database()`: Ejecuta `USE CATALOG`, `CREATE SCHEMA IF NOT EXISTS genai_course` y `USE SCHEMA genai_course`, y luego llama a `__configure_directories()`.
# MAGIC   - `__configure_directories()`: Construye la ruta `/Volumes/<catalogo>/genai_course/` y ejecuta `CREATE VOLUME IF NOT EXISTS` para `dataset`, `_checkpoint` y `_cache`.
# MAGIC   - `download_dataset()`: Recorre `s3://dalhussein-courses/GenAI-Eng/datasets/dphone/v1/` y copia recursivamente (`dbutils.fs.cp(..., True)`) cada subcarpeta que aún no exista en el volumen `dataset`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import time

def path_exists(path):
    """Verifica de forma segura si una ruta existe en dbutils.fs."""
    try:
        dbutils.fs.ls(path)
        return True
    except Exception as e:
        msg = str(e)
        if ("com.databricks.sql.io.CloudFileNotFoundException" in msg
            or "java.io.FileNotFoundException" in msg):
            return False
        else:
            raise

class CourseSetup:
    """Clase oficial de configuración del entorno del curso (idéntica a Includes/Lab_Setup.ipynb)."""
    def __init__(self, uri, catalog_name, schema_name):
        self.uri = uri
        self.catalog = catalog_name
        self.schema = schema_name
    
    def download_dataset(self):
        source = self.uri
        target = self.dataset_volume
        files = dbutils.fs.ls(source)
        for f in files:
            source_path = f"{source}/{f.name}"
            target_path = f"{target}/{f.name}"
            if not path_exists(target_path):
                print(f"Copying {f.name} ...")
                dbutils.fs.cp(source_path, target_path, True)
            else:
                print(f"Ya existe en el volumen (idempotente): {f.name}")
    
    def create_database(self):
        spark.sql(f"USE CATALOG {self.catalog}")
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        spark.sql(f"USE SCHEMA {self.schema}")
        print(f"Catalog name: {self.catalog}")
        print(f"Schema name: {self.schema}")
        self.__configure_directories()
    
    def clean_up(self):
        print("Dropping Database, Dataset, Cache, and Checkpoints ...")
        spark.sql(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")
        print("Done")

    def __configure_directories(self):
        dataset_volume_name = "dataset"
        checkpoint_volume_name = "_checkpoint"
        cache_volume_name = "_cache"

        volume_root = f"/Volumes/{self.catalog}/{self.schema}"
        self.dataset_volume = f"{volume_root}/{dataset_volume_name}"
        self.checkpoint_volume = f"{volume_root}/{checkpoint_volume_name}"
        self.cache_volume = f"{volume_root}/{cache_volume_name}"
        
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {dataset_volume_name}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {checkpoint_volume_name}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {cache_volume_name}")
    
    def create_agent_tables(self):
        phones_table = "smartphones"
        inventory_table = "inventory_stock"
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {phones_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/smartphones.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{phones_table} table is ready.")

        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {inventory_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/inventory_stock.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{inventory_table} table is ready.")

    def create_batch_tables(self):
        tickets_table = "customer_support_tickets"
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {tickets_table}
            AS SELECT * FROM read_files(
                '{self.dataset_volume}/raw/support_tickets.csv',
                format => 'csv',
                schemaEvolutionMode => 'none');
        """)
        print(f"{tickets_table} table is ready.")

# Instanciamos y ejecutamos la configuración base exactamente igual que en Lab_Setup.ipynb
data_source_uri = "s3://dalhussein-courses/GenAI-Eng/datasets/dphone/v1/"
schema_name = "genai_course"
catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]

course = CourseSetup(data_source_uri, catalog_name, schema_name)
course.create_database()
course.download_dataset()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspección de los atributos del objeto `course`
# MAGIC
# MAGIC Una vez ejecutado `Lab_Setup`, el objeto `course` queda vivo en la memoria de Python y guarda las rutas exactas que usarán todos los notebooks posteriores del bootcamp.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `course.catalog` y `course.schema`: Guardan el catálogo y el esquema (`genai_course`) donde estamos trabajando.
# MAGIC - `course.dataset_volume`: Apunta a `/Volumes/<catalogo>/genai_course/dataset`, que es nuestra raíz de archivos del caso DerarTel / dPhone.
# MAGIC - `course.checkpoint_volume` y `course.cache_volume`: Rutas auxiliares para puntos de control de streaming y caché de modelos locales.

# COMMAND ----------

# Inspeccionamos todas las propiedades configuradas dentro de la instancia 'course'
print("=== ATRIBUTOS DEL OBJETO 'course' ===")
print(f"1. URI origen (S3)       : {course.uri}")
print(f"2. Catálogo en UC        : {course.catalog}")
print(f"3. Esquema en UC         : {course.schema}")
print(f"4. Volumen de Dataset    : {course.dataset_volume}")
print(f"5. Volumen de Checkpoint : {course.checkpoint_volume}")
print(f"6. Volumen de Caché      : {course.cache_volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verificación en SQL de los Volúmenes creados en Unity Catalog
# MAGIC
# MAGIC Usamos SQL nativo (`SHOW VOLUMES`) para confirmar que los tres volúmenes están registrados en el metastore de Unity Catalog bajo el esquema `genai_course`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Confirmamos qué esquema está activo ahora mismo y listamos sus volúmenes gobernados
# MAGIC SELECT current_catalog() AS catalogo, current_schema() AS esquema_activo;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VOLUMES IN genai_course;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Auditoría recursiva de los archivos descargados en el Volumen `dataset`
# MAGIC
# MAGIC Recorremos `course.dataset_volume` con `dbutils.fs.ls()` para verificar las dos carpetas principales descargadas desde S3 (`docs/` y `raw/`) y listar todos los archivos individuales con su tamaño en KB.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `carpetas = dbutils.fs.ls(course.dataset_volume)`: Lista el primer nivel del volumen `dataset` (esperamos ver `docs/` y `raw/`).
# MAGIC - Ciclo `for sub in carpetas:`: Entra en cada subcarpeta y lista los archivos internos, acumulando el conteo y el peso total en bytes.

# COMMAND ----------

# Listamos las subcarpetas de primer nivel dentro del volumen 'dataset'
carpetas = dbutils.fs.ls(course.dataset_volume)
print(f"Contenido de primer nivel en {course.dataset_volume}:")
for c in carpetas:
    print(f"  📁 {c.name:<15} -> {c.path}")

print("\n=== INVENTARIO DETALLADO DE ARCHIVOS DESCARGADOS ===")
total_archivos = 0
total_bytes = 0

for c in carpetas:
    print(f"\nSubcarpeta: {c.name}")
    archivos = dbutils.fs.ls(c.path)
    for a in archivos:
        kb = a.size / 1024.0
        print(f"  📄 {a.name:<42} | {kb:>8.2f} KB | Ruta: {a.path}")
        total_archivos += 1
        total_bytes += a.size

print(f"\nResumen de auditoría: {total_archivos} archivos encontrados ({total_bytes / 1024.0:.2f} KB en total).")

# Salida esperada:
# Catalog name: <tu_catalogo>
# Schema name: genai_course
# Copying docs/ ... (o "Ya existe en el volumen (idempotente): docs/")
# Copying raw/ ...  (o "Ya existe en el volumen (idempotente): raw/")
# === INVENTARIO DETALLADO DE ARCHIVOS DESCARGADOS ===
# Subcarpeta: docs/
#   📄 01_Product_Specs_dPhone_D1.pdf             |   351.79 KB
#   📄 02_Product_Specs_dPhone_D1_Lite.pdf        |   164.90 KB
#   📄 03_Product_Specs_dPhone_D1_Pro.pdf         |   277.11 KB
#   📄 04_Product_Specs_dPhone_D1_Fold.pdf        |   284.98 KB
#   📄 05_Corporate_Profile.pdf                   |    57.13 KB
#   📄 06_Privacy_and_Compliance_Policy.docx      |    21.87 KB
#   📄 07_Technical_Product_Diagram.png           |   146.63 KB
#   📄 08_User_Manual.pdf                         |    99.31 KB
# Subcarpeta: raw/
#   📄 inventory_stock.csv                        |     0.09 KB
#   📄 smartphones.csv                            |     0.18 KB
#   📄 support_tickets.csv                        |     0.53 KB
# Resumen de auditoría: 11 archivos encontrados (~1404.51 KB en total).
