# Databricks notebook source
# MAGIC %md
# MAGIC # 🧭 Hora 1 — Ejemplo 1: Explorar el Entorno de Trabajo en Databricks
# MAGIC
# MAGIC - **Objetivo:** Realizar el primer reconocimiento técnico del entorno de ejecución en Databricks, inspeccionando la sesión de Apache Spark, la versión de Python, el usuario activo, el catálogo y esquema predeterminados en Unity Catalog, el uso de comandos mágicos (`%md`, `%sql`, `%python`) y las utilidades `dbutils`.
# MAGIC - **Tiempo estimado:** 8 minutos.
# MAGIC - **Concepto que ilustra:** Arquitectura del Lakehouse en vivo, separación de cómputo y almacenamiento, y contexto de sesión en Unity Catalog (`catálogo.esquema`).
# MAGIC - **Requisitos previos:** Tener el notebook conectado a un recurso de cómputo (**Serverless** o clúster del curso en estado **Running**). No requiere haber ejecutado `Lab_Setup` todavía.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inspección del motor de ejecución (Python + Apache Spark)
# MAGIC
# MAGIC En un script tradicional de Python tendrías que importar e inicializar `SparkSession`. En un notebook de Databricks, el objeto global `spark` y el objeto de utilidades `dbutils` **ya vienen pre-inicializados** en cuanto conectas el cómputo.
# MAGIC
# MAGIC ### Explicación línea por línea de la siguiente celda:
# MAGIC - `import sys, platform`: Importamos los módulos estándar de Python para consultar metadatos del sistema operativo y del intérprete.
# MAGIC - `print(f"Versión de Python: {sys.version.split()[0]}")`: Imprime la versión exacta del intérprete Python que corre en el nodo Driver.
# MAGIC - `print(f"Arquitectura / SO: {platform.platform()}")`: Muestra el contenedor Linux subyacente administrado por Databricks.
# MAGIC - `print(f"Versión de Apache Spark: {spark.version}")`: Consulta el atributo `.version` del objeto global `spark` pre-cargado por el Databricks Runtime.

# COMMAND ----------

# Inspección del motor Python y Apache Spark pre-inicializado en Databricks
import sys
import platform

# Mostramos la versión exacta del intérprete Python activo en el contenedor
print(f"Versión de Python       : {sys.version.split()[0]}")

# Mostramos el sistema operativo subyacente donde corre el nodo Driver
print(f"Arquitectura / SO       : {platform.platform()}")

# Verificamos que el objeto global 'spark' ya existe sin necesidad de crearlo manualmente
print(f"Versión de Apache Spark : {spark.version}")
print(f"Tipo del objeto spark   : {type(spark)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ¿Quién soy y dónde estoy parado en Unity Catalog? (Desde Python con `spark.sql`)
# MAGIC
# MAGIC Todo activo en Databricks vive bajo una dirección de tres niveles: `catalogo.esquema.objeto`. Antes de crear cualquier tabla o volumen, un buen ingeniero verifica siempre su contexto actual.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `contexto_df = spark.sql(...)`: Ejecuta una consulta SQL desde Python y devuelve un DataFrame de PySpark sin traer aún los datos a memoria local (evaluación perezosa o *lazy evaluation*).
# MAGIC - `SELECT current_user() AS usuario_actual`: Función nativa que devuelve el correo electrónico de tu cuenta autenticada en Databricks.
# MAGIC - `current_catalog() AS catalogo_actual`: Devuelve el catálogo raíz asignado por defecto a tu sesión.
# MAGIC - `current_schema() AS esquema_actual`: Devuelve el esquema o base de datos activa (por defecto suele ser `default` antes de correr `Lab_Setup`).
# MAGIC - `fila = contexto_df.collect()[0]`: Trae la primera fila del resultado al nodo Driver para imprimir sus valores en consola.
# MAGIC - `display(contexto_df)`: Función especial de Databricks que renderiza cualquier DataFrame como una tabla interactiva HTML con opciones de ordenamiento, filtrado y descarga.

# COMMAND ----------

# Consultamos nuestro contexto de identidad y ubicación en Unity Catalog desde Python
contexto_df = spark.sql("""
    SELECT 
        current_user()      AS usuario_actual,
        current_catalog()   AS catalogo_actual,
        current_schema()    AS esquema_actual,
        current_timestamp() AS fecha_hora_servidor
""")

# Extraemos la primera fila como objeto Row de PySpark para imprimir un resumen legible
fila = contexto_df.collect()[0]
print("=== RESUMEN DE SESIÓN EN UNITY CATALOG ===")
print(f"Usuario autenticado : {fila['usuario_actual']}")
print(f"Catálogo activo     : {fila['catalogo_actual']}")
print(f"Esquema activo      : {fila['esquema_actual']}")
print(f"Hora del servidor   : {fila['fecha_hora_servidor']}")

# Renderizamos la tabla interactiva nativa de Databricks
display(contexto_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Poliglotismo en acción: ejecutando SQL nativo con el comando mágico `%sql`
# MAGIC
# MAGIC Aunque este notebook tenga **Python** como lenguaje principal, podemos convertir cualquier celda en un editor SQL puro escribiendo `%sql` en su primera línea.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `# MAGIC %sql`: Le indica a Databricks que el contenido de esta celda debe enviarse directamente al motor SQL del Lakehouse y que el resultado debe mostrarse automáticamente con `display()`.
# MAGIC - `SHOW SCHEMAS IN ...` / `SHOW SCHEMAS`: Lista todos los esquemas (bases de datos) visibles dentro de nuestro catálogo actual.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Listamos todos los esquemas disponibles actualmente dentro de nuestro catálogo activo
# MAGIC SHOW SCHEMAS;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Explorando la caja de herramientas `dbutils` (Databricks Utilities)
# MAGIC
# MAGIC `dbutils` es un objeto exclusivo de Databricks para interactuar con el sistema de archivos (`dbutils.fs`), secretos (`dbutils.secrets`), encadenamiento de notebooks (`dbutils.notebook`) y controles de interfaz (`dbutils.widgets`).
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `dbutils.fs.help()` (o inspeccionar sus métodos): Muestra las operaciones de sistema de archivos soportadas (`ls`, `cp`, `mv`, `rm`, `mkdirs`, `head`, `put`).
# MAGIC - `raices = dbutils.fs.ls("/")`: Lista el directorio raíz virtual del entorno para comprobar que el sistema de archivos responde correctamente.

# COMMAND ----------

# Inspeccionamos los submódulos disponibles dentro de dbutils
submodulos = [attr for attr in dir(dbutils) if not attr.startswith("_")]
print(f"Submódulos disponibles en dbutils: {submodulos}\n")

# Listamos las rutas raíz visibles desde dbutils.fs
raices = dbutils.fs.ls("/")
print("Rutas en la raíz del sistema de archivos ('/'):")
for item in raices:
    print(f"  - Ruta: {item.path:<25} | Nombre: {item.name:<20} | Tamaño: {item.size} bytes")

# Salida esperada:
# Versión de Python       : 3.11.x (o 3.12.x según el Databricks Runtime)
# Arquitectura / SO       : Linux-...-x86_64-with-glibc...
# Versión de Apache Spark : 3.5.x / 4.0.x
# Tipo del objeto spark   : <class 'pyspark.sql.connect.session.SparkSession'> (o SparkSession clásico)
# === RESUMEN DE SESIÓN EN UNITY CATALOG ===
# Usuario autenticado : <tu_correo>@...
# Catálogo activo     : workspace (o el nombre del catálogo del curso)
# Esquema activo      : default
# Submódulos disponibles en dbutils: ['credentials', 'data', 'fs', 'jobs', 'library', 'notebook', 'preview', 'secrets', 'widgets']
# Rutas en la raíz del sistema de archivos ('/'):
#   - Ruta: dbfs:/Volumes/            | Nombre: Volumes/             | Tamaño: 0 bytes
#   - Ruta: dbfs:/Workspace/          | Nombre: Workspace/           | Tamaño: 0 bytes
#   - Ruta: dbfs:/databricks-datasets/| Nombre: databricks-datasets/ | Tamaño: 0 bytes
