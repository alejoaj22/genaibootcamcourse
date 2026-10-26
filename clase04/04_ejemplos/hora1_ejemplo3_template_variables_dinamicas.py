# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 — Ejemplo 3: Prompts Parametrizados con Variables Dinámicas y Versionado en Producción
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Construir motores de plantillas robustas para flujos de datos automatizados en Databricks, comparando f-strings, `string.Template` y `Jinja2` (con validación de variables obligatorias `StrictUndefined`, bucles y condicionales), integrando un registro versionado de prompts (`PROMPT_REGISTRY`) e invocaciones masivas mediante SQL (`ai_query` + `format_string`).
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Separación entre lógica de prompting y datos operacionales, prevención de inyecciones de sintaxis, validación en tiempo de compilación de parámetros faltantes y procesamiento distribuido de prompts sobre DataFrames de Apache Spark.
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b`.

# COMMAND ----------

# MAGIC %pip install jinja2

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración de Invocación y Helper Base

# COMMAND ----------

import json
import re
import string
import time
from typing import Dict, Any, List
import jinja2
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"


def _esc(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace("'", "\\'")


def extraer_texto(salida) -> str:
    if salida is None:
        return ""
    s = str(salida).strip()
    if s.startswith("[") and '"type"' in s:
        try:
            bloques = json.loads(s)
            textos = [b.get("text", "") for b in bloques if isinstance(b, dict) and b.get("type") == "text"]
            if textos:
                return "\n".join(textos).strip()
        except Exception:
            pass
    return s


def llamar_llm(prompt: str, modelo: str = MODELO_20B, temperature: float = 0.0,
               max_tokens: int = 800) -> dict:
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{_esc(prompt)}',
        modelParameters => named_struct('temperature', CAST({float(temperature)} AS DOUBLE),
                                        'max_tokens', CAST({int(max_tokens)} AS INT))
    ) AS respuesta
    """
    t0 = time.time()
    try:
        salida = spark.sql(sql).collect()[0]["respuesta"]
    except Exception as e:
        salida = spark.sql(f"SELECT ai_query('{modelo}', '{_esc(prompt)}') AS respuesta").collect()[0]["respuesta"]
    return {
        "texto": extraer_texto(salida),
        "latencia_s": round(time.time() - t0, 2)
    }


print("Entorno y librerías inicializadas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Los Tres Niveles de Parametrización de Prompts
# MAGIC
# MAGIC 1. **Nivel 1 (f-strings):** Fácil y rápido, pero propenso a errores cuando el prompt contiene llaves JSON `{}` y no valida variables nulas/faltantes de forma declarativa.
# MAGIC 2. **Nivel 2 (`string.Template`):** Usa sintaxis `$variable`. Más seguro frente a inyecciones de código Python arbitrario.
# MAGIC 3. **Nivel 3 (`Jinja2` con `StrictUndefined`):** Estándar de la industria. Permite directivas condicionales (`{% if %}`), bucles para inyectar ejemplos few-shot dinámicos (`{% for %}`) y detiene la ejecución inmediatamente si falta una variable obligatoria.

# COMMAND ----------

# Contexto de prueba: Datos de un cliente y ticket dPhone
PARAMETROS_CASO = {
    "nombre_cliente": "Valeria Gómez",
    "tipo_usuario": "VIP_ENTERPRISE",
    "modelo_dispositivo": "dPhone D1 Fold",
    "dias_compra": 45,
    "problema_reportado": "La pantalla interna presenta una línea negra horizontal tras doblarse.",
    "ejemplos_resolucion": [
        {"caso": "Falla en bisagra", "accion": "Reemplazo de unidad prioritario"},
        {"caso": "Daño accidental", "accion": "Cotización con descuento VIP"}
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implementación A: Plantilla con f-string (Básica)

# COMMAND ----------

prompt_fstring = f"""
<instructions>
Eres un analista de soporte de garantías DerarTel.
Cliente: {PARAMETROS_CASO['nombre_cliente']} ({PARAMETROS_CASO['tipo_usuario']})
Dispositivo: {PARAMETROS_CASO['modelo_dispositivo']} - Días de uso: {PARAMETROS_CASO['dias_compra']}
Problema: {PARAMETROS_CASO['problema_reportado']}
</instructions>
Responde con la acción a tomar en una sola frase.
"""
print("Prompt f-string generado exitosamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implementación B: Plantilla con Jinja2 Avanzado (Condicionales + Bucles Few-Shot + StrictUndefined)

# COMMAND ----------

PLANTILLA_JINJA_TEXTO = """
<instructions>
Eres un Analista Senior de Garantías y Retención para DerarTel Mobile.
Atiendes una solicitud de nivel prioritario.
</instructions>

<customer_profile>
Nombre: {{ nombre_cliente }}
Segmento: {{ tipo_usuario }}
Dispositivo: {{ modelo_dispositivo }}
Antigüedad: {{ dias_compra }} días desde activación
</customer_profile>

<business_rules>
{% if tipo_usuario == 'VIP_ENTERPRISE' %}
- PROTOCOLO VIP ACTIVO: Si el dispositivo tiene menos de 90 días, otorgar reemplazo express inmediato sin costo.
{% else %}
- PROTOCOLO ESTÁNDAR: Solicitar diagnóstico en taller autorizado si supera los 30 días.
{% endif %}
</business_rules>

<reference_examples>
{% for ex in ejemplos_resolucion %}
- Caso: {{ ex.caso }} -> Resolución sugerida: {{ ex.accion }}
{% endfor %}
</reference_examples>

<ticket_description>
{{ problema_reportado }}
</ticket_description>

<output_requirement>
Determina la resolución aplicable y cita la regla de negocio aplicada.
</output_requirement>
""".strip()

# El entorno Jinja con StrictUndefined genera error si una variable no fue suministrada
jinja_env = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False
)
template_jinja = jinja_env.from_string(PLANTILLA_JINJA_TEXTO)

prompt_jinja_renderizado = template_jinja.render(**PARAMETROS_CASO)
print("=== PROMPT RENDERIZADO CON JINJA2 ===")
print(prompt_jinja_renderizado[:400] + "\n[...resto del prompt...]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validación de Variables Faltantes (Fail-Fast Pattern)
# MAGIC Si enviamos parámetros incompletos a producción, `StrictUndefined` previene llamadas costosas al LLM con prompts truncados.

# COMMAND ----------

parametros_incompletos = {
    "nombre_cliente": "Carlos Pérez",
    # Falta 'tipo_usuario' a propósito
    "modelo_dispositivo": "dPhone D1 Lite",
    "dias_compra": 10,
    "problema_reportado": "No enciende",
    "ejemplos_resolucion": []
}

try:
    template_jinja.render(**parametros_incompletos)
    print("Inesperado: Renderizó sin error.")
except jinja2.UndefinedError as err:
    print(f"ÉXITO: Captura inmediata de variable faltante: {err}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Catálogo Versionado de Prompts (`PROMPT_REGISTRY`)
# MAGIC En ingeniería de IA en producción, los prompts no se escriben dispersos en el código; se versionan con identificadores únicos y metadata de auditoría.

# COMMAND ----------

PROMPT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "soporte_garantias_v1.0": {
        "version": "1.0",
        "descripcion": "Clasificador básico de incidentes de garantía",
        "template": template_jinja,
        "modelo_recomendado": MODELO_20B,
        "temperature": 0.0,
        "variables_obligatorias": ["nombre_cliente", "tipo_usuario", "modelo_dispositivo", "dias_compra", "problema_reportado"]
    }
}

print(f"Registro cargado con {len(PROMPT_REGISTRY)} versión(es) de prompt.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Invocación Masiva Distribuida en Spark con `ai_query` + SQL
# MAGIC Cuando procesamos miles de tickets, la mejor práctica en Databricks es usar una vista temporal y construir el prompt en SQL usando `format_string()` o `CONCAT()`.

# COMMAND ----------

datos_lote = [
    ("CLI-01", "Andrés Marín", "ESTANDAR", "dPhone D1", 15, "El parlante inferior distorsiona el sonido."),
    ("CLI-02", "Beatriz Soto", "VIP_ENTERPRISE", "dPhone D1 Fold", 20, "Bisagra bloqueada a 90 grados."),
    ("CLI-03", "Camilo Rúa", "ESTANDAR", "dPhone D1 Lite", 120, "Batería dura menos de 3 horas."),
    ("CLI-04", "Diana Vega", "VIP_ENTERPRISE", "dPhone D1 Pro", 10, "Cámara Halo no enfoca objetos lejanos."),
    ("CLI-05", "Esteban Gil", "ESTANDAR", "dPhone D1", 5, "Sensor biométrico Vein-Scan no responde.")
]

df_tickets = spark.createDataFrame(datos_lote, ["cliente_id", "nombre", "segmento", "modelo", "dias_uso", "problema"])
df_tickets.createOrReplaceTempView("tickets_operaciones_temp")

# Consulta SQL con ai_query procesando todo el lote en paralelo
query_batch_sql = f"""
WITH prompts_ensamblados AS (
    SELECT
        cliente_id,
        nombre,
        segmento,
        modelo,
        format_string(
            '<ticket cliente="%s" segmento="%s" modelo="%s" dias="%d"><issue>%s</issue></ticket> Analiza si aplica reemplazo express o taller.',
            nombre, segmento, modelo, dias_uso, problema
        ) AS prompt_completo
    FROM tickets_operaciones_temp
)
SELECT
    cliente_id,
    segmento,
    modelo,
    ai_query(
        '{MODELO_20B}',
        prompt_completo,
        modelParameters => named_struct('temperature', 0.0, 'max_tokens', 120)
    ) AS resolucion_ia
FROM prompts_ensamblados
"""

df_resultado_batch = spark.sql(query_batch_sql)
display(df_resultado_batch)

# COMMAND ----------

# Salida esperada:
# 1. Validación de Jinja2 capturando UndefinedError cuando falta una variable obligatoria.
# 2. Renderizado de prompt estructurado con directivas condicionales evaluadas según el perfil del cliente.
# 3. Consulta distribuida en Databricks SQL donde ai_query procesa las 5 filas en paralelo aplicando la lógica diferenciada para clientes VIP vs Estándar.
