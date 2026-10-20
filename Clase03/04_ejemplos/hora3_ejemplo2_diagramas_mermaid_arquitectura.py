# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejemplo 2: Generación de Algoritmos, Condicionales y Diagramas Mermaid.js con IA
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
# MAGIC
# MAGIC * **Objetivo:** Utilizar `ai_query()` para traducir reglas complejas de negocio con múltiples ramas condicionales en: (1) Pseudocódigo estructurado con tipos de datos explícitos y (2) Diagramas de flujo y secuencia en sintaxis **Mermaid.js** renderizados visualmente dentro del notebook de Databricks.
# MAGIC * **Tiempo estimado:** 9 minutos.
# MAGIC * **Concepto que ilustra:** *Diagrams as Code* asistido por IA, formalización previa a la codificación y renderizado interactivo con `displayHTML()`.
# MAGIC * **Requisitos previos:** Clúster en Databricks con acceso a `ai_query()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Función Renderizadora de Diagramas Mermaid.js en Databricks
# MAGIC Construimos una función auxiliar que toma cualquier bloque de código `Mermaid.js` generado por el LLM, limpia posibles etiquetas markdown y lo dibuja gráficamente en la celda de salida mediante `displayHTML()`.

# COMMAND ----------

import re
from typing import Dict

MODELO_DIAGRAMADOR = "databricks-gpt-oss-120b"

def limpiar_codigo_mermaid(texto: str) -> str:
    """Extrae únicamente el código Mermaid eliminando backticks si existen."""
    bloque = re.search(r"```(?:mermaid)?\s*(.*?)```", texto, re.DOTALL)
    if bloque:
        return bloque.group(1).strip()
    return texto.strip()

def renderizar_mermaid_en_databricks(codigo_mermaid: str, titulo: str = "Diagrama Generado por IA") -> None:
    """Renderiza visualmente un diagrama Mermaid dentro de un notebook de Databricks."""
    mermaid_limpio = limpiar_codigo_mermaid(codigo_mermaid)
    html = f"""
    <div style="font-family: sans-serif; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; background: #ffffff;">
        <h3 style="margin-top: 0; color: #1f2328;">{titulo}</h3>
        <div class="mermaid">
{mermaid_limpio}
        </div>
        <details style="margin-top: 12px;">
            <summary style="cursor: pointer; color: #0969da; font-weight: 600;">Ver código fuente Mermaid.js</summary>
            <pre style="background: #f6f8fa; padding: 10px; border-radius: 6px; overflow-x: auto;">{mermaid_limpio}</pre>
        </details>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    try:
        displayHTML(html)
    except NameError:
        print(mermaid_limpio)

def consultar_llm_mermaid(prompt: str, modelo: str = MODELO_DIAGRAMADOR) -> str:
    prompt_esc = prompt.replace("\\", "\\\\").replace("'", "''")
    sql = f"""
    SELECT ai_query(
        '{modelo}',
        '{prompt_esc}',
        modelParameters => named_struct('temperature', CAST(0.0 AS DOUBLE), 'max_tokens', CAST(650 AS INT))
    ) AS respuesta
    """
    try:
        res = spark.sql(sql).collect()[0]["respuesta"]
    except Exception:
        res = spark.sql(f"SELECT ai_query('{modelo}', '{prompt_esc}') AS respuesta").collect()[0]["respuesta"]
    return res.strip() if res else ""

print("Utilidades de generación y renderizado Mermaid listas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Caso 1: De Política de Crédito y Antifraude a Pseudocódigo Tipado + `flowchart TD`
# MAGIC Antes de programar un motor de decisión en Python o SQL, pedimos al modelo que estructure las condiciones y genere un diagrama de flujo `flowchart TD` sin errores de sintaxis (envolviendo las etiquetas con paréntesis o símbolos entre comillas dobles).

# COMMAND ----------

reglas_motor_riesgo = """
El motor de aprobación instantánea de microcréditos evalúa cada solicitud así:
1. Primero valida si el documento de identidad está en la lista de bloqueo (bloqueado: bool). Si está bloqueado, rechaza inmediatamente con código 'RECHAZO_LISTA_NEGRA'.
2. Si no está bloqueado, verifica la edad del solicitante (edad: int). Si es menor de 21 o mayor de 75 años, rechaza con 'RECHAZO_POLITICA_EDAD'.
3. Luego evalúa el puntaje crediticio (score: int entre 0 y 1000) y el ratio de endeudamiento (dti: float entre 0.0 y 1.0):
   - Si score >= 750 y dti <= 0.40: Aprueba automáticamente con 'APROBADO_PREFERENCIAL'.
   - Si score >= 600 y dti <= 0.50: Aprueba con condiciones ('APROBADO_ESTANDAR').
   - Si score >= 550 y dti > 0.50: Envía a comité humano ('REVISION_MANUAL_ANALISTA').
   - En cualquier otro caso: Rechaza con 'RECHAZO_SCORE_INSUFICIENTE'.
""".strip()

prompt_flowchart = f"""
### ROL
Actúa como un Arquitecto de Software experto en diseño algorítmico y Mermaid.js.

### REGLAS DE NEGOCIO DE ENTRADA
{reglas_motor_riesgo}

### TAREA
Genera únicamente el código de un diagrama de flujo en sintaxis `flowchart TD` de Mermaid.js que represente con total exactitud todos los nodos de decisión, tipos de datos y códigos de salida del motor de riesgo.

### REGLAS ESTRICTAS DE SINTAXIS MERMAID
1. Inicia exactamente con la línea `flowchart TD`.
2. TODA etiqueta de nodo o rombo que contenga paréntesis, dos puntos, operadores (`>=`, `<=`, `>`) o espacios DEBE ir encerrada entre comillas dobles, por ejemplo: `N1{{"¿score >= 750 y dti <= 0.40?"}}` o `R1["RECHAZO_LISTA_NEGRA"]`.
3. No uses etiquetas HTML dentro de los nodos.
4. Devuelve SOLO el código Mermaid plano sin explicaciones ni bloques markdown.
""".strip()

codigo_flowchart = consultar_llm_mermaid(prompt_flowchart)
print("=== CÓDIGO MERMAID GENERADO (FLOWCHART TD) ===")
print(limpiar_codigo_mermaid(codigo_flowchart))

renderizar_mermaid_en_databricks(codigo_flowchart, titulo="Árbol de Decisión del Motor de Riesgo Crediticio")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Caso 2: Generación de un Diagrama de Secuencia (`sequenceDiagram`) para una Arquitectura RAG + Agente
# MAGIC También podemos pedir a `ai_query()` que documente la interacción temporal entre microservicios, bases de datos vectoriales y endpoints de modelos usando `sequenceDiagram`.

# COMMAND ----------

prompt_secuencia = """
Genera un diagrama `sequenceDiagram` en Mermaid.js que muestre el flujo de una consulta en una arquitectura empresarial en Databricks:
1. El Usuario envía una pregunta al API Gateway.
2. El API Gateway invoca al Orquestador en Databricks.
3. El Orquestador consulta el Vector Search Index para recuperar los 3 fragmentos relevantes.
4. El Orquestador ensambla el contexto y llama a `ai_query('databricks-gpt-oss-120b')`.
5. El modelo devuelve la respuesta fundamentada, el Orquestador registra la traza en una tabla Delta de auditoría y responde al Usuario.

RESTRICCIÓN: Devuelve únicamente el código Mermaid iniciando con `sequenceDiagram`, sin bloques markdown ni explicaciones.
""".strip()

codigo_secuencia = consultar_llm_mermaid(prompt_secuencia)
print("=== CÓDIGO MERMAID GENERADO (SEQUENCE DIAGRAM) ===")
print(limpiar_codigo_mermaid(codigo_secuencia))

renderizar_mermaid_en_databricks(codigo_secuencia, titulo="Diagrama de Secuencia: Pipeline de Inferencia en Databricks")

# COMMAND ----------

# Salida esperada:
# 1. Código `flowchart TD` válido con todas las ramas del motor de riesgo crediticio y etiquetas entrecomilladas, renderizado visualmente en la celda de salida de Databricks.
# 2. Código `sequenceDiagram` mostrando los 5 pasos de interacción entre Usuario, API Gateway, Orquestador, Vector Search, `ai_query` y Delta Lake, renderizado en HTML.
