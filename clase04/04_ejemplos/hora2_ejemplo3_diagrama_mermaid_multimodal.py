# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 — Ejemplo 3: Salidas Multiformato e Interpretación Bidireccional de Diagramas de Arquitectura (Mermaid.js y Visión)
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Demostrar la generación y validación de salidas multiformato a partir de un requerimiento técnico de dPhone (texto explicativo -> consulta Databricks SQL -> tabla Markdown -> diagramas Mermaid.js renderizables en Databricks con `displayHTML`), y ejecutar la **interpretación inversa**: alimentar al modelo con código Mermaid y diagramas visuales para auditoría de arquitectura, detección de riesgos y síntesis de DDL.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Representaciones semánticas estructuradas más allá de texto plano, verificación sintáctica de gramáticas Mermaid y razonamiento multimodal sobre topologías de sistemas y modelos entidad-relación (ERD).
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelo `databricks-gpt-oss-20b` (y opcionalmente modelo de visión).

# COMMAND ----------

# MAGIC %pip install pillow

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Infraestructura de Invocación y Helper Estándar

# COMMAND ----------

import json
import re
import time
from typing import Dict, Any
import pandas as pd

MODELO_20B = "databricks-gpt-oss-20b"
MODELO_120B = "databricks-gpt-oss-120b"


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
               max_tokens: int = 1000) -> dict:
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


def renderizar_mermaid(codigo_mermaid: str) -> None:
    """Renderiza código Mermaid en un notebook de Databricks usando el CDN oficial."""
    codigo_limpio = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", codigo_mermaid.strip())
    html = f"""
    <div class="mermaid" style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;">
    {codigo_limpio}
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    displayHTML(html)


print("Infraestructura multiformato lista.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generación Multiformato: De Requerimiento Técnico a Mermaid + SQL
# MAGIC Solicitamos al modelo estructurar la arquitectura Medallion de soporte de dPhone.

# COMMAND ----------

REQUERIMIENTO_SOPORTE = """
Requerimiento de Arquitectura de Datos para DerarTel:
Necesitamos diseñar el pipeline de procesamiento de tickets de soporte para la línea dPhone.
1. La capa Bronze ingesta eventos crudos desde la API de Zendesk y Webhooks de la app Fluid OS en JSON.
2. La capa Silver limpia los textos, ejecuta anonimización de PII, extrae el modelo de dPhone y calcula el sentimiento.
3. La capa Gold agrega métricas de fallas por componente (batería, cámara Halo, pantalla, bisagra) y calcula el tiempo medio de resolución (MTTR) por país.
4. Un módulo de Alertas tempranas notifica al equipo de Calidad de Hardware si la tasa de incidentes de un lote supera el 5%.
""".strip()

prompt_generar_mermaid = f"""
<instructions>
Eres un Arquitecto de Soluciones de Datos Senior en Databricks.
A partir del requerimiento técnico suministrado, genera un diagrama de flujo en sintaxis Mermaid.js (flowchart TD) que ilustre la arquitectura Medallion completa y el flujo de alertas.
</instructions>

<requirement>
{REQUERIMIENTO_SOPORTE}
</requirement>

<rules>
- Usa la sintaxis: flowchart TD
- Subgraphs obligatorios: Bronze, Silver, Gold, Notificaciones
- Devuelve ÚNICAMENTE el bloque de código Mermaid, sin comentarios adicionales.
</rules>
"""

res_mermaid = llamar_llm(prompt_generar_mermaid, modelo=MODELO_20B, temperature=0.0)
codigo_mermaid_generado = res_mermaid["texto"]
print("=== CÓDIGO MERMAID GENERADO ===")
print(codigo_mermaid_generado)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validación Sintáctica de Código Mermaid
# MAGIC Antes de renderizar en dashboards o documentación, verificamos reglas elementales de la gramática Mermaid (tokens de inicio válidos, balance de corchetes y comillas).

# COMMAND ----------

def validar_sintaxis_mermaid(codigo: str) -> Dict[str, Any]:
    c = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", codigo.strip())
    lineas = [l.strip() for l in c.split("\n") if l.strip() and not l.strip().startswith("%%")]
    if not lineas:
        return {"valido": False, "error": "Código vacío"}
    
    encabezados_validos = ("flowchart", "graph", "sequencediagram", "classdiagram", "erdiagram", "stateDiagram")
    primer_token = lineas[0].lower().split()[0]
    
    if not primer_token.startswith(encabezados_validos):
        return {"valido": False, "error": f"Tipo de diagrama no soportado o inválido: {primer_token}"}
        
    # Verificar balance de delimitadores
    corchetes_abiertos = c.count("[") - c.count("]")
    llaves_abiertas = c.count("{") - c.count("}")
    parentesis_abiertos = c.count("(") - c.count(")")
    
    if corchetes_abiertos != 0 or llaves_abiertas != 0 or parentesis_abiertos != 0:
        return {
            "valido": False,
            "error": f"Desbalance de delimitadores: [ {corchetes_abiertos}, {{ {llaves_abiertas}, ( {parentesis_abiertos}"
        }
        
    return {"valido": True, "tipo": primer_token, "lineas": len(lineas)}


check_mermaid = validar_sintaxis_mermaid(codigo_mermaid_generado)
print("Resultado de validación sintáctica Mermaid:", check_mermaid)

if check_mermaid["valido"]:
    print("Renderizando diagrama en Databricks...")
    renderizar_mermaid(codigo_mermaid_generado)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interpretación Inversa: Del Diagrama Mermaid a la Auditoría de Riesgos y DDL
# MAGIC Proporcionamos un diagrama Entidad-Relación (ERD) en Mermaid al LLM (`databricks-gpt-oss-120b`) para que realice una auditoría arquitectónica y derive las sentencias `CREATE TABLE` en Databricks SQL.

# COMMAND ----------

MERMAID_ERD_DPHONE = """
erDiagram
    CLIENTES ||--o{ TICKETS : "radica"
    PRODUCTOS ||--o{ TICKETS : "afecta"
    TICKETS ||--|{ HISTORIAL_GARANTIA : "genera"

    CLIENTES {
        string cliente_id PK
        string nombre
        string email
        string pais
        string segmento
    }
    PRODUCTOS {
        string modelo_sku PK
        string nombre_comercial
        int garantia_meses
        decimal precio_usd
    }
    TICKETS {
        string ticket_id PK
        string cliente_id FK
        string modelo_sku FK
        timestamp fecha_creacion
        string componente_falla
        string estado
    }
    HISTORIAL_GARANTIA {
        string reclamo_id PK
        string ticket_id FK
        string tipo_resolucion
        decimal costo_reparacion_usd
        boolean aprobado
    }
""".strip()

prompt_auditoria_erd = f"""
<instructions>
Eres un Principal Data Architect en Databricks.
Analiza el siguiente modelo Entidad-Relación en formato Mermaid ERD para el sistema de soporte y garantías de dPhone.
1. Identifica posibles cuellos de botella o debilidades en el modelado (ej. falta de trazabilidad de reemplazo de partes).
2. Genera el script en Databricks SQL para crear la tabla `TICKETS` aplicando mejores prácticas (tipos de datos precisos, restricciones Delta y particionamiento recomendado).
</instructions>

<erd_diagram>
{MERMAID_ERD_DPHONE}
</erd_diagram>

<output_rules>
- Presenta primero el análisis de riesgos arquitectónicos.
- Presenta luego el bloque Databricks SQL ejecutable.
</output_rules>
"""

res_auditoria = llamar_llm(prompt_auditoria_erd, modelo=MODELO_120B, temperature=0.0)
print("=== REPORTE DE AUDITORÍA ARQUITECTÓNICA Y DDL DERIVADO ===")
print(res_auditoria["texto"])

# COMMAND ----------

# Salida esperada:
# 1. Generación de código Mermaid.js sintácticamente correcto para la arquitectura Medallion de soporte dPhone.
# 2. Validación de gramática y renderizado interactivo en el notebook mediante displayHTML.
# 3. Interpretación inversa de un ERD en Mermaid produciendo recomendaciones de arquitectura y sentencias Databricks SQL optimizadas.
