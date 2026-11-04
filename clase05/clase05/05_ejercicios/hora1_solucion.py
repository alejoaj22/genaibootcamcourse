# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 1 (Hora 1) — Solución: Arquitectura RAG para el asistente de soporte dPhone
# MAGIC **Objetivo:** ejemplo resuelto de fuentes, capas Medallion, riesgos y diagrama Mermaid.
# MAGIC **Tiempo estimado:** 5 minutos
# MAGIC **Concepto que ilustra:** diseño de arquitectura RAG + Medallion para datos no estructurados.
# MAGIC **Requisitos previos:** clúster encendido (no requiere datos).

# COMMAND ----------

# Reto de ejemplo: asistente de soporte para clientes de dPhone.
reto = "Asistente de soporte dPhone"
necesidad = "Responder preguntas de garantía, uso, configuración y disponibilidad de equipos con citas verificables"

# COMMAND ----------

# Fuentes: documentos (RAG) y tablas (SQL). Los documentos tienen texto libre; las cifras exactas viven en tablas.
fuentes = [
    ("Manuales y guías dPhone", "PDF nativo y DOCX", "RAG"),
    ("Políticas de garantía escaneadas", "PDF escaneado", "RAG"),
    ("Catálogo de smartphones", "Tabla (CSV)", "SQL"),
    ("Inventario y stock", "Tabla (CSV)", "SQL"),
]

# COMMAND ----------

# Columnas por capa (Gold se detalla en la Clase 6).
capas = {
    "bronze": ["path", "modificationTime", "length", "content", "ingestion_time"],
    "silver": ["path", "parsed_content", "plain_text", "error_status"],
    "gold":   ["chunk_id", "path", "chunk_text", "embedding"],
}

# COMMAND ----------

# Riesgos con mitigación concreta por etapa del ciclo.
riesgos = [
    ("PDF escaneado con baja calidad", "Extracción", "Revisión visual con document_renderer y cuarentena si hay poco texto por página"),
    ("Tablas complejas mal parseadas", "Extracción", "Validar elementos tipo table y reintentar con otro enfoque"),
    ("Documentos duplicados o versiones antiguas", "Ingestión", "Hash sha2 del contenido y uso de modificationTime"),
    ("Información sensible en documentos", "Ingestión", "Gobierno con Unity Catalog y permisos en el Volume"),
    ("El LLM formateador altera el contenido", "Extracción", "Comparar longitud original vs plain_text y muestrear manualmente"),
]

# COMMAND ----------

# Generación del diagrama Mermaid.
lineas = ["flowchart LR"]
for i, (nombre, formato, tipo) in enumerate(fuentes):
    destino = "B" if tipo == "RAG" else "SQLQ"
    lineas.append(f'  S{i}["{nombre}<br/>{formato}"] --> {destino}')
lineas.append('  B["Bronze<br/>' + ", ".join(capas["bronze"]) + '"] --> S["Silver<br/>' + ", ".join(capas["silver"]) + '"]')
lineas.append('  S --> G["Gold<br/>' + ", ".join(capas["gold"]) + '"] --> V["Vector Search"] --> A["Asistente / Agente"]')
lineas.append('  SQLQ["Tablas SQL"] --> A')
mermaid = "\n".join(lineas)
print(mermaid)

# Mostramos el diagrama renderizado si hay acceso a internet.
displayHTML(f"""
<pre class="mermaid">{mermaid}</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{ startOnLoad: true }});
</script>
""")

# COMMAND ----------

# Tabla de riesgos y autoverificación.
display(spark.createDataFrame(riesgos, ["riesgo", "etapa", "mitigacion"]))
assert len(fuentes) >= 3, "Faltan fuentes (mínimo 3)"
assert len(riesgos) >= 3, "Faltan riesgos (mínimo 3)"
print("Autoverificación superada")

# COMMAND ----------

# Salida esperada:
# - Texto Mermaid con 4 fuentes que fluyen a Bronze (2 RAG) o a "Tablas SQL" (2 SQL).
# - Diagrama renderizado (con acceso a internet) y tabla de 5 riesgos.
# - "Autoverificación superada".
