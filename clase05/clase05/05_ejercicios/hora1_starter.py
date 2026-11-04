# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio 1 (Hora 1) — Starter: Diagrama de arquitectura RAG para tu reto
# MAGIC **Objetivo:** documentar fuentes, capas Medallion y riesgos de tu reto y generar un diagrama Mermaid.
# MAGIC **Tiempo estimado:** 15 minutos
# MAGIC **Concepto que ilustra:** diseño de arquitectura RAG + Medallion para datos no estructurados.
# MAGIC **Requisitos previos:** clúster encendido (no requiere datos). Reemplaza cada `# TODO`.

# COMMAND ----------

# TODO 1: describe tu reto y la necesidad de información (una frase)
reto = "TODO: nombre de tu reto"
necesidad = "TODO: qué preguntas debe resolver el asistente"

# COMMAND ----------

# TODO 2: lista al menos 3 fuentes. Cada una: (nombre, formato, tipo) donde tipo es "RAG" o "SQL"
fuentes = [
    # ("Manual de usuario", "PDF nativo", "RAG"),
    # TODO: agrega tus fuentes aquí
]

# COMMAND ----------

# TODO 3: describe las columnas de cada capa
capas = {
    "bronze": ["TODO: columnas de Bronze"],
    "silver": ["TODO: columnas de Silver"],
    "gold":   ["TODO: columnas de Gold"],
}

# COMMAND ----------

# TODO 4: riesgos de calidad y su mitigación: (riesgo, etapa, mitigacion)
riesgos = [
    # ("PDF escaneado", "Extracción", "Revisión visual y fallback a OCR"),
    # TODO: agrega al menos 3
]

# COMMAND ----------

# Generación del diagrama Mermaid (no editar): usa tus datos para construir el grafo.
lineas = ["flowchart LR"]
for i, (nombre, formato, tipo) in enumerate(fuentes):
    destino = "B" if tipo == "RAG" else "SQLQ"
    lineas.append(f'  S{i}["{nombre}<br/>{formato}"] --> {destino}')
lineas.append('  B["Bronze<br/>' + ", ".join(capas["bronze"]) + '"] --> S["Silver<br/>' + ", ".join(capas["silver"]) + '"]')
lineas.append('  S --> G["Gold<br/>' + ", ".join(capas["gold"]) + '"] --> V["Vector Search"] --> A["Asistente / Agente"]')
lineas.append('  SQLQ["Tablas SQL"] --> A')
mermaid = "\n".join(lineas)
print(mermaid)

# COMMAND ----------

# Tabla de riesgos y autoverificación mínima.
if riesgos:
    display(spark.createDataFrame(riesgos, ["riesgo", "etapa", "mitigacion"]))
assert len(fuentes) >= 3, "Faltan fuentes (mínimo 3)"
assert len(riesgos) >= 3, "Faltan riesgos (mínimo 3)"
print("Autoverificación superada")

# COMMAND ----------

# Salida esperada:
# - Con los TODO sin completar, fallará el assert de fuentes (comportamiento esperado).
# - Completado: texto Mermaid con el flujo fuentes → Bronze → Silver → Gold → Vector Search → Asistente,
#   una tabla de ≥ 3 riesgos y el mensaje "Autoverificación superada".
