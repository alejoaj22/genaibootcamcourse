# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 1 (solución) — Matriz de decisión: Prompting vs RAG vs Fine-tuning
# MAGIC **Objetivo:** solución de referencia con pesos razonados, recomendación por escenario, justificación y análisis de sensibilidad.
# MAGIC **Tiempo estimado:** 5 minutos (lectura y ejecución)
# MAGIC **Concepto que ilustra:** decisión empresarial entre prompting, RAG y fine-tuning (analítico; no se entrena nada).
# MAGIC **Requisitos previos:** ninguno (Python y pandas).
# MAGIC
# MAGIC > Los pesos de esta solución son **una** posibilidad defendible; otras parejas pueden llegar a pesos distintos y conclusiones válidas si las argumentan.

# COMMAND ----------

import pandas as pd

CRITERIOS = ["frescura_datos", "citas_trazabilidad", "aprovecha_ejemplos", "latencia_costo",
             "estilo_formato", "gobernanza_permisos", "esfuerzo_implementacion"]

OPCIONES = ["Prompting", "RAG", "Fine-tuning (LoRA)", "RAG + Fine-tuning"]

# Matriz genérica de puntajes (0 a 5; 5 = la opción satisface muy bien el criterio).
PUNTAJES = pd.DataFrame(
    [
        [1, 1, 2, 3, 3, 3, 5],   # Prompting: barato de implementar, pero no actualiza ni cita
        [5, 5, 2, 2, 3, 5, 3],   # RAG: datos frescos, citas y permisos; más tokens y piezas
        [1, 1, 5, 4, 5, 2, 1],   # Fine-tuning: estilo y costo por consulta; datos "quedan en los pesos"
        [5, 5, 5, 3, 5, 3, 1],   # RAG + Fine-tuning: lo mejor en calidad, lo más caro de construir
    ],
    index=OPCIONES, columns=CRITERIOS,
)
display(PUNTAJES)

# COMMAND ----------

# Pesos razonados (cada escenario suma 1.0).
PESOS = {
    # A: la información cambia cada semana, hay que citar y respetar permisos de distribuidores.
    "A - Soporte dPhone (manuales cambian cada semana)": {
        "frescura_datos": 0.25, "citas_trazabilidad": 0.20, "aprovecha_ejemplos": 0.02,
        "latencia_costo": 0.10, "estilo_formato": 0.05, "gobernanza_permisos": 0.23,
        "esfuerzo_implementacion": 0.15,
    },
    # B: volumen enorme, latencia baja, formato JSON fijo y 50.000 ejemplos etiquetados.
    "B - Clasificación de 200.000 tickets/día": {
        "frescura_datos": 0.02, "citas_trazabilidad": 0.02, "aprovecha_ejemplos": 0.25,
        "latencia_costo": 0.30, "estilo_formato": 0.20, "gobernanza_permisos": 0.06,
        "esfuerzo_implementacion": 0.15,
    },
    # C: normas vigentes (frescura y citas), formato jurídico estricto y 5.000 ejemplos de abogados.
    "C - Asistente jurídico bancario": {
        "frescura_datos": 0.20, "citas_trazabilidad": 0.20, "aprovecha_ejemplos": 0.12,
        "latencia_costo": 0.03, "estilo_formato": 0.20, "gobernanza_permisos": 0.15,
        "esfuerzo_implementacion": 0.10,
    },
}

# COMMAND ----------

# Validación de los pesos: deben estar completos y sumar 1.0.
def validar_pesos(pesos):
    problemas = []
    for escenario, w in pesos.items():
        if any(v is None for v in w.values()):
            problemas.append(f"{escenario}: faltan pesos (hay None).")
        elif abs(sum(w.values()) - 1.0) > 1e-6:
            problemas.append(f"{escenario}: los pesos suman {sum(w.values()):.2f}, deben sumar 1.00.")
    return problemas

problemas = validar_pesos(PESOS)
assert not problemas, problemas
print("✅ Pesos válidos en los tres escenarios.")

# COMMAND ----------

# Cálculo del puntaje ponderado por opción.
def puntaje_ponderado(pesos_escenario):
    vector = pd.Series(pesos_escenario)[CRITERIOS]
    return (PUNTAJES * vector).sum(axis=1).round(2).sort_values(ascending=False)

resultados = {}
for escenario, w in PESOS.items():
    resultados[escenario] = puntaje_ponderado(w)
    print(f"\n{escenario}")
    display(resultados[escenario].to_frame("puntaje_ponderado"))
    print("→ Recomendación:", resultados[escenario].index[0])

# COMMAND ----------

# Justificaciones (2 líneas por escenario: por qué gana y riesgo principal).
JUSTIFICACIONES = {
    "A - Soporte dPhone (manuales cambian cada semana)":
        "Gana RAG (4,24): la información cambia semanalmente y debe citarse; con RAG basta actualizar la tabla/índice y se pueden aplicar permisos por distribuidor. "
        "Riesgo: calidad de recuperación (chunking, índice desactualizado); mitigación: sincronizar el índice y evaluar el retrieval.",
    "B - Clasificación de 200.000 tickets/día":
        "Gana Fine-tuning con LoRA (3,78): con 50.000 ejemplos etiquetados, un modelo pequeño ajustado da formato JSON estable, baja latencia y bajo costo por ticket. "
        "Riesgo: deriva si cambian las categorías (hay que reentrenar) y que RAG + FT queda muy cerca (3,68) si más adelante se necesita conocimiento externo.",
    "C - Asistente jurídico bancario":
        "Gana RAG + Fine-tuning ligero (4,24): normas que cambian cada mes y deben citarse (RAG) y formato jurídico estricto aprendido de 5.000 ejemplos (LoRA). "
        "Riesgo: mayor esfuerzo y datos sensibles; mitigación: anonimizar los ejemplos, gobernar el índice con Unity Catalog y evaluar con abogados.",
}
for escenario, texto in JUSTIFICACIONES.items():
    print(f"- {escenario}\n  {texto}\n")

# COMMAND ----------

# Análisis de sensibilidad: duplicamos el peso de gobernanza_permisos y renormalizamos.
def sensibilidad(pesos_escenario, criterio="gobernanza_permisos", factor=2.0):
    w = dict(pesos_escenario)
    w[criterio] = w[criterio] * factor
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    return puntaje_ponderado(w).index[0]

filas = []
for escenario, w in PESOS.items():
    base = resultados[escenario].index[0]
    nueva = sensibilidad(w)
    filas.append({"escenario": escenario, "ganador_base": base,
                  "ganador_con_gobernanza_x2": nueva, "cambia": base != nueva})
tabla_sens = pd.DataFrame(filas)
display(tabla_sens)

# COMMAND ----------

# Verificación automática de la solución de referencia.
esperado = {
    "A - Soporte dPhone (manuales cambian cada semana)": "RAG",
    "B - Clasificación de 200.000 tickets/día": "Fine-tuning (LoRA)",
    "C - Asistente jurídico bancario": "RAG + Fine-tuning",
}
for escenario, ganador in esperado.items():
    assert resultados[escenario].index[0] == ganador, f"{escenario}: esperado {ganador}"
print("✅ La recomendación de cada escenario coincide con la solución de referencia.")

LECCION = ("La principal lección de la matriz es: no existe una técnica universal; los pesos de los criterios "
           "(frescura, costo, estilo, gobernanza) cambian el ganador, y RAG y fine-tuning se complementan.")
print(LECCION)

# COMMAND ----------

# Salida esperada:
# - Matriz de puntajes de 4 opciones × 7 criterios y "✅ Pesos válidos en los tres escenarios."
# - Escenario A: RAG 4,24 · RAG + Fine-tuning 3,74 · Prompting 2,38 · Fine-tuning (LoRA) 1,81.
# - Escenario B: Fine-tuning (LoRA) 3,78 · RAG + Fine-tuning 3,68 · Prompting 2,97 · RAG 2,65.
# - Escenario C: RAG + Fine-tuning 4,24 · RAG 3,95 · Fine-tuning (LoRA) 2,52 · Prompting 2,28.
# - Justificaciones impresas y tabla de sensibilidad con gobernanza ×2: A sigue siendo RAG (cambia = False); B sigue siendo Fine-tuning (LoRA) (cambia = False); C CAMBIA a RAG (4,09 frente a 4,08 de RAG + Fine-tuning): con datos sensibles, mayor peso a la gobernanza favorece a RAG porque el fine-tuning deja datos en los pesos.
# - "✅ La recomendación de cada escenario coincide con la solución de referencia."
