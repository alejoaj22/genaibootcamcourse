# Databricks notebook source
# MAGIC %md
# MAGIC # Ejercicio Hora 1 (starter) — Matriz de decisión: Prompting vs RAG vs Fine-tuning
# MAGIC **Objetivo:** completar los pesos de tres escenarios y obtener una recomendación ponderada y argumentada.
# MAGIC **Tiempo estimado:** 8 minutos
# MAGIC **Concepto que ilustra:** decisión empresarial entre prompting, RAG y fine-tuning (analítico; no se entrena nada).
# MAGIC **Requisitos previos:** haber leído `hora1_enunciado.md`. Solo usa Python y pandas.
# MAGIC
# MAGIC Las celdas con `# TODO` deben completarse. El resto se ejecuta tal cual.

# COMMAND ----------

import pandas as pd

CRITERIOS = ["frescura_datos", "citas_trazabilidad", "aprovecha_ejemplos", "latencia_costo",
             "estilo_formato", "gobernanza_permisos", "esfuerzo_implementacion"]

OPCIONES = ["Prompting", "RAG", "Fine-tuning (LoRA)", "RAG + Fine-tuning"]

# Matriz genérica de puntajes (0 a 5; 5 = la opción satisface muy bien el criterio).
# Orden de las columnas = orden de CRITERIOS.
PUNTAJES = pd.DataFrame(
    [
        [1, 1, 2, 3, 3, 3, 5],   # Prompting
        [5, 5, 2, 2, 3, 5, 3],   # RAG
        [1, 1, 5, 4, 5, 2, 1],   # Fine-tuning (LoRA)
        [5, 5, 5, 3, 5, 3, 1],   # RAG + Fine-tuning
    ],
    index=OPCIONES, columns=CRITERIOS,
)
display(PUNTAJES)

# COMMAND ----------

# TODO 1: completa los pesos de cada escenario. Cada diccionario debe tener los 7 criterios y SUMAR 1.0.
# Pista: reparte 1.0 pensando en qué es crítico (peso alto) y qué es irrelevante (peso bajo, pero no cero).
PESOS = {
    "A - Soporte dPhone (manuales cambian cada semana)": {
        "frescura_datos": None,        # TODO
        "citas_trazabilidad": None,    # TODO
        "aprovecha_ejemplos": None,    # TODO
        "latencia_costo": None,        # TODO
        "estilo_formato": None,        # TODO
        "gobernanza_permisos": None,   # TODO
        "esfuerzo_implementacion": None,  # TODO
    },
    "B - Clasificación de 200.000 tickets/día": {
        "frescura_datos": None,        # TODO
        "citas_trazabilidad": None,    # TODO
        "aprovecha_ejemplos": None,    # TODO
        "latencia_costo": None,        # TODO
        "estilo_formato": None,        # TODO
        "gobernanza_permisos": None,   # TODO
        "esfuerzo_implementacion": None,  # TODO
    },
    "C - Asistente jurídico bancario": {
        "frescura_datos": None,        # TODO
        "citas_trazabilidad": None,    # TODO
        "aprovecha_ejemplos": None,    # TODO
        "latencia_costo": None,        # TODO
        "estilo_formato": None,        # TODO
        "gobernanza_permisos": None,   # TODO
        "esfuerzo_implementacion": None,  # TODO
    },
}

# COMMAND ----------

# Validación de los pesos (no editar): avisa si falta algo o si no suman 1.0.
def validar_pesos(pesos):
    problemas = []
    for escenario, w in pesos.items():
        if any(v is None for v in w.values()):
            problemas.append(f"{escenario}: faltan pesos (hay None).")
        elif abs(sum(w.values()) - 1.0) > 1e-6:
            problemas.append(f"{escenario}: los pesos suman {sum(w.values()):.2f}, deben sumar 1.00.")
    return problemas

problemas = validar_pesos(PESOS)
if problemas:
    print("⚠️ Completa los TODO antes de continuar:")
    for p in problemas:
        print(" -", p)
else:
    print("✅ Pesos válidos en los tres escenarios.")

# COMMAND ----------

# Cálculo del puntaje ponderado por opción (no editar).
def puntaje_ponderado(pesos_escenario):
    vector = pd.Series(pesos_escenario)[CRITERIOS]
    return (PUNTAJES * vector).sum(axis=1).round(2).sort_values(ascending=False)

resultados = {}
if not problemas:
    for escenario, w in PESOS.items():
        resultados[escenario] = puntaje_ponderado(w)
        print(f"\n{escenario}")
        display(resultados[escenario].to_frame("puntaje_ponderado"))
        print("→ Recomendación:", resultados[escenario].index[0])

# COMMAND ----------

# TODO 2: escribe 2 líneas por escenario: por qué ganó la opción y cuál es su riesgo principal.
JUSTIFICACIONES = {
    "A - Soporte dPhone (manuales cambian cada semana)": "TODO: ...",
    "B - Clasificación de 200.000 tickets/día": "TODO: ...",
    "C - Asistente jurídico bancario": "TODO: ...",
}

# COMMAND ----------

# Análisis de sensibilidad (no editar): duplicamos el peso de gobernanza_permisos y renormalizamos.
def sensibilidad(pesos_escenario, criterio="gobernanza_permisos", factor=2.0):
    w = dict(pesos_escenario)
    w[criterio] = w[criterio] * factor
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    return puntaje_ponderado(w).index[0]

if not problemas:
    filas = []
    for escenario, w in PESOS.items():
        base = resultados[escenario].index[0]
        nueva = sensibilidad(w)
        filas.append({"escenario": escenario, "ganador_base": base,
                      "ganador_con_gobernanza_x2": nueva, "cambia": base != nueva})
    display(pd.DataFrame(filas))

# COMMAND ----------

# TODO 3: completa la frase final con tu aprendizaje principal.
LECCION = "La principal lección de la matriz es: TODO"
print(LECCION)

# COMMAND ----------

# Salida esperada:
# - Mientras queden TODO, aparece el aviso "⚠️ Completa los TODO antes de continuar" con los escenarios incompletos.
# - Al completar pesos que suman 1.0: tres tablas de puntaje ponderado (4 opciones, ordenadas de mayor a menor) y una recomendación por escenario.
# - Resultado esperado con pesos razonables: A → RAG; B → Fine-tuning (LoRA) (o RAG + Fine-tuning por poca diferencia); C → RAG + Fine-tuning.
# - Tabla de sensibilidad con la columna "cambia" (True/False) para cada escenario.
# - Al final se imprime tu frase de aprendizaje.
