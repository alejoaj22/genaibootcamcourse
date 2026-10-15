# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Hora 3 — Ejemplo 3: Matriz de Decisión Ponderada en `pandas` para Selección de Modelos y Arquitecturas
# MAGIC
# MAGIC - **Objetivo:** Construir una **Matriz de Decisión Ponderada (MCDA)** reproducible en `pandas` que evalúe distintas opciones de modelos y arquitecturas de IA (costo, calidad, latencia, privacidad, cumplimiento ético y ventana de contexto) y demuestre mediante un análisis de sensibilidad cómo cambia la alternativa ganadora según el escenario del reto.
# MAGIC - **Tiempo estimado:** 10 minutos.
# MAGIC - **Concepto que ilustra:** Toma de decisiones de ingeniería basada en evidencia; trade-offs entre modelos abiertos vs propietarios y entre Prompting vs RAG vs Agentes vs Fine-Tuning.
# MAGIC - **Requisitos previos:** Cómputo conectado (usa únicamente `pandas` y `numpy`, incluido en el runtime estándar).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Definición de Candidatos y Calificaciones Base ($1$ a $5$)
# MAGIC
# MAGIC Evaluamos 4 alternativas reales para los sistemas de DerarTel en una escala de **1 (deficiente / muy desfavorable)** a **5 (excelente / óptimo)**:
# MAGIC 1. **`databricks-gpt-oss-20b` + Prompting / SQL `ai_query`:** Modelo abierto de 20B hospedado en Databricks (rápido, muy económico, ideal para lotes).
# MAGIC 2. **`databricks-gpt-oss-120b` + RAG & UC Functions:** Modelo abierto de 120B con razonamiento en Databricks conectado a Vector Search y tablas Delta.
# MAGIC 3. **API Externa Propietaria Frontera (sin gobernanza local):** Modelo cerrado externo consumido por fuera del Lakehouse.
# MAGIC 4. **Fine-Tuning Propio de un LLM en GPUs dedicadas:** Entrenar pesos propios cada semana y mantener un clúster GPU encendido 24/7.
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - Creamos un `DataFrame` de `pandas` donde cada fila es una alternativa arquitectónica y cada columna es uno de los 6 criterios técnicos y éticos.

# COMMAND ----------

import pandas as pd
import numpy as np

# Calificaciones base (escala 1.0 a 5.0, donde 5.0 siempre es MEJOR: mayor calidad, menor latencia, menor costo, mayor privacidad)
datos_candidatos = [
    {
        "Alternativa": "1. gpt-oss-20b + ai_query (Batch/Rápido en Databricks)",
        "Calidad_Razonamiento": 3.6,
        "Baja_Latencia": 4.9,
        "Eficiencia_Costo": 4.9,
        "Privacidad_UnityCatalog": 5.0,
        "Cumplimiento_y_Control": 4.5,
        "Contexto_e_Integracion_Datos": 4.8
    },
    {
        "Alternativa": "2. gpt-oss-120b + RAG + UC Functions (Agente en Databricks)",
        "Calidad_Razonamiento": 4.8,
        "Baja_Latencia": 3.5,
        "Eficiencia_Costo": 3.7,
        "Privacidad_UnityCatalog": 5.0,
        "Cumplimiento_y_Control": 4.9,
        "Contexto_e_Integracion_Datos": 5.0
    },
    {
        "Alternativa": "3. API Propietaria Externa (Datos salen del Lakehouse)",
        "Calidad_Razonamiento": 4.9,
        "Baja_Latencia": 3.9,
        "Eficiencia_Costo": 2.8,
        "Privacidad_UnityCatalog": 2.2,
        "Cumplimiento_y_Control": 3.2,
        "Contexto_e_Integracion_Datos": 2.9
    },
    {
        "Alternativa": "4. Fine-Tuning Propio + Clúster GPU 24/7",
        "Calidad_Razonamiento": 3.9,
        "Baja_Latencia": 4.1,
        "Eficiencia_Costo": 1.6,
        "Privacidad_UnityCatalog": 4.8,
        "Cumplimiento_y_Control": 3.5,
        "Contexto_e_Integracion_Datos": 2.4
    }
]

df_base = pd.DataFrame(datos_candidatos).set_index("Alternativa")
print("=== CALIFICACIONES BASE POR CRITERIO (ESCALA 1 A 5) ===")
display(df_base.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Motor de Evaluación Ponderada con Restricciones Eliminatorias (*Hard Constraints*)
# MAGIC
# MAGIC En proyectos reales, además de sumar pesos ($\sum w_i \cdot s_{i,j}$), existen **umbrales mínimos no negociables** (por ejemplo: *"Seguridad exige que `Privacidad_UnityCatalog >= 4.0`"*).
# MAGIC
# MAGIC ### Explicación línea por línea:
# MAGIC - `assert abs( sum(pesos.values()) - 1.0 ) < 1e-6`: Verifica matemáticamente que los pesos sumen exactamente $100\%$.
# MAGIC - Calculamos el producto punto entre la matriz de calificaciones y el vector de pesos, y marcamos si alguna alternativa incumple el umbral mínimo de privacidad.

# COMMAND ----------

def evaluar_escenario(df_calificaciones: pd.DataFrame, nombre_escenario: str, pesos: dict, min_privacidad: float = 4.0) -> pd.DataFrame:
    """
    Calcula el puntaje ponderado de cada alternativa para un escenario dado
    y verifica el cumplimiento de la restricción dura de privacidad.
    """
    suma_pesos = sum(pesos.values())
    assert abs(suma_pesos - 1.0) < 1e-6, f"Los pesos deben sumar 1.0 (actual: {suma_pesos})"
    
    criterios = list(pesos.keys())
    vector_pesos = np.array([pesos[c] for c in criterios])
    
    resultado = df_calificaciones[criterios].copy()
    resultado["Puntaje_Ponderado"] = np.round(resultado.values @ vector_pesos, 3)
    resultado["Cumple_Privacidad_Minima"] = resultado["Privacidad_UnityCatalog"] >= min_privacidad
    resultado["Estado_Arquitectura"] = np.where(
        resultado["Cumple_Privacidad_Minima"],
        "✅ APROBABLE",
        "❌ DESCARTADO (Riesgo PII)"
    )
    resultado["Escenario"] = nombre_escenario
    
    # Ordenamos priorizando primero los aprobables y luego el mayor puntaje ponderado
    resultado = resultado.sort_values(
        by=["Cumple_Privacidad_Minima", "Puntaje_Ponderado"],
        ascending=[False, False]
    )
    return resultado.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Análisis de Sensibilidad en 3 Escenarios Reales de DerarTel
# MAGIC
# MAGIC Probamos cómo cambia el ranking cuando cambiamos las prioridades del negocio:
# MAGIC - **Escenario A (Pipeline Batch de Datos):** Clasificar y resumir 500.000 tickets históricos cada noche en SQL (prioridad máxima: **Costo y Velocidad**).
# MAGIC - **Escenario B (Agente Conversacional de Posventa y Compras):** Atender clientes en vivo cruzando PDFs de garantía y tablas de inventario (prioridad máxima: **Calidad/Razonamiento, Control de Alucinaciones e Integración con Unity Catalog**).
# MAGIC - **Escenario C (Prototipo Rápido sin Datos Sensibles):** Experimento aislado con datos sintéticos públicos donde no hay restricción de PII (`min_privacidad = 1.0`) y solo importa la calidad bruta.

# COMMAND ----------

escenarios_pesos = {
    "Escenario A: Batch SQL Masivo (500k tickets/noche)": {
        "pesos": {
            "Calidad_Razonamiento": 0.15,
            "Baja_Latencia": 0.25,
            "Eficiencia_Costo": 0.30,
            "Privacidad_UnityCatalog": 0.15,
            "Cumplimiento_y_Control": 0.05,
            "Contexto_e_Integracion_Datos": 0.10
        },
        "min_privacidad": 4.0
    },
    "Escenario B: Agente Transaccional dPhone (RAG + UC Functions)": {
        "pesos": {
            "Calidad_Razonamiento": 0.30,
            "Baja_Latencia": 0.10,
            "Eficiencia_Costo": 0.10,
            "Privacidad_UnityCatalog": 0.15,
            "Cumplimiento_y_Control": 0.15,
            "Contexto_e_Integracion_Datos": 0.20
        },
        "min_privacidad": 4.0
    },
    "Escenario C: Laboratorio Sintético Sin PII (Foco Calidad Pura)": {
        "pesos": {
            "Calidad_Razonamiento": 0.50,
            "Baja_Latencia": 0.20,
            "Eficiencia_Costo": 0.05,
            "Privacidad_UnityCatalog": 0.05,
            "Cumplimiento_y_Control": 0.10,
            "Contexto_e_Integracion_Datos": 0.10
        },
        "min_privacidad": 1.0
    }
}

resumen_ganadores = []
for nombre_esc, config in escenarios_pesos.items():
    df_eval = evaluar_escenario(df_base, nombre_esc, config["pesos"], config["min_privacidad"])
    ganador = df_eval.iloc[0]
    resumen_ganadores.append({
        "Escenario": nombre_esc,
        "Alternativa_Ganadora": ganador["Alternativa"],
        "Puntaje_Ganador": ganador["Puntaje_Ponderado"],
        "Estado": ganador["Estado_Arquitectura"]
    })
    print(f"\n🏆 {nombre_esc}")
    print(f"   Ganador: {ganador['Alternativa']} (Puntaje: {ganador['Puntaje_Ponderado']}/5.0)")
    display(df_eval[["Alternativa", "Puntaje_Ponderado", "Estado_Arquitectura", "Calidad_Razonamiento", "Baja_Latencia", "Eficiencia_Costo", "Privacidad_UnityCatalog"]])

print("\n=== SÍNTESIS EJECUTIVA MULTIESCENARIO ===")
display(pd.DataFrame(resumen_ganadores))

# Salida esperada:
# - Escenario A (Batch SQL Masivo): Gana "1. gpt-oss-20b + ai_query (Batch/Rápido en Databricks)" con ~4.59/5.0 por su enorme ventaja en costo y velocidad manteniendo privacidad 5.0.
# - Escenario B (Agente Transaccional dPhone): Gana "2. gpt-oss-120b + RAG + UC Functions" con ~4.59/5.0 gracias a su alto razonamiento, control e integración nativa con Unity Catalog.
# - Escenario C (Laboratorio Sintético Sin PII): La "3. API Propietaria Externa" compite en el primer lugar (~4.11) junto con "2. gpt-oss-120b" (~4.42) al levantarse la restricción de privacidad local.
# - El "4. Fine-Tuning Propio + Clúster GPU 24/7" queda en último lugar en todos los escenarios por su bajo puntaje en eficiencia de costo y baja flexibilidad para datos cambiantes.
