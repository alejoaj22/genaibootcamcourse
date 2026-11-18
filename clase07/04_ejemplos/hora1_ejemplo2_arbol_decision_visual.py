# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 1 · Ejemplo 2 — Árbol de decisión visual: Prompting vs RAG vs Fine-tuning (y calculadora de cuantización)
# MAGIC **Objetivo:** dibujar el árbol de decisión empresarial, resaltar el camino que sigue cada caso de uso y estimar el impacto de la cuantización en memoria y GPUs.
# MAGIC **Tiempo estimado:** 10 minutos
# MAGIC **Concepto que ilustra:** decisión analítica (NO se entrena nada): prompting → RAG → fine-tuning (SFT/LoRA/QLoRA) y cuantización (FP16, INT8, INT4).
# MAGIC **Requisitos previos:** ninguno (Python puro, pandas y matplotlib, preinstalados en Databricks). No usa modelos ni tablas.
# MAGIC
# MAGIC > Las cifras de memoria son **aproximaciones** (parámetros × bytes por parámetro + un margen). La pérdida de calidad por cuantización depende del modelo y la tarea: siempre se mide con un conjunto de prueba propio.

# COMMAND ----------

# 1) Reglas del árbol. Cada caso se describe con cinco preguntas de sí/no.
import math
import pandas as pd

def recomendar(caso):
    """Devuelve (recomendación, camino) donde camino es la lista de nodos visitados."""
    camino = ["Q1"]
    if caso["basta_prompt"]:                         # Q1: ¿un buen prompt con ejemplos alcanza?
        camino.append("L_PROMPT")
        return "Prompting (con ejemplos)", camino

    camino.append("Q2")
    externo = caso["necesita_conocimiento_externo"]  # Q2: ¿falta conocimiento externo, cambiante o citable?
    comportamiento = caso["necesita_comportamiento"] and caso["tiene_datos_etiquetados"]

    if externo:
        camino.append("Q3A")                         # Q3A: ¿además hay que cambiar el comportamiento con muchos ejemplos?
        if comportamiento:
            camino.append("L_RAGFT")
            return "RAG + Fine-tuning ligero (LoRA)", camino
        camino.append("L_RAG")
        return "RAG", camino

    camino.append("Q3B")                             # Q3B: sin conocimiento externo, ¿hay comportamiento + datos?
    if comportamiento:
        camino.append("L_FT")
        return "Fine-tuning (LoRA/QLoRA)", camino
    camino.append("L_AVANZADO")
    return "Prompting avanzado + recolectar datos y reevaluar", camino

# COMMAND ----------

# 2) Cinco casos de ejemplo (negocio real, no solo técnicos).
CASOS = {
    "1. Soporte con manuales que cambian cada semana": dict(
        basta_prompt=False, necesita_conocimiento_externo=True,
        necesita_comportamiento=False, tiene_datos_etiquetados=False),
    "2. Clasificar el tono de 200.000 tickets/día (50.000 etiquetados)": dict(
        basta_prompt=False, necesita_conocimiento_externo=False,
        necesita_comportamiento=True, tiene_datos_etiquetados=True),
    "3. Redactar correos con el tono corporativo (guía de 1 página)": dict(
        basta_prompt=True, necesita_conocimiento_externo=False,
        necesita_comportamiento=False, tiene_datos_etiquetados=False),
    "4. Asistente legal: citar normas vigentes + formato jurídico estricto (5.000 ejemplos)": dict(
        basta_prompt=False, necesita_conocimiento_externo=True,
        necesita_comportamiento=True, tiene_datos_etiquetados=True),
    "5. Chatbot interno nuevo, sin documentos ni ejemplos aún": dict(
        basta_prompt=False, necesita_conocimiento_externo=False,
        necesita_comportamiento=False, tiene_datos_etiquetados=False),
}

filas = []
for nombre, caso in CASOS.items():
    rec, camino = recomendar(caso)
    filas.append({"caso": nombre, "recomendación": rec, "camino": " → ".join(camino)})
display(pd.DataFrame(filas))

# COMMAND ----------

# 3) Dibujo del árbol con matplotlib. Si matplotlib no está disponible, se imprime una versión en texto.
POS = {   # nodo: (x, y, texto, tipo)
    "Q1": (6.5, 9.0, "¿Un buen prompt con ejemplos\nresuelve la tarea?", "q"),
    "L_PROMPT": (2.0, 7.0, "PROMPTING\n(empieza siempre aquí)", "l"),
    "Q2": (9.0, 7.0, "¿Falta conocimiento externo,\ncambiante o que haya que citar?", "q"),
    "Q3A": (5.5, 4.6, "¿Además hay que cambiar estilo/formato\ncon muchos ejemplos etiquetados?", "q"),
    "Q3B": (11.5, 4.6, "¿Necesita comportamiento específico\ncon muchos ejemplos etiquetados?", "q"),
    "L_RAGFT": (3.5, 2.0, "RAG + FINE-TUNING\nLIGERO (LoRA)", "l"),
    "L_RAG": (7.5, 2.0, "RAG", "l"),
    "L_FT": (10.0, 2.0, "FINE-TUNING\n(LoRA / QLoRA)", "l"),
    "L_AVANZADO": (13.0, 2.0, "PROMPTING AVANZADO\n+ recolectar datos", "l"),
}
ARISTAS = [("Q1", "L_PROMPT", "Sí"), ("Q1", "Q2", "No"),
           ("Q2", "Q3A", "Sí"), ("Q2", "Q3B", "No"),
           ("Q3A", "L_RAGFT", "Sí"), ("Q3A", "L_RAG", "No"),
           ("Q3B", "L_FT", "Sí"), ("Q3B", "L_AVANZADO", "No")]

def dibujar_arbol(camino=None, titulo="Árbol de decisión: Prompting vs RAG vs Fine-tuning"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    camino = camino or []
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15); ax.set_ylim(0.8, 10); ax.axis("off"); ax.set_title(titulo, fontsize=14, fontweight="bold")

    # Aristas primero (para que queden detrás de las cajas)
    for a, b, etiqueta in ARISTAS:
        xa, ya = POS[a][0], POS[a][1]; xb, yb = POS[b][0], POS[b][1]
        activa = (a in camino) and (b in camino)
        ax.annotate("", xy=(xb, yb + 0.55), xytext=(xa, ya - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=3.0 if activa else 1.2,
                                    color="#d62728" if activa else "#999999"))
        ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.1, etiqueta, fontsize=10,
                color="#d62728" if activa else "#555555", fontweight="bold")

    # Nodos
    for nodo, (x, y, texto, tipo) in POS.items():
        visitado = nodo in camino
        if tipo == "q":
            color = "#ffd8d8" if visitado else "#dbe9ff"
        else:
            color = "#ffb3b3" if visitado else "#d8f2d8"
        caja = FancyBboxPatch((x - 1.55, y - 0.55), 3.1, 1.1, boxstyle="round,pad=0.05",
                              fc=color, ec="#d62728" if visitado else "#444444", lw=2.5 if visitado else 1)
        ax.add_patch(caja)
        ax.text(x, y, texto, ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()

def arbol_en_texto():
    print("Q1 ¿Un buen prompt con ejemplos resuelve la tarea?")
    print("   Sí → PROMPTING")
    print("   No → Q2 ¿Falta conocimiento externo/cambiante/citable?")
    print("        Sí → Q3A ¿También estilo/formato con muchos ejemplos?  Sí → RAG + FT ligero · No → RAG")
    print("        No → Q3B ¿Comportamiento específico con muchos ejemplos? Sí → FINE-TUNING · No → PROMPTING AVANZADO + datos")

try:
    dibujar_arbol()
except Exception as e:
    print("No se pudo dibujar con matplotlib:", e)
    arbol_en_texto()

# COMMAND ----------

# 4) Resaltamos el camino de cada caso de ejemplo.
for nombre, caso in CASOS.items():
    rec, camino = recomendar(caso)
    try:
        dibujar_arbol(camino, titulo=f"{nombre}\n→ {rec}")
    except Exception as e:
        print(f"{nombre} → {rec} (camino: {' → '.join(camino)})")

# COMMAND ----------

# 5) Calculadora de memoria por tamaño de modelo y precisión (aproximada).
BITS = {"FP16": 16, "INT8": 8, "INT4": 4}
TAMANOS_B = [7, 13, 70]          # miles de millones de parámetros
MARGEN = 0.20                    # margen para caché KV, activaciones y sobrecarga (20 %)
GPU_GB = 80                      # memoria de una GPU de 80 GB (ej. clase A100/H100 de 80 GB)

def memoria_pesos_gb(parametros_b, bits):
    """Memoria solo de los pesos: parámetros × bytes por parámetro (1B params × 1 byte ≈ 1 GB)."""
    return parametros_b * bits / 8

filas = []
for t in TAMANOS_B:
    for nombre_prec, bits in BITS.items():
        pesos = memoria_pesos_gb(t, bits)
        total = pesos * (1 + MARGEN)
        filas.append({"modelo": f"{t}B", "precisión": nombre_prec, "pesos_GB": round(pesos, 1),
                      "total_con_margen_GB": round(total, 1), "GPUs_80GB_minimas": math.ceil(total / GPU_GB)})
tabla_mem = pd.DataFrame(filas)
display(tabla_mem)

# COMMAND ----------

# 6) Gráfico de barras agrupadas: memoria de pesos por tamaño y precisión.
try:
    import matplotlib.pyplot as plt
    pivote = tabla_mem.pivot(index="modelo", columns="precisión", values="pesos_GB")[["FP16", "INT8", "INT4"]]
    pivote = pivote.loc[[f"{t}B" for t in TAMANOS_B]]
    ax = pivote.plot(kind="bar", figsize=(8, 5), color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Memoria de pesos (GB)")
    ax.set_title("Impacto de la cuantización en memoria (aproximado)")
    for contenedor in ax.containers:
        ax.bar_label(contenedor, fmt="%.1f", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("No se pudo graficar:", e)

# COMMAND ----------

# 7) Caso de discusión: un modelo de 70B en FP16 vs INT4. ¿Qué cambia para el negocio?
fp16 = tabla_mem[(tabla_mem["modelo"] == "70B") & (tabla_mem["precisión"] == "FP16")].iloc[0]
int4 = tabla_mem[(tabla_mem["modelo"] == "70B") & (tabla_mem["precisión"] == "INT4")].iloc[0]
print(f"70B en FP16: ~{fp16['total_con_margen_GB']} GB → mínimo {fp16['GPUs_80GB_minimas']} GPUs de 80 GB")
print(f"70B en INT4: ~{int4['total_con_margen_GB']} GB → mínimo {int4['GPUs_80GB_minimas']} GPU(s) de 80 GB")
print("Preguntas para decidir: ¿cuánta calidad perdemos en NUESTRO golden set? ¿El ahorro de GPUs lo compensa?")
print("¿Un modelo más pequeño con LoRA ya resuelve la tarea con menos costo?")

# COMMAND ----------

# 8) Tabla de apoyo: benchmarks públicos y para qué decisión sirven (solo lectura, sin ejecutar nada).
benchmarks = pd.DataFrame([
    {"benchmark": "MMLU", "mide": "Conocimiento general en 57 áreas", "formato": "Opción múltiple",
     "límite": "Saturado y posiblemente contaminado; no mide su dominio"},
    {"benchmark": "HellaSwag", "mide": "Sentido común (completar frases)", "formato": "Opción múltiple",
     "límite": "Los modelos recientes lo saturan"},
    {"benchmark": "HumanEval", "mide": "Generación de código Python", "formato": "Pruebas unitarias (pass@k)",
     "límite": "Solo código; no sirve para atención al cliente"},
    {"benchmark": "MT-Bench", "mide": "Conversación de varios turnos", "formato": "LLM como juez",
     "límite": "El juez tiene sesgos (longitud, estilo)"},
    {"benchmark": "Chatbot Arena", "mide": "Preferencia humana en general", "formato": "Votos a ciegas (Elo)",
     "límite": "Preferencia general, no su tarea ni su idioma"},
])
display(benchmarks)

# COMMAND ----------

# Salida esperada:
# - Tabla de 5 casos: 1 → RAG · 2 → Fine-tuning (LoRA/QLoRA) · 3 → Prompting (con ejemplos) · 4 → RAG + Fine-tuning ligero (LoRA) · 5 → Prompting avanzado + recolectar datos y reevaluar.
# - Una figura del árbol base y cinco figuras con el camino del caso resaltado en rojo (o el árbol en texto si falla matplotlib).
# - Tabla de memoria: 7B → 14 / 7 / 3,5 GB de pesos; 13B → 26 / 13 / 6,5 GB; 70B → 140 / 70 / 35 GB (FP16 / INT8 / INT4).
# - GPUs de 80 GB mínimas con margen del 20 %: 70B FP16 → 3; 70B INT4 → 1.
# - Gráfico de barras agrupadas y tabla descriptiva de benchmarks.
