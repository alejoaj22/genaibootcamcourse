# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 2 · Ejemplo 3 — Comparar configuraciones de Model Serving (`workload_size` y `scale_to_zero`)
# MAGIC **Objetivo:** analizar las diferencias técnicas y financieras entre configuraciones de `EndpointCoreConfigInput` (`Small`, `Medium`, `Large`; `scale_to_zero_enabled=True` vs `False`), medir latencia real en llamadas consecutivas al endpoint y recomendar la configuración óptima según el patrón de tráfico.
# MAGIC **Tiempo estimado:** 10 minutos.
# MAGIC **Concepto que ilustra:** toma de decisiones de arquitectura y FinOps para Model Serving (RA2 y RA4).
# MAGIC **Requisitos previos:** acceso a `WorkspaceClient()` (opcionalmente `dphone_endpoint` en estado `READY` para la prueba de latencia en vivo).

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../Includes/Lab_Setup

# COMMAND ----------

import time
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

# 1) Catálogo de configuraciones típicas en proyectos GenAI de Databricks.
# Nota: Los precios por hora/DBU son ilustrativos para comparación relativa FinOps entre escenarios.
perfiles_configuracion = [
    {
        "escenario": "Dev / Bootcamp / Demos puntuales",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
        "horas_activas_dia": 2.0,
        "dbu_por_hora_activa": 4.0,
        "cold_start_esperado_s": "30–90 s",
        "concurrencia_objetivo": "1–4 req/s",
    },
    {
        "escenario": "App interna horario laboral (8am–6pm)",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
        "horas_activas_dia": 24.0,
        "dbu_por_hora_activa": 4.0,
        "cold_start_esperado_s": "0 s (Always-On)",
        "concurrencia_objetivo": "2–6 req/s",
    },
    {
        "escenario": "Soporte clientes alta demanda (SLA estricto)",
        "workload_size": "Medium",
        "scale_to_zero_enabled": False,
        "horas_activas_dia": 24.0,
        "dbu_por_hora_activa": 8.0,
        "cold_start_esperado_s": "0 s (Always-On)",
        "concurrencia_objetivo": "8–20 req/s",
    },
    {
        "escenario": "Campaña masiva / picos críticos",
        "workload_size": "Large",
        "scale_to_zero_enabled": False,
        "horas_activas_dia": 24.0,
        "dbu_por_hora_activa": 16.0,
        "cold_start_esperado_s": "0 s (Always-On)",
        "concurrencia_objetivo": "20–50 req/s",
    },
]

PRECIO_REFERENCIA_DBU_USD = 0.07  # Tarifa de referencia para comparación relativa

filas = []
for p in perfiles_configuracion:
    dbus_mes = p["horas_activas_dia"] * 30 * p["dbu_por_hora_activa"]
    costo_mes_usd = dbus_mes * PRECIO_REFERENCIA_DBU_USD
    filas.append({
        "Escenario": p["escenario"],
        "workload_size": p["workload_size"],
        "scale_to_zero": p["scale_to_zero_enabled"],
        "Cold Start": p["cold_start_esperado_s"],
        "Concurrencia": p["concurrencia_objetivo"],
        "DBUs/mes (est.)": round(dbus_mes, 1),
        "Costo mensual rel. (USD)": round(costo_mes_usd, 2),
    })

df_finops = pd.DataFrame(filas)
display(df_finops)

# COMMAND ----------

# 2) Validamos sintácticamente múltiples objetos EndpointCoreConfigInput sin crear endpoints duplicados.
model_fullname = f"{course.catalog}.{course.schema}.dphone_agent"

for nombre_perfil, size, s2z in [
    ("prototipo_economico", "Small", True),
    ("produccion_estable", "Medium", False),
]:
    cfg = EndpointCoreConfigInput.from_dict({
        "served_models": [
            {
                "model_name": model_fullname,
                "model_version": 1,
                "scale_to_zero_enabled": s2z,
                "workload_size": size,
                "environment_vars": {},
            }
        ],
        # Ejemplo de enrutamiento Canary (100 % en prototipo; podría ser 90/10 con dos modelos)
        "traffic_config": {
            "routes": [
                {"served_model_name": f"dphone_agent-1", "traffic_percentage": 100}
            ]
        },
    })
    print(f"Configuración '{nombre_perfil}' válida -> workload_size={cfg.served_models[0].workload_size}, scale_to_zero={cfg.served_models[0].scale_to_zero_enabled}")

# COMMAND ----------

# 3) Medimos latencia de 3 llamadas consecutivas si 'dphone_endpoint' existe y está READY.
w = WorkspaceClient()
endpoint_name = "dphone_endpoint"
consultas_prueba = [
    "What is the battery capacity of dPhone D1?",
    "List the available dPhone D1 models.",
    "Is the dPhone D1 Pro waterproof?",
]

mediciones = []
try:
    ep = w.serving_endpoints.get(name=endpoint_name)
    ready_str = str(ep.state.ready)
    print(f"Endpoint '{endpoint_name}' encontrado. Estado: {ready_str}")

    if "READY" in ready_str:
        for idx, q in enumerate(consultas_prueba, start=1):
            t0 = time.time()
            _ = w.api_client.do(
                "POST",
                f"/serving-endpoints/{endpoint_name}/invocations",
                body={"input": [{"role": "user", "content": q}]},
            )
            dt = time.time() - t0
            mediciones.append({
                "llamada": idx,
                "nota": "Posible cold start" if idx == 1 else "Endpoint caliente",
                "consulta": q,
                "latencia_s": round(dt, 2),
            })
        display(pd.DataFrame(mediciones))
    else:
        print("El endpoint aún no está READY; omite la medición en vivo por ahora.")
except Exception as e:
    print(f"No se pudo consultar '{endpoint_name}' (puedes ejecutar primero Ejemplo 1): {e}")

# COMMAND ----------

# 4) Selector automático de recomendación según restricciones de negocio.
def recomendar_configuracion_serving(presupuesto_bajo: bool, tolera_cold_start: bool, reqs_por_minuto: int) -> dict:
    scale_to_zero = presupuesto_bajo and tolera_cold_start
    if reqs_por_minuto <= 30:
        workload_size = "Small"
    elif reqs_por_minuto <= 150:
        workload_size = "Medium"
    else:
        workload_size = "Large"
    return {
        "scale_to_zero_enabled": scale_to_zero,
        "workload_size": workload_size,
        "justificacion": (
            f"Con {reqs_por_minuto} req/min se sugiere '{workload_size}'. "
            + ("Se habilita scale-to-zero para ahorrar DBUs." if scale_to_zero else "Se desactiva scale-to-zero para evitar cold start.")
        ),
    }

print("Caso A (Prototipo Bootcamp):", recomendar_configuracion_serving(True, True, 10))
print("Caso B (App Atención Cliente 24/7):", recomendar_configuracion_serving(False, False, 90))

# COMMAND ----------

# Salida esperada:
# - Tabla FinOps comparando 4 perfiles (`Small`/`Medium`/`Large` y `scale_to_zero` True/False) con DBUs y costo relativo mensual.
# - Validación de dos objetos `EndpointCoreConfigInput` con `traffic_config`.
# - Si `dphone_endpoint` está `READY`, tabla con latencia de 3 llamadas consecutivas evidenciando diferencia entre primera llamada y llamadas calientes.
# - Recomendaciones automáticas para dos casos de negocio.
