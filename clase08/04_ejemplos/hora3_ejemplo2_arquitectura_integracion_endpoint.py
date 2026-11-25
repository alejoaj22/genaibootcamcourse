# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 · Ejemplo 2 — Arquitectura de integración: cómo la aplicación consume el Endpoint
# MAGIC **Objetivo:** implementar y probar la capa cliente desacoplada que usará la aplicación de la Clase 9 (Databricks App / Streamlit / backend Python) para consumir `dphone_endpoint` con contrato JSON estable, manejo de timeouts (cold start) y respaldo elegante ante errores.
# MAGIC **Tiempo estimado:** 10 minutos.
# MAGIC **Concepto que ilustra:** separación de responsabilidades UI ↔ Model Serving Endpoint y diseño del contrato de integración (RA5).
# MAGIC **Requisitos previos:** opcionalmente `dphone_endpoint` en estado `READY`; si no existe aún, el ejemplo activa un modo de simulación de contrato para validar la arquitectura sin bloqueos.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import uuid
from dataclasses import dataclass, asdict
from databricks.sdk import WorkspaceClient

# 1) Definimos el Contrato de Respuesta de la Aplicación (independiente de los detalles internos de Databricks).
@dataclass
class RespuestaAppContrato:
    request_id: str
    status: str           # "ok" | "error" | "timeout"
    answer_text: str
    latency_ms: int
    source_mode: str      # "serving_endpoint" | "mock_contract"
    error_detail: str = ""

def extraer_texto_responses_agent(payload: dict) -> str:
    """Extrae de forma segura el texto final desde el formato estándar ResponsesAgent."""
    if not isinstance(payload, dict):
        return str(payload)
    for item in reversed(payload.get("output", [])):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", "")
    # Respaldo si el endpoint responde bajo formato predictions o choices
    if "predictions" in payload:
        return str(payload["predictions"])
    return json.dumps(payload, ensure_ascii=False)

# COMMAND ----------

# 2) Cliente reutilizable que la UI (Streamlit / FastAPI / Databricks App) importará en la Clase 9.
class DPhoneEndpointClient:
    def __init__(self, endpoint_name: str = "dphone_endpoint", timeout_s: int = 60, permitir_mock_si_no_existe: bool = True):
        self.endpoint_name = endpoint_name
        self.timeout_s = timeout_s
        self.permitir_mock = permitir_mock_si_no_existe
        self.workspace_client = WorkspaceClient()

    def consultar(self, pregunta_usuario: str, session_id: str = "ui-session-01") -> RespuestaAppContrato:
        req_id = str(uuid.uuid4())[:8]
        t0 = time.time()

        # Validación básica de entrada antes de gastar una llamada al endpoint
        if not pregunta_usuario or not pregunta_usuario.strip():
            return RespuestaAppContrato(
                request_id=req_id,
                status="error",
                answer_text="Por favor escribe una consulta válida.",
                latency_ms=0,
                source_mode="client_validation",
                error_detail="Empty query",
            )

        body = {
            "input": [{"role": "user", "content": pregunta_usuario.strip()}],
            "custom_inputs": {"session_id": session_id},
        }

        try:
            # Verificamos si el endpoint está disponible en el workspace
            ep = self.workspace_client.serving_endpoints.get(name=self.endpoint_name)
            if "READY" not in str(ep.state.ready):
                raise RuntimeError(f"Endpoint '{self.endpoint_name}' en estado {ep.state.ready}")

            raw_response = self.workspace_client.api_client.do(
                "POST",
                f"/serving-endpoints/{self.endpoint_name}/invocations",
                body=body,
            )
            texto = extraer_texto_responses_agent(raw_response)
            return RespuestaAppContrato(
                request_id=req_id,
                status="ok",
                answer_text=texto,
                latency_ms=int((time.time() - t0) * 1000),
                source_mode="serving_endpoint",
            )

        except Exception as err:
            if self.permitir_mock:
                # Permite al desarrollador Frontend avanzar con el contrato JSON aunque el endpoint esté apagado
                return RespuestaAppContrato(
                    request_id=req_id,
                    status="ok",
                    answer_text=f"[SIMULACIÓN CONTRATO] Respuesta estructurada para: '{pregunta_usuario}'. (Motivo fallback: {err})",
                    latency_ms=int((time.time() - t0) * 1000),
                    source_mode="mock_contract",
                    error_detail=str(err),
                )
            return RespuestaAppContrato(
                request_id=req_id,
                status="error",
                answer_text="El asistente no está disponible temporalmente. Intenta de nuevo en unos segundos.",
                latency_ms=int((time.time() - t0) * 1000),
                source_mode="serving_endpoint",
                error_detail=str(err),
            )

# COMMAND ----------

# 3) Probamos el cliente con 3 casos: consulta válida, consulta vacía y formato REST equivalente para apps externas.
cliente = DPhoneEndpointClient(endpoint_name="dphone_endpoint", permitir_mock_si_no_existe=True)

for q in ["What models are available for the dPhone D1?", ""]:
    res = cliente.consultar(q, session_id="demo-clase08")
    print(json.dumps(asdict(res), indent=2, ensure_ascii=False))

# COMMAND ----------

# 4) Imprimimos el comando cURL equivalente (para apps externas en React/Node/FastAPI fuera de Databricks).
host = cliente.workspace_client.config.host or "https://<tu-workspace>.cloud.databricks.com"
curl_ejemplo = f"""
curl -X POST "{host}/serving-endpoints/dphone_endpoint/invocations" \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"input": [{{"role": "user", "content": "What models are available for the dPhone D1?"}}]}}'
"""
print("Ejemplo de consumo HTTP estándar desde cualquier backend externo:\n", curl_ejemplo)

# COMMAND ----------

# Salida esperada:
# - Dos objetos JSON cumpliendo exactamente el contrato `RespuestaAppContrato` (`request_id`, `status`, `answer_text`, `latency_ms`, `source_mode`).
# - La consulta vacía es rechazada localmente sin gastar DBU ni llamar al endpoint.
# - Si `dphone_endpoint` está `READY`, `source_mode` será `"serving_endpoint"`; si no está disponible, cae ordenadamente a `"mock_contract"`.
# - Comando `curl` listo para documentar la integración con frontend o backend externo.
