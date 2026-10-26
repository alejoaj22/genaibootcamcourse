# Databricks notebook source
# MAGIC %md
# MAGIC # Hora 3 — Ejemplo 1: Validación Robusta con Pydantic y Bucles de Auto-Corrección (Self-Correction Loops)
# MAGIC **Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**
# MAGIC
# MAGIC * **Objetivo:** Implementar un sistema de validación en múltiples capas con **Pydantic v2** (`field_validator`, `model_validator`, restricciones semánticas y de negocio) sobre respuestas generadas por LLMs, y construir un **bucle de auto-corrección (Self-Correction Loop)** con retroalimentación guiada del error (`<validation_errors>`) y auto-reflexión con `databricks-gpt-oss-120b` para reparar fallas sin intervención humana.
# MAGIC * **Tiempo estimado:** 10 minutos.
# MAGIC * **Concepto que ilustra:** Resiliencia en pipelines de software con IA, diferencia entre validez sintáctica (JSON) y validez semántica/de negocio, y arquitectura de reflexión (*Critic -> Reviser*).
# MAGIC * **Requisitos previos:** Clúster o Serverless de Databricks con acceso a `ai_query()` y modelos `databricks-gpt-oss-20b` y `databricks-gpt-oss-120b`.

# COMMAND ----------

# MAGIC %pip install pydantic

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
from typing import Dict, Any, List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

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
               max_tokens: int = 900) -> dict:
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


def parsear_json_limpio(texto: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (texto or "").strip())
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("No se pudo extraer un bloque JSON válido de la respuesta.")


print("Infraestructura de validación inicializada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Modelo de Validación Pydantic v2: Reglas de Negocio Estrictas
# MAGIC El esquema valida no solo tipos, sino coherencia de negocio:
# MAGIC 1. El modelo debe pertenecer a la familia oficial dPhone (`Literal[...]`).
# MAGIC 2. El precio debe ser estrictamente positivo.
# MAGIC 3. El almacenamiento debe estar en la lista de capacidades soportadas (256GB, 512GB, 1TB).
# MAGIC 4. Las citas (`citas_documentos`) deben coincidir con IDs de documentos registrados en el contexto (`DOC-01`, `DOC-02`).
# MAGIC 5. Si `aplica_garantia` es False, el campo `justificacion_rechazo` es obligatorio.

# COMMAND ----------

DOCUMENTOS_REGISTRADOS = ["DOC-01", "DOC-02"]
ALMACENAMIENTOS_PERMITIDOS = ["256GB", "512GB", "1TB"]

class ResolucionGarantia(BaseModel):
    modelo_dispositivo: Literal["dPhone D1", "dPhone D1 Lite", "dPhone D1 Pro", "dPhone D1 Fold"]
    precio_estimado_usd: float = Field(..., gt=0, description="Precio de referencia del equipo")
    almacenamiento: str
    aplica_garantia: bool
    citas_documentos: List[str]
    justificacion_resolucion: str
    justificacion_rechazo: Optional[str] = None

    @field_validator("almacenamiento")
    @classmethod
    def validar_almacenamiento(cls, v: str) -> str:
        if v not in ALMACENAMIENTOS_PERMITIDOS:
            raise ValueError(f"Almacenamiento '{v}' inválido. Valores permitidos: {ALMACENAMIENTOS_PERMITIDOS}")
        return v

    @field_validator("citas_documentos")
    @classmethod
    def validar_citas(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Debe incluir al menos una cita de documento fuente (ej. ['DOC-01']).")
        for cita in v:
            if cita not in DOCUMENTOS_REGISTRADOS:
                raise ValueError(f"Cita '{cita}' desconocida. Fuentes válidas: {DOCUMENTOS_REGISTRADOS}")
        return v

    @model_validator(mode="after")
    def validar_coherencia_rechazo(self):
        if not self.aplica_garantia and not self.justificacion_rechazo:
            raise ValueError("Cuando 'aplica_garantia' es False, 'justificacion_rechazo' no puede ser nulo o vacío.")
        return self


print("Modelo Pydantic v2 compilado con validaciones de negocio.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Implementación del Bucle de Auto-Corrección (*Self-Correction Loop*)
# MAGIC Cuando la respuesta del LLM no pasa las aserciones de Pydantic, en lugar de lanzar una excepción hacia el usuario final, el sistema reinyecta el traceback de validación en un nuevo prompt de corrección guiado.

# COMMAND ----------

def ejecutar_self_correction_loop(
    requerimiento: str,
    max_intentos: int = 3,
    modelo_generador: str = MODELO_20B,
    modelo_revisor: str = MODELO_120B
) -> Dict[str, Any]:
    historial_intentos = []
    
    # Prompt inicial (diseñado intencionalmente débil o abierto para ilustrar la auto-corrección)
    prompt_actual = f"""
    Eres un analista de soporte de garantías DerarTel.
    Evalúa la siguiente solicitud: "{requerimiento}"
    Fuentes oficiales: DOC-01 (Ficha técnica dPhone D1/Lite/Pro/Fold), DOC-02 (Políticas 2028).
    Devuelve un JSON con:
    modelo_dispositivo, precio_estimado_usd, almacenamiento, aplica_garantia (true/false),
    citas_documentos (lista de strings), justificacion_resolucion, justificacion_rechazo.
    """
    
    objeto_validado = None
    ultimo_error = None
    
    for intento in range(1, max_intentos + 1):
        t0 = time.time()
        print(f"\n--- [Intento {intento}/{max_intentos}] Invocando modelo... ---")
        
        # En el intento 1 usamos modelo estándar; si falla, podemos escalar o reinyectar feedback
        modelo_a_usar = modelo_generador if intento == 1 else modelo_revisor
        res_llm = llamar_llm(prompt_actual, modelo=modelo_a_usar, temperature=0.0)
        duracion = round(time.time() - t0, 2)
        
        texto_salida = res_llm["texto"]
        
        # Paso 1: Parseo sintáctico JSON
        try:
            datos_dict = parsear_json_limpio(texto_salida)
            parse_json_ok = True
        except Exception as e_json:
            datos_dict = None
            parse_json_ok = False
            ultimo_error = f"Error sintáctico JSON: {str(e_json)}"
            
        # Paso 2: Validación semántica Pydantic
        if parse_json_ok:
            try:
                objeto_validado = ResolucionGarantia(**datos_dict)
                historial_intentos.append({
                    "intento": intento,
                    "modelo": modelo_a_usar,
                    "latencia_s": duracion,
                    "estado": "EXITO",
                    "error": None
                })
                print(f"ÉXITO en Intento {intento}: Contrato de negocio 100% satisfecho.")
                break
            except ValidationError as e_val:
                ultimo_error = "; ".join([f"{err['loc']}: {err['msg']}" for err in e_val.errors()])
        
        historial_intentos.append({
            "intento": intento,
            "modelo": modelo_a_usar,
            "latencia_s": duracion,
            "estado": "FALLO_VALIDACION",
            "error": ultimo_error
        })
        print(f"FALLO en Intento {intento}: {ultimo_error}")
        
        # Paso 3: Construcción del Prompt de Auto-Corrección (Reflexión)
        prompt_actual = f"""
<instructions>
Tu respuesta anterior falló las validaciones del sistema.
Corrige el objeto JSON para que satisfaga estrictamente el modelo de datos.
</instructions>

<previous_output>
{texto_salida}
</previous_output>

<validation_errors>
{ultimo_error}
</validation_errors>

<rules>
- Devuelve ÚNICAMENTE el objeto JSON corregido.
- Revisa cuidadosamente los valores permitidos para almacenamiento (256GB, 512GB, 1TB) y citas (DOC-01, DOC-02).
- Si aplica_garantia es False, el campo justificacion_rechazo es obligatorio.
</rules>
"""

    return {
        "objeto": objeto_validado,
        "exito": objeto_validado is not None,
        "historial": historial_intentos
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Demostración Práctica con Caso Problemático
# MAGIC Evaluamos un caso donde un cliente reporta daño accidental en un **dPhone D1** con almacenamiento atípico ("128GB" - no existe en catálogo) y donde no aplica garantía por negligencia.

# COMMAND ----------

CASO_GARANTIA_COMPLEJO = (
    "Cliente reporta que sumergió su dPhone D1 (modelo de 128GB según él) "
    "en agua salada y la batería se hinchó. El equipo fue comprado hace 14 meses. "
    "La garantía no cubre inmersión en agua marina."
)

resultado_loop = ejecutar_self_correction_loop(CASO_GARANTIA_COMPLEJO, max_intentos=3)

df_historial = pd.DataFrame(resultado_loop["historial"])
display(spark.createDataFrame(df_historial))

if resultado_loop["exito"]:
    print("\n=== OBJETO PYDANTIC VALIDADO FINAL ===")
    print(resultado_loop["objeto"].model_dump_json(indent=2))

# COMMAND ----------

# Salida esperada:
# 1. Definición robusta de ResolucionGarantia con validadores de campo y modelo.
# 2. El bucle detecta errores en el primer intento (ej. valor de almacenamiento no soportado o falta de citas válidas).
# 3. Reinyecta el mensaje de error de Pydantic en el prompt del segundo intento.
# 4. El modelo 120b corrige los campos y devuelve un objeto 100% válido, registrando el historial completo de telemetría.
