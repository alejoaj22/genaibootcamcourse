# Ejercicio Hora 1 — Golden Set (10+ queries) y 3 `Guidelines` propias de negocio

| Campo | Valor |
|---|---|
| Hora | 1 · Evaluación de GenAI con MLflow |
| Modalidad | Individual o en parejas mixtas (negocio + técnico) |
| Tiempo en clase | 10–15 minutos |
| Archivos | `hora1_starter.py` (esqueleto con `# TODO`) · `hora1_solucion.py` (referencia) · `../datos/eval_dataset_dphone.csv` (apoyo opcional) |

## Objetivo
Construir un conjunto de evaluación robusto (`eval_dataset`) con **al menos 10 consultas** (mezclando casos sin ground truth y casos con `expected_facts` / `expected_response`) y diseñar **3 `Guidelines` propias** que reflejen reglas reales de negocio para evaluar el agente `dphone_agent` mediante `mlflow.genai.evaluate()`.

## Qué debes hacer
1. Abre `hora1_starter.py` y ejecuta las celdas iniciales para cargar `models:/<catálogo>.genai_course.dphone_agent/1` y validar `predict_wrapper(query)`.
2. Completa el `eval_dataset` en el `# TODO 1` hasta tener **mínimo 10 consultas** variadas:
   - Al menos 4 consultas de especificaciones técnicas o catálogo con `expected_facts` o `expected_response`.
   - Al menos 2 consultas sin ground truth (`"expectations": None`) sobre procedimientos o soporte.
   - Al menos 2 consultas sensibles/adversas (intento de pedir descuentos, solicitar datos personales o preguntas fuera de alcance de dPhone).
   *(Tip: puedes apoyarte en `../datos/eval_dataset_dphone.csv`, pero agrega al menos 2 consultas originales creadas por tu equipo).*
3. Define **3 `Guidelines` propias** en el `# TODO 2` usando `model="databricks:/databricks-gpt-oss-120b"` (por ejemplo: claridad/estructura, regla comercial anti-alucinación de precios o política de privacidad y no competencia).
4. En el `# TODO 3`, ejecuta `mlflow.genai.evaluate()` combinando `RelevanceToQuery()`, `RetrievalRelevance()`, `RetrievalGroundedness()`, `Safety()`, `Correctness()`, `RetrievalSufficiency()` y tus 3 `Guidelines`.
5. Inspecciona `eval_results.result_df` en el `# TODO 4` e identifica **al menos 1 consulta** que obtuvo puntuación baja o falló alguna regla, explicando si la causa raíz está en el **Retrieval (datos)** o en la **Generación (prompt/modelo)**.

## Entregable
- Notebook `hora1_starter.py` ejecutado con las métricas agregadas visibles, la tabla `result_df` y un comentario breve de diagnóstico sobre el caso con menor puntaje.

## Criterios de verificación (formativos)
| Criterio | Qué se espera |
|---|---|
| Cobertura del dataset | ≥ 10 consultas con mezcla de hechos verificables, casos abiertos y casos adversos. |
| Calidad de Guidelines | 3 reglas redactadas con criterios observables y verificables (sin ambigüedades). |
| Diagnóstico | Separa correctamente fallos de `Retrieval` vs `Groundedness`/`Correctness` usando los diagramas de la clase. |
