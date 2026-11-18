# Rúbrica — Ejercicio Integrador Clase 7 (100 puntos)

**Producto evaluado:** notebook del agente dPhone (4+ herramientas de Unity Catalog + RAG) probado con 5+ consultas, con trazas, evaluación inicial y reflexión.

| # | Criterio | Peso |
|---|---|---|
| 1 | Herramientas gobernadas en Unity Catalog | 20 |
| 2 | RAG como herramienta | 10 |
| 3 | System prompt profesional | 15 |
| 4 | Pruebas del agente | 25 |
| 5 | Trazabilidad y evaluación | 10 |
| 6 | Seguridad y gobernanza | 10 |
| 7 | Calidad y comunicación | 10 |
| | **Total** | **100** |

---

## 1. Herramientas gobernadas en Unity Catalog (20 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 18–20 | ≥ 4 funciones UC con `COMMENT`/docstring que dice qué hace, cuándo usarla y qué devuelve; todos los parámetros descritos; ≥ 1 herramienta propia nueva con validación de entradas y casos límite probados; el agente las carga con **lista explícita** (sin `*`). |
| Bueno | 14–17 | ≥ 4 funciones UC con metadatos correctos; 1 herramienta propia; carga con lista explícita o comodín justificado. |
| Básico | 8–13 | 4 funciones registradas pero con descripciones vagas o parámetros sin descripción; sin herramienta propia o sin validación. |
| Insuficiente | 0–7 | Menos de 4 funciones UC, o funciones que no se ejecutan. |

## 2. RAG como herramienta (10 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 9–10 | `VectorSearchRetrieverTool` sobre `dphone_vs_index` con `tool_description` clara (qué documentos, idioma, cuándo usarla); el agente la usa en ≥ 1 consulta y cita la fuente (`path`). |
| Bueno | 7–8 | Herramienta configurada y usada, descripción correcta pero sin indicar idioma o sin citar fuentes. |
| Básico | 4–6 | Herramienta configurada pero con descripción genérica; usada de forma inconsistente. |
| Insuficiente | 0–3 | RAG ausente o no funcional. |

## 3. System prompt profesional (15 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 14–15 | Incluye persona/tono, alcance, reglas de herramientas **verificables**, formato y ejemplo; exige confirmación antes del enlace de compra; prohíbe inventar datos; rechaza fuera de alcance; ignora instrucciones maliciosas. |
| Bueno | 11–13 | Cuatro de los cinco bloques y al menos tres de las cuatro reglas de seguridad. |
| Básico | 6–10 | Prompt genérico ("sé amable") con pocas reglas verificables. |
| Insuficiente | 0–5 | Prompt vacío o contradictorio. |

## 4. Pruebas del agente (25 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 22–25 | ≥ 5 consultas (ideal 7) con herramientas esperadas definidas **antes** de ejecutar; incluye RAG, cadena de ≥ 2 herramientas, acción con confirmación y caso adverso; tabla con resultado y análisis de las fallas; ≥ 80 % de aciertos en herramientas esperadas. |
| Bueno | 17–21 | ≥ 5 consultas con herramientas esperadas; incluye al menos 3 de los 4 tipos; análisis breve. |
| Básico | 10–16 | 3–4 consultas o sin herramientas esperadas definidas; sin caso adverso. |
| Insuficiente | 0–9 | Menos de 3 consultas o sin evidencia de ejecución. |

## 5. Trazabilidad y evaluación (10 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 9–10 | ≥ 2 trazas analizadas (qué spans, cuántas llamadas al LLM, tiempos) y `mlflow.genai.evaluate` con un scorer predefinido y el scorer propio; interpreta resultados con criterio (límites del LLM juez). |
| Bueno | 7–8 | Trazas y evaluación presentes; interpretación superficial. |
| Básico | 4–6 | Solo trazas o solo evaluación. |
| Insuficiente | 0–3 | Sin evidencia de trazas ni de evaluación. |

## 6. Seguridad y gobernanza (10 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 9–10 | Reflexión de 8–10 líneas que cubre: permisos `EXECUTE`/menor privilegio, riesgo de inyección de prompt, exceso de agencia (acción de compra), límite `max_iter`, confirmación humana y evaluación continua; propone al menos una mejora concreta. |
| Bueno | 7–8 | Cubre cuatro de los puntos con ejemplos del propio agente. |
| Básico | 4–6 | Menciona seguridad de forma general sin conexión con el agente. |
| Insuficiente | 0–3 | Sin reflexión o incorrecta (por ejemplo, "el prompt basta para la seguridad"). |

## 7. Calidad y comunicación (10 puntos)

| Nivel | Puntos | Descriptor |
|---|---|---|
| Excelente | 9–10 | Notebook ejecutable de arriba hacia abajo sin editar nada, comentarios en español claros, celdas ordenadas, salida esperada documentada, bitácora de iteración con cambio y efecto. |
| Bueno | 7–8 | Se ejecuta con ajustes mínimos; comentarios correctos; bitácora presente. |
| Básico | 4–6 | Requiere correcciones para ejecutarse; poca documentación. |
| Insuficiente | 0–3 | No se ejecuta o no se entiende. |

---

## Penalizaciones y bonificaciones
- **−5** si hay secretos, tokens o datos personales reales dentro del código o del prompt.
- **−5** si usa el comodín `*` sin justificarlo en el informe.
- **−10** si el notebook entregado no tiene salidas visibles (no hay evidencia de ejecución).
- **+3** (máximo, sin exceder 100) por la extensión opcional: herramienta de escalamiento con regla de prompt, o registro del agente con `resources` correctos.

## Conversión de nota
| Puntaje | Calificación |
|---|---|
| 90–100 | Sobresaliente |
| 75–89 | Notable |
| 60–74 | Aprobado |
| < 60 | Requiere reentrega |

## Lista de verificación rápida para el estudiante
- [ ] ≥ 4 funciones UC con `COMMENT` y parámetros descritos
- [ ] ≥ 1 herramienta propia nueva con validación
- [ ] Lista explícita de funciones (sin `*`)
- [ ] `VectorSearchRetrieverTool` con `tool_description`
- [ ] System prompt con 5 bloques y reglas de seguridad
- [ ] ≥ 5 consultas con herramientas esperadas (RAG, cadena, acción, adversa)
- [ ] ≥ 2 trazas revisadas
- [ ] `mlflow.genai.evaluate` ejecutado
- [ ] Reflexión de seguridad (8–10 líneas)
- [ ] Notebook ejecutado con salidas visibles
