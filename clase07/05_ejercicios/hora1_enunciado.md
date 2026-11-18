# Ejercicio Hora 1 — Matriz de decisión: Prompting vs RAG vs Fine-tuning

| Campo | Valor |
|---|---|
| Hora | 1 · De asistentes a agentes y toma de decisiones |
| Modalidad | En parejas mixtas (una persona de negocio + una técnica) |
| Tiempo en clase | 8 minutos (+ debrief al inicio de la Hora 2) |
| Archivos | `hora1_starter.py` (esqueleto) · `hora1_solucion.py` (referencia, se entrega después de la clase) |
| Tipo | Analítico: **no se ejecuta ningún fine-tuning ni cuantización** |

## Objetivo
Aplicar el árbol de decisión de la clase para recomendar, con argumentos, la estrategia más adecuada (Prompting, RAG, Fine-tuning o RAG + Fine-tuning ligero) en tres escenarios empresariales, usando una matriz de decisión ponderada.

## Contexto
La matriz compara **4 opciones** con **7 criterios**. Los puntajes de las opciones ya vienen dados en el starter (matriz genérica de 0 a 5, donde **5 = la opción satisface muy bien el criterio**). Su trabajo es decidir **cuánto pesa cada criterio en cada escenario** y justificar la recomendación.

### Criterios
| # | Criterio | ¿Qué mide? (5 = mejor) |
|---|---|---|
| 1 | `frescura_datos` | Mantiene información actualizada sin reentrenar. |
| 2 | `citas_trazabilidad` | Permite citar la fuente y auditar la respuesta. |
| 3 | `aprovecha_ejemplos` | Saca partido de ejemplos etiquetados disponibles. |
| 4 | `latencia_costo` | Baja latencia y bajo costo por consulta a gran volumen. |
| 5 | `estilo_formato` | Controla de forma estable el estilo y el formato de salida. |
| 6 | `gobernanza_permisos` | Respeta permisos por usuario y permite borrar/actualizar datos. |
| 7 | `esfuerzo_implementacion` | Poco esfuerzo y poco tiempo para implementarlo. |

### Escenarios
**A — Soporte dPhone.** Un asistente responde 40.000 consultas al mes sobre manuales, garantías y precios que **cambian cada semana**. Las respuestas deben **citar el documento** y respetar que algunas tarifas solo las ven los distribuidores. No hay ejemplos etiquetados todavía.

**B — Clasificación de tickets.** Una empresa recibe **200.000 tickets al día** y debe clasificar tono y urgencia, devolviendo siempre un **JSON con formato fijo**, con latencia menor a 300 ms y costo mínimo por ticket. Tiene **50.000 tickets etiquetados** por agentes humanos. El conocimiento externo casi no cambia.

**C — Asistente jurídico bancario.** Un banco quiere un asistente que redacte conceptos jurídicos con **formato estricto**, **citando normas vigentes** (se actualizan cada mes). Tiene **5.000 documentos revisados por abogados** como ejemplos y maneja datos sensibles de clientes.

## Qué debe hacer
1. Abra `hora1_starter.py`. Verá la matriz de puntajes por opción ya cargada.
2. Para **cada escenario**, complete los **pesos** de los 7 criterios (decimales que **sumen exactamente 1,0**). Pregúntese: ¿qué criterios son críticos y cuáles irrelevantes en este escenario?
3. Ejecute la celda de cálculo: obtendrá el puntaje ponderado de cada opción y la recomendación.
4. Escriba en `JUSTIFICACIONES` **2 líneas por escenario**: por qué ganó la opción y qué riesgo principal tiene.
5. Revise la celda de sensibilidad: ¿cambia la recomendación si el peso de `gobernanza_permisos` se duplica?
6. (Opcional) Ajuste **dos** puntajes de la matriz con los que no esté de acuerdo y explique el porqué.

## Entregable
- El starter completo con los pesos, las justificaciones y la tabla de resultados.
- Una frase final: *"La principal lección de la matriz es …"*.

## Criterios de evaluación (formativos)
| Criterio | Qué se espera |
|---|---|
| Coherencia | Los pesos reflejan las necesidades de cada escenario (no son iguales en los tres). |
| Argumentación | La justificación menciona al menos dos criterios con mayor peso. |
| Matemática | Los pesos suman 1,0 y la recomendación sale del puntaje, no de la intuición. |
| Pensamiento crítico | Identifica un riesgo de la opción ganadora y qué la haría cambiar. |

## Pistas
- Si la información cambia cada semana, ¿qué opción obliga a reentrenar?
- Si hay que **citar** la fuente, ¿qué opción lo permite de forma natural?
- A 200.000 consultas/día, ¿qué pesa más: la frescura de los datos o el costo por consulta?
- Fine-tuning y RAG **no son excluyentes**: la cuarta opción combina ambos.
