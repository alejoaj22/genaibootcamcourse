# Ejercicio 1 (Hora 1) — Diagrama de arquitectura RAG para tu reto

**Tiempo:** 15 minutos · **Modalidad:** parejas · **Archivos:** `hora1_starter.py` (esqueleto) y `hora1_solucion.py` (ejemplo resuelto con dPhone)

## Contexto
Ya conoces el ciclo de vida RAG (Ingestión → Extracción → Chunking → Vectorización → Búsqueda → Síntesis) y la arquitectura Medallion. Ahora aplícalo al reto integrador de tu equipo.

## Instrucciones
1. Nombra tu reto y define en una frase la **necesidad de información** que resolverá el asistente.
2. Lista al menos **3 fuentes de datos** (nombre, formato —PDF nativo, PDF escaneado, DOCX, imagen, CSV—, volumen aproximado, quién es el dueño).
3. Para cada fuente decide: ¿entra a un pipeline RAG (no estructurada) o se consulta con SQL (estructurada)?
4. Define qué contendrá cada capa:
   - **Bronze:** columnas y metadatos de trazabilidad.
   - **Silver:** formato de salida del parseo.
   - **Gold:** chunks y metadatos (se completará en la Clase 6).
5. Identifica **al menos 3 riesgos de calidad** (por ejemplo, PDFs escaneados, tablas complejas, documentos duplicados) y una mitigación para cada uno.
6. Completa `hora1_starter.py` para generar tu diagrama Mermaid y tu tabla de riesgos.

## Entregable
- Notebook completado con el diagrama Mermaid (texto) y la tabla de riesgos.
- Capturar el diagrama renderizado (o pegar el texto Mermaid en https://mermaid.live).

## Criterios de revisión (autoevaluación)
- [ ] Hay ≥ 3 fuentes con su formato.
- [ ] Cada fuente está asignada a RAG o SQL con justificación.
- [ ] Las tres capas Medallion están descritas con columnas concretas.
- [ ] Hay ≥ 3 riesgos con mitigación.
- [ ] El diagrama respeta el orden del ciclo RAG.

## Pistas
- Piensa primero en las **preguntas reales** que harán los usuarios; de ellas se derivan las fuentes.
- Un dato puede estar en dos formas (una tabla de precios en CSV y una política en PDF): el asistente puede necesitar ambas.
