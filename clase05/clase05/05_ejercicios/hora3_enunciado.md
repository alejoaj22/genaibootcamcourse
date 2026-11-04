# Ejercicio 3 (Hora 3) — Parsear y analizar la calidad

**Tiempo:** 15 minutos (puede terminarse como tarea) · **Modalidad:** individual · **Archivos:** `hora3_starter.py`, `hora3_solucion.py`

## Contexto
`ai_parse_document()` puede fallar o producir resultados pobres (PDFs escaneados, tablas complejas, archivos corruptos). Un pipeline profesional mide la calidad y decide qué hacer con cada documento.

## Instrucciones
1. Toma de 2 a 5 documentos de tu tabla `bronze_reto` (o de `bronze_docs_raw`). **Usa `LIMIT`** para controlar costos.
2. Parsea con `ai_parse_document(content, map('version','2.0'))`.
3. Calcula por documento:
   - `error_status` (texto), `n_paginas`, `n_elementos`.
   - `n_caracteres` del texto extraído (concatenación de los contenidos de los elementos).
   - `chars_por_pagina`.
4. Clasifica cada documento en un `estado_calidad`:
   - `ERROR` si hay `error_status`,
   - `VACIO` si hay menos de 50 caracteres,
   - `SOSPECHOSO` si hay menos de 200 caracteres por página (posible escaneado o fallo parcial),
   - `OK` en otro caso.
5. Define una `accion_fallback` por estado: `OK → continuar a Silver`, `SOSPECHOSO → revisión visual`, `VACIO → reintentar/OCR alternativo`, `ERROR → cuarentena`.
6. Guarda el resultado en la tabla `silver_reto_calidad`.
7. En una celda markdown, redacta 3-5 líneas con tus conclusiones: ¿qué documento es el más problemático y qué harías?

## Criterios de revisión
- [ ] Se usó `LIMIT` antes de parsear.
- [ ] Las métricas se calculan de forma reproducible (SQL/PySpark).
- [ ] Los estados de calidad siguen las reglas.
- [ ] Hay una acción de fallback por estado.
- [ ] Las conclusiones son específicas, basadas en las métricas.

## Pistas
- Navega el VARIANT con `parsed_content:document:elements`.
- Usa `transform(...)` y `array_join(...)` para concatenar el contenido de los elementos.
- Si un documento falla, el resto debe procesarse igualmente (no uses `collect()` para todo el lote).
