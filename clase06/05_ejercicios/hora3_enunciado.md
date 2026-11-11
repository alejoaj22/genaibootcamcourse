# Ejercicio Hora 3 — Reranking Antes y Después: Evaluación con P@3 y MRR

**Modalidad:** parejas mixtas  
**Tiempo en clase:** 10 minutos de arranque + finalización autónoma (20–25 minutos)  
**Archivos:** `hora3_starter.py` (esqueleto con `# TODO`) · `hora3_solucion.py` (solución)  
**Resultado de aprendizaje:** RA3 (evaluación formal de pipelines de recuperación de 2 etapas y cuantificación del impacto del reranking).

---

## Contexto de Negocio
El equipo de Producto de DerarTel observa que a veces el pasaje clave de respuesta no queda en la posición 1, sino en la posición 3 o 4 del `top-5`. Esto obliga a inyectar más contexto al LLM, aumentando el costo de inferencia y arriesgándose al efecto *Lost in the Middle*. Se plantea incorporar una etapa de **Reranking** (`DatabricksReranker` o `FlashRank`), pero la Dirección Técnica exige cuantificar:
1. ¿Cuánto mejora la posición del primer documento relevante (MRR)?
2. ¿Cuánto mejora la precisión en el top-3 (P@3)?
3. ¿Cuál es el sobrecosto en latencia de inferencia por query?

---

## Tareas

1. **Seleccionar 5 consultas desafiantes** del dominio dPhone donde el orden de los candidatos sea crítico (por ejemplo consultas que mencionan características secundarias como "Halo camera", "titanium chassis", "Vein-Scan", "Satellite SOS", "microkernel").
2. **Recuperar k=10 candidatos iniciales** usando búsqueda vectorial estándar (`similarity_search` base).
3. **Aplicar Reranking** a los 10 candidatos:
   - Configuración A: Sin reranking (Línea base).
   - Configuración B: `DatabricksReranker` (o `FlashRank` si se ejecuta localmente).
4. **Calcular métricas formales**:
   - **Precision@3 (P@3):** fracción de los 3 primeros documentos que son relevantes.
   - **Reciprocal Rank (RR):** inverso de la posición del primer documento relevante ($1/\text{rank}$).
   - **MRR (Mean Reciprocal Rank):** promedio del RR sobre las 5 consultas.
   - **Latencia promedio:** tiempo en milisegundos por consulta en cada configuración.
5. **Generar un reporte comparativo**:
   - Tabla de métricas: Configuración vs P@3 vs MRR vs Latencia (ms).
   - Gráfico de barras de P@3 y MRR antes/después.
   - Decisión de arquitectura: ¿Vale la pena pagar la latencia adicional del reranker para el caso de soporte técnico de dPhone?

---

## Criterios de Evaluación (15 puntos)
| Criterio | Puntos |
|---|---|
| Implementación de recuperación inicial k=10 y reranking k=3 | 4 |
| Cálculo riguroso de P@3 y MRR para ambas configuraciones | 5 |
| Medición y reporte de latencia por etapa | 3 |
| Análisis crítico de balance calidad vs latencia | 3 |
