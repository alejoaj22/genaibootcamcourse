# Ejercicio Hora 2 — 5 Queries × 3 Modos de Búsqueda: Semántica vs Full-Text vs Híbrida

**Modalidad:** parejas mixtas (perfil de negocio + perfil técnico)  
**Tiempo en clase:** 7 minutos de arranque + finalización autónoma (20–25 minutos)  
**Archivos:** `hora2_starter.py` (esqueleto con `# TODO`) · `hora2_solucion.py` (solución)  
**Resultado de aprendizaje:** RA3 (evaluación cualitativa y selección de estrategia de recuperación en Vector Search).

---

## Contexto de Negocio
DerarTel necesita saber qué estrategia de búsqueda (`similarity_search`, `FULL_TEXT` o `hybrid`) debe ser la predeterminada para el chatbot de soporte y ventas. No todas las preguntas de los clientes son iguales: algunas son conceptuales ("¿resiste caídas al agua?"), otras buscan referencias técnicas precisas ("Helios G1M", "IP69", "512GB") y otras pueden ser consultas fuera de dominio ("¿cuánto cuesta un iPhone 15?").

---

## Tareas

1. **Diseñar 5 consultas de prueba** en inglés (el corpus y el modelo `databricks-gte-large-en` están en inglés):
   - **Q1 (Conceptual/Paráfrasis):** pregunta en lenguaje natural sin usar palabras técnicas textuales.
   - **Q2 (Conceptual/Explicativa):** pregunta sobre capacidades de software o ecosistema.
   - **Q3 (Término exacto / Código):** búsqueda por siglas de procesador o tecnología ("Helios G1M", "6G Ready").
   - **Q4 (Especificación técnica / Ficha):** búsqueda por atributo numérico ("5,500 mAh", "IP69").
   - **Q5 (Consulta trampa / Fuera de dominio):** pregunta sobre algo inexistente en el catálogo de dPhone.

2. **Ejecutar cada consulta en los 3 modos de búsqueda** (`semantic`, `FULL_TEXT`, `hybrid`) con `num_results=3`, recuperando `id`, `path` y `chunk`.

3. **Construir una matriz comparativa** en un DataFrame que registre:
   - `query_id`, `categoria`, `query_text`
   - `modo` (`semantic`, `FULL_TEXT`, `hybrid`)
   - `top1_doc`: nombre del archivo del documento en posición 1
   - `top1_score`: score devuelto por Vector Search
   - `top1_preview`: primeros 100 caracteres del texto recuperado
   - `es_relevante_top1`: booleano (evaluación humana: ¿el chunk responde la intención?)

4. **Calcular la tasa de acierto cualitativa** (`top1_accuracy`) para cada modo (proporción de consultas con `es_relevante_top1 == True`).

5. **Reflexión técnica y de negocio**:
   - Redactar un veredicto de 4–6 líneas respondiendo: ¿Qué modo debe quedar activo por defecto en la función que consumirá el Agente en la Clase 7 y por qué?

---

## Criterios de Evaluación (15 puntos)
| Criterio | Puntos |
|---|---|
| Definición equilibrada de las 5 consultas según tipología | 3 |
| Ejecución correcta de los 3 modos vía `similarity_search` | 4 |
| Construcción de tabla de resultados con scores y previews | 4 |
| Veredicto técnico justificado con la tasa de acierto y tipología | 4 |
