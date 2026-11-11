# Ejercicio Hora 1 — Tres configuraciones de chunking: medir antes de decidir

**Modalidad:** parejas mixtas (perfil de negocio + perfil técnico)
**Tiempo en clase:** 5 minutos de arranque + finalización en casa (20–30 minutos)
**Archivos:** `hora1_starter.py` (esqueleto con `# TODO`) · `hora1_solucion.py` (solución)
**Resultado de aprendizaje:** RA3 (diseño del componente de recuperación) y RA1 (relación tokens/caracteres).

## Contexto
DerarTel quiere un asistente que responda preguntas sobre sus teléfonos dPhone. El equipo de datos ya tiene la tabla `silver_docs_parsed`. Antes de indexar, deben decidir **qué `chunk_size` y `chunk_overlap` usar**. Una mala elección hará que el asistente responda con fragmentos incompletos o con ruido.

## Tareas
1. **Definir tres configuraciones** de `(chunk_size, chunk_overlap)`. Sugeridas: `(500, 50)`, `(1500, 150)` y `(3000, 300)`. Pueden proponer otras, pero deben ser distintas entre sí.
2. **Aplicarlas a toda la tabla Silver** usando una `pandas_udf` (como en el notebook 1.1C), sin escribir tablas nuevas.
3. **Calcular por configuración**:
   - `n_chunks` (total de chunks),
   - `avg_len` y `p90_len` (longitud promedio y percentil 90 en caracteres),
   - `pct_cortos` (% de chunks con menos de 100 caracteres),
   - `factor_duplicacion` (suma de longitudes de chunks / suma de longitudes de los textos originales),
   - `pct_fin_frase` (% de chunks que terminan en `.`, `!`, `?` o `|`; proxy de "no cortar a mitad de idea"),
   - `tokens_max_aprox` (longitud máxima / 4).
4. **Prueba de integridad de hechos clave**: el perfil de negocio elige 3 hechos que el asistente debe poder responder (sugeridos: `IP69`, `5,500`, `Helios`). Para cada configuración, verificar si existe al menos un chunk que contenga el hecho **y** la palabra `dPhone`.
5. **Recomendación**: escribir en la última celda (3–5 líneas) qué configuración recomiendan y por qué, citando al menos dos métricas.

## Entregable
Notebook ejecutado con la tabla comparativa, la tabla de integridad y la recomendación.

## Criterios de evaluación (15 puntos del componente "ejercicios")
| Criterio | Puntos |
|---|---|
| Tres configuraciones aplicadas con `pandas_udf` sobre toda la tabla Silver | 4 |
| Las seis métricas calculadas correctamente | 5 |
| Prueba de integridad de hechos clave | 3 |
| Recomendación argumentada con métricas | 3 |

## Pistas
- Para parametrizar la UDF, crea una función "fábrica" que reciba `chunk_size` y `chunk_overlap` y devuelva la `pandas_udf`.
- `F.expr("percentile_approx(len, 0.9)")` calcula el percentil 90.
- `F.col("chunk").rlike(r"[.!?|]\s*$")` detecta finales de frase.
