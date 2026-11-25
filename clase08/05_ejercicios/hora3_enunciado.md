# Ejercicio Hora 3 — Propuesta Formal del Prototipo Funcional (MVP, Arquitectura y Roles)

| Campo | Valor |
|---|---|
| Hora | 3 · Gestión del Proyecto y Arquitectura de la Solución |
| Modalidad | En equipos mixtos definitivos (3–4 personas: Datos + Desarrollo + Negocio/QA) |
| Tiempo en clase | 15 minutos |
| Archivos | `hora3_starter.py` (esqueleto con `# TODO`) · `hora3_solucion.py` (referencia) |

## Objetivo
Definir y congelar el alcance técnico y de negocio del **prototipo funcional** que cada equipo construirá y monitoreará en la Clase 9 (y presentará en el Demo Day de la Clase 12), garantizando conexión explícita entre datos en Databricks, evaluación MLflow, Serving Endpoint, interfaz de usuario y distribución de roles en equipo mixto.

## Qué debes hacer
1. Abre `hora3_starter.py` en equipo.
2. En el `# TODO 1`, redacta el **Problema y Propuesta de Valor** (usuario objetivo, dolor actual y objetivo medible).
3. En el `# TODO 2`, aplica la **Regla 1-1-1** (1 usuario, 1 flujo, 1 métrica) y delimita:
   - Máximo **3 funcionalidades `in_scope`** (MVP real para Clases 8–9).
   - Mínimo **3 funcionalidades `out_of_scope`** (qué queda para la v2 y NO se construye ahora).
4. En el `# TODO 3`, describe la **Arquitectura de Integración** (tablas/índice en Unity Catalog, modelo y Serving Endpoint, aplicación consumidora en Clase 9) y define el **contrato JSON** exacto (`request` y `response`).
5. En el `# TODO 4`, asigna los **roles concretos del equipo mixto** (Data/AI Engineer, App/Backend Developer, Product/Negocio & QA) con entregables claros para hoy (Clase 8) y para el Sprint (Clase 9).
6. En el `# TODO 5`, documenta al menos **2 riesgos técnicos o de negocio** con su mitigación y ejecuta el validador automático para generar el documento Markdown final de la propuesta.

## Entregable
- Notebook `hora3_starter.py` ejecutado sin errores de validación (`AssertionError`), imprimiendo la Propuesta Formal en Markdown lista para usar en el Sprint de la Clase 9.

## Criterios de verificación (formativos)
| Criterio | Qué se espera |
|---|---|
| Alcance realista | Diferencia claramente MVP de PoC y Producto; `in_scope` ≤ 3 ítems y `out_of_scope` explícito. |
| Integración técnica | Conecta datos en Unity Catalog → Serving Endpoint → App/Batch con contrato JSON válido. |
| Calidad y métricas | Incluye umbral cuantitativo en MLflow (`Correctness`, `Safety`, `Guidelines`). |
| Equipos mixtos | Todos los perfiles tienen entregables accionables en paralelo para Clase 8 y Clase 9. |
