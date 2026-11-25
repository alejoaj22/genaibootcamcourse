# Ejercicio Hora 2 — Desplegar Serving Endpoint y ejecutar 5 queries batch con `ai_query()`

| Campo | Valor |
|---|---|
| Hora | 2 · Deployment Productivo y Batch Inference |
| Modalidad | Individual o por equipo (respetando la cuota de endpoints del workspace) |
| Tiempo en clase | 15 minutos |
| Archivos | `hora2_starter.py` (esqueleto con `# TODO`) · `hora2_solucion.py` (referencia) · `../datos/tickets_soporte_muestra.csv` (respaldo) |

## Objetivo
Configurar un Serving Endpoint en Databricks para el agente `dphone_agent` usando `EndpointCoreConfigInput`, verificar que el servicio responda en estado `READY` y ejecutar **5 consultas en lote (batch inference)** con `ai_query()` en SQL sobre tickets abiertos de `customer_support_tickets`, persistiendo los resultados en una tabla Delta.

## Qué debes hacer
1. Abre `hora2_starter.py` y ejecuta `%run ../../Includes/Lab_Setup_04` para asegurar que la tabla `customer_support_tickets` esté disponible en `<catálogo>.genai_course`.
2. En el `# TODO 1`, completa el diccionario `endpoint_config_dict` con `model_name`, `model_version`, `scale_to_zero_enabled=True` y `workload_size="Small"`.
3. En el `# TODO 2`, utiliza `WorkspaceClient()` para desplegar el endpoint (si no existe y tu workspace tiene cuota) o conectarte al endpoint compartido `dphone_endpoint` si ya fue creado por el profesor, verificando que su estado sea `READY`.
4. En el `# TODO 3`, envía **una consulta de prueba en vivo** mediante el SDK de Databricks (`w.api_client.do(...)`) para comprobar que responde correctamente y medir su latencia.
5. En el `# TODO 4`, escribe una consulta SQL con `ai_query()` que procese **exactamente 5 filas** con `status = 'open'` de `customer_support_tickets` y guarde la salida en la tabla `<catálogo>.genai_course.mi_lote_tickets_eval`.
6. En el `# TODO 5`, inspecciona las 5 respuestas generadas y responde en un print: en cuál de los 5 tickets exigirías revisión humana obligatoria antes de enviar al cliente.

## Entregable
- Notebook `hora2_starter.py` ejecutado con evidencia de endpoint `READY`, latencia de llamada SDK y la tabla Delta con las **5 filas** procesadas por `ai_query()`.

## Criterios de verificación (formativos)
| Criterio | Qué se espera |
|---|---|
| Configuración Serving | `EndpointCoreConfigInput` correctamente construido (`scale_to_zero_enabled=True`, `workload_size="Small"`). |
| Validación de servicio | Endpoint `READY` e invocación de prueba exitosa con registro de latencia. |
| Batch Inference SQL | Uso correcto de `ai_query(endpoint => ..., request => ...)` con `WHERE status = 'open'` y `LIMIT 5`. |
| Criterio FinOps y QA | Uso consciente de `LIMIT 5` y análisis crítico de al menos 1 respuesta generada. |
