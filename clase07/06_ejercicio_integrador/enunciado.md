# Ejercicio Integrador — Clase 7: Agente dPhone con 4+ herramientas y RAG

| Campo | Valor |
|---|---|
| Clase | 7 · TRANSFORMAR (Parte 3) |
| Modalidad | Individual (pueden consultar en parejas, pero cada quien entrega su notebook) |
| Tiempo estimado | 90–120 minutos fuera de clase |
| Entrega | Antes de la Clase 8 (el agente se evaluará y desplegará en esa sesión) |
| Archivos | `starter.py` (esqueleto con `# TODO`) · `solucion.py` (referencia, se entrega después) · `rubrica.md` (100 puntos) |
| Resultados de aprendizaje | RA3 (principal), RA2 y RA4 |

## 1. Objetivo
Construir, probar y documentar un **agente funcional** para soporte y ventas de dPhone que combine **al menos cuatro herramientas gobernadas en Unity Catalog** y la **herramienta de recuperación (RAG)** sobre el índice `dphone_vs_index`, y demostrar con **cinco o más consultas** que usa las herramientas correctas.

## 2. Requisitos obligatorios
1. **Mínimo 4 herramientas de Unity Catalog** en `<catálogo>.genai_course`, con `COMMENT` útil y parámetros descritos. Pueden ser las del notebook 2.3 (`get_latest_phone`, `check_inventory`, `get_current_user`, `generate_checkout_url`) y las del ejercicio de la Hora 2.
2. **Al menos 1 herramienta nueva y propia** (SQL o Python) que usted diseñe y valide (por ejemplo `get_phones_under_price`, `estimate_delivery_date` o una idea suya).
3. **RAG:** el agente usa `VectorSearchRetrieverTool` sobre `dphone_vs_index` con un `tool_description` claro.
4. **Menor privilegio:** el agente carga las funciones con una **lista explícita de nombres**, no con el comodín `*`.
5. **System prompt profesional** con persona, alcance, reglas de herramientas, formato y un ejemplo. Debe incluir: confirmar antes de generar un enlace de compra, no inventar datos, rechazar lo que esté fuera de alcance e ignorar instrucciones maliciosas.
6. **Pruebas:** como mínimo **5 consultas** distintas (se recomiendan 7), cada una con las herramientas que se espera que use. Debe incluir:
   - una consulta que use **RAG** (documentación),
   - una consulta que **encadene dos o más herramientas**,
   - una consulta con **acción** (enlace de compra) con confirmación explícita,
   - una consulta **adversa** (fuera de alcance o inyección de prompt).
7. **Trazabilidad:** abra al menos **dos trazas** en MLflow (una exitosa y una compleja) y anote qué observó.
8. **Evaluación inicial:** ejecute `mlflow.genai.evaluate` con al menos un scorer predefinido (`RelevanceToQuery` o `Safety`) y el scorer propio de uso de herramientas.
9. **Reflexión de seguridad y gobernanza** (8–10 líneas) sobre permisos, riesgos y límites del agente.

## 3. Datos y contexto de apoyo
- Tablas: `smartphones` (`phone_id`, `model`, `color`, `price`, `release_date`, …) e `inventory_stock` (`item_id`, `quantity`).
- Índice: `dphone_vs_index` (columnas `id`, `path`, `chunk`); documentos **en inglés**.
- LLM: `databricks-gpt-oss-120b`.
- Consultas sugeridas: ver `../datos/consultas_agente_dphone.csv`.

## 4. Pasos sugeridos
1. Abra `starter.py` y ejecute la instalación y el `%run`.
2. Verifique que existen el índice y las funciones; complete la lista de **herramientas propias**.
3. Escriba su `SYSTEM_PROMPT` y la descripción del índice (`RAG_DESCRIPTION`).
4. Defina sus 5 o más consultas con las herramientas esperadas.
5. Ejecute la construcción del agente y la batería de pruebas.
6. Revise las trazas y ajuste el prompt o las descripciones de las herramientas **una vez** (iteración).
7. Ejecute la evaluación y complete la bitácora y la reflexión.
8. Entregue el notebook ejecutado (con salidas visibles) y capturas de las dos trazas.

## 5. Entregables
| # | Entregable | Formato |
|---|---|---|
| 1 | Notebook `starter.py` completado y ejecutado | Archivo `.py` de Databricks (o `.ipynb` exportado) con salidas |
| 2 | Tabla de pruebas (consulta, herramientas esperadas, herramientas usadas, resultado) | Dentro del notebook |
| 3 | Capturas de 2 trazas de MLflow | Imágenes o enlaces |
| 4 | Bitácora de iteración (qué cambió y qué efecto tuvo) | Dentro del notebook |
| 5 | Reflexión de seguridad y gobernanza | Dentro del notebook |

## 6. Evaluación
Se califica con `rubrica.md` (100 puntos). Los criterios principales son: herramientas gobernadas (20), RAG (10), system prompt (15), pruebas (25), trazabilidad y evaluación (10), seguridad y gobernanza (10) y calidad/comunicación (10).

## 7. Consejos
- Una herramienta bien descrita vale más que tres mal descritas: lea sus `COMMENT` como si fuera el LLM.
- Si el agente no usa una herramienta, revise primero **la descripción** y **el prompt**, luego los permisos.
- Los modelos son probabilísticos: ejecute dos veces las consultas que fallen antes de cambiar el diseño.
- No exponga secretos ni datos sensibles en el prompt.
- Guarde la lista final de herramientas: la usaremos para evaluar y desplegar en la Clase 8.

## 8. Extensión opcional
- Agregue una herramienta de **escalamiento** (por ejemplo, una función que registre un ticket) y una regla de prompt para usarla cuando el cliente reporte un daño.
- Registre el agente con `mlflow.pyfunc.log_model` (usando el `agent.py` generado) y declare los `resources`.
