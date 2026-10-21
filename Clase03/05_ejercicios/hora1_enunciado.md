# Laboratorio Práctico Hora 1: Matriz de Calibración Estocástica (`temperature` y `top_p`)
**Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
**Duración estimada:** 20 minutos | **Modalidad:** Individual / Parejas en Databricks

---

## 1. Contexto del Ejercicio
En sistemas de Ingeniería de Datos y Desarrollo de Software con IA Generativa, usar los hiperparámetros por defecto del proveedor es una fuente común de fallos silenciosos:
- Si la **temperatura es muy alta** al generar consultas SQL o esquemas JSON, el modelo puede introducir variaciones de nombres de columnas o romper la sintaxis en una de cada cinco ejecuciones.
- Si la **temperatura y el `top_p` son cero/mínimos** al pedir escenarios creativos de pruebas de estrés (*chaos testing*) o nombres de features analíticos, el modelo repetirá siempre las mismas ideas obvias.

En este laboratorio construirás un banco de pruebas empírico en Databricks usando `ai_query()` para evaluar **5 combinaciones de `temperature` y `top_p`** sobre un prompt híbrido que exige tanto rigor técnico (SQL) como ideación analítica.

---

## 2. Las 5 Combinaciones Experimentales a Evaluar

Deberás configurar y ejecutar en `clase03/05_ejercicios/hora1_starter.py` la siguiente matriz:

| ID | Nombre del Perfil | `temperature` | `top_p` | Hipótesis de Comportamiento |
|---|---|---|---|---|
| **C1** | **Congelado Determinista** | `0.0` | `1.00` | Máxima reproducibilidad; las corridas repetidas deben ser idénticas carácter por carácter. |
| **C2** | **Técnico Conservador** | `0.2` | `0.90` | Sintaxis SQL estable con mínima variación en comentarios o alias. |
| **C3** | **Balanceado Estándar** | `0.5` | `0.85` | Equilibrio entre precisión técnica y fluidez explicativa. |
| **C4** | **Núcleo Estrecho con Temp Alta** | `0.8` | `0.20` | Experimento de choque: aunque la temperatura es alta (`0.8`), el recorte agresivo de `top_p = 0.20` restringe el muestreo al 20% superior de masa probabilística. |
| **C5** | **Exploración Abierta** | `1.0` | `0.98` | Alta diversidad léxica y creatividad, pero mayor riesgo de deriva en formatos estrictos. |

---

## 3. Instrucciones Paso a Paso

1. **Importa y abre** el archivo `clase03/05_ejercicios/hora1_starter.py` en tu Workspace de Databricks.
2. **Paso 1 (TODO 1):** Completa la función `ejecutar_con_parametros()` para que invoque `ai_query()` pasando `temperature`, `top_p` y `max_tokens` dentro de `modelParameters => named_struct(...)`.
3. **Paso 2 (TODO 2):** Completa el prompt de Ingeniería de Contexto (`PROMPT_AUDITORIA_DATOS`) inyectando:
   - El **Rol** de Ingeniero Senior de Datos en Databricks.
   - El **Esquema DDL** de la tabla `main.riesgos.transacciones_tarjeta`.
   - La **Tarea Dual**: (Parte A) escribir una consulta SQL que detecte clientes con más de 3 transacciones rechazadas en el mismo día, y (Parte B) proponer 2 hipótesis creativas de fraude emergente.
   - Las **Restricciones de Salida**.
4. **Paso 3 (TODO 3):** Define la lista de diccionarios con las **5 combinaciones (`C1` a `C5`)** y ejecuta el bucle experimental (2 corridas por cada combinación) para medir:
   - ¿Son idénticas la Corrida 1 y la Corrida 2? (Determinismo empírico).
   - ¿Respeta exactamente las columnas del DDL?
   - Índice de diversidad léxica (vocabulario único / total de palabras).
5. **Paso 4 (Reflexión para el Reto Integrador):** Responde en la última celda qué perfil (`C1` a `C5`) usarán en cada módulo del Reto Integrador de tu equipo.

---

## 4. Criterios de Éxito
- El notebook se ejecuta de principio a fin sin errores en Databricks.
- La tabla final muestra las métricas comparativas de las 5 combinaciones.
- Se comprueba que `C1 (temp=0.0, top_p=1.0)` produce determinismo estricto y `C5 (temp=1.0, top_p=0.98)` maximiza la diversidad de vocabulario.
