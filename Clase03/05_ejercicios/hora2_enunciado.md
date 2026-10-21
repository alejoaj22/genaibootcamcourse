# Laboratorio Práctico Hora 2: Zero-Shot vs. Few-Shot vs. Chain-of-Thought (CoT)
**Módulo 3: Soluciones con IA Generativa — Clase 3: HACER (Parte 1)**
**Duración estimada:** 20 minutos | **Modalidad:** Individual / Equipos en Databricks

---

## 1. Objetivo del Laboratorio
En este ejercicio resolverás **3 problemas técnicos reales de complejidad creciente** en Databricks utilizando `ai_query()`, aplicando y comparando tres técnicas de prompting:
1. **Zero-Shot:** Instrucción directa sin ejemplos ni pasos intermedios.
2. **Few-Shot Balanceado:** Inyección de 3 ejemplos representativos `(Entrada -> Salida)` en el contexto.
3. **Chain-of-Thought (CoT) con Scratchpad:** Descomposición explícita paso a paso dentro de `[SCRATCHPAD]...[/SCRATCHPAD]` antes de emitir `[VEREDICTO_FINAL]`.

---

## 2. Los 3 Problemas a Resolver

### Problema 1: Clasificación de Alertas de Calidad de Datos (Data Quality Triage)
Debes clasificar un mensaje ambiguo de alerta de un pipeline de datos en una de las etiquetas corporativas exactas (`DQ_SCHEMA_DRIFT`, `DQ_NULL_SPIKE`, `DQ_LATENCY_SLA`, `DQ_FALSE_ALARM`):
> **Caso a evaluar:** *"Atención: la columna `telefono_secundario` pasó de 12% a 14% de valores vacíos tras la campaña opcional de registro rápido web, dentro del umbral permitido del 25%."*
- **Verdad Terreno Esperada:** `DQ_FALSE_ALARM` (porque aunque subieron los vacíos del 12% al 14%, está por debajo del umbral máximo permitido del 25%).

### Problema 2: Conciliación de Inventario Multibodega y Merma Logística
Calcular el stock disponible real para venta online (`stock_vendible`) de un SKU aplicando reglas secuenciales:
> **Caso a evaluar:** *"Bodega Central tiene 450 unidades y Bodega Norte tiene 350 unidades. Del total físico combinado, el 5% se descarta por merma de humedad. De las unidades buenas restantes, 160 están reservadas para pedidos B2B ya pagados y se debe retener un colchón de seguridad fijo de 40 unidades que no se muestran en la web."*
- **Verdad Terreno Esperada:** `560` unidades (Físico total: $450 + 350 = 800$; menos $5\%$ de merma ($40$) = $760$ unidades buenas; menos $160$ B2B y menos $40$ de colchón = $560$).

### Problema 3: Detección de Contradicción Lógica en Reglas de Negocio
Auditar un conjunto de 3 reglas condicionales propuestas para un sistema de descuentos y determinar qué reglas entran en conflicto directo sobre la condición de mora:
> **Caso a evaluar:**
> - *Regla A:* Clientes con `antiguedad >= 24` meses reciben `DESCUENTO_15`.
> - *Regla B:* Clientes con `mora_activa == True` tienen prohibido cualquier descuento (`SIN_DESCUENTO`).
> - *Regla C:* Clientes de categoría `CORPORATIVO` reciben siempre `DESCUENTO_20` sin importar su estado de mora.
> - *Pregunta:* Evalúa el caso de un cliente `CORPORATIVO` con `antiguedad = 36` y `mora_activa = True`. ¿Cuáles son las dos reglas que tienen una contradicción directa e irreconciliable sobre la mora?
- **Verdad Terreno Esperada:** `CONFLICTO_BC`.

---

## 3. Instrucciones Paso a Paso
1. Abre `clase03/05_ejercicios/hora2_starter.py` en Databricks.
2. Completa `# TODO 1`: Añade ejemplos Few-Shot balanceados para enseñar el criterio de umbrales de Data Quality, cálculo de inventario y conflictos de reglas.
3. Completa `# TODO 2`: Diseña el bloque `[SCRATCHPAD]` paso a paso para resolver los cálculos de inventario y la auditoría lógica.
4. Ejecuta la matriz comparativa (`3 problemas x 3 técnicas = 9 ejecuciones`) y analiza en la tabla resumen cómo evoluciona la tasa de acierto.
