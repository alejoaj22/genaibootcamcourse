# Laboratorio Práctico Hora 2: Pipeline Multimodal de Procesamiento de Órdenes de Compra dPhone

**Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**  
**Tiempo estimado:** 20 minutos (15 min en aula + 5 min de puesta en común)  
**Modalidad:** Individual o en parejas del Reto Integrador  
**Entregables:** `hora2_starter.py` completado y verificado en Databricks  

---

## 1. Contexto del Problema y Objetivos de Aprendizaje

Las empresas del sector tecnológico reciben permanentemente órdenes de compra (PO - *Purchase Orders*) y facturas escaneadas o en formato PDF enviadas por distribuidores mayoristas. En este laboratorio, procesarás una **orden de compra comercial para distribuidores de dPhone**, extrayendo sus partidas de productos mediante un pipeline multimodal y validando la coherencia aritmética de los subtotales y totales antes de cargarlos a las tablas analíticas de Databricks.

### Objetivos:
- Construir en memoria una imagen de documento que simula una orden de compra comercial escaneada utilizando `Pillow (PIL)`.
- Aplicar la función `ai_parse_document(content, map('version', '2.0'))` (con fallback de emulación garantizado) para extraer la estructura tabular.
- Utilizar `ai_query()` con **Structured Outputs** (`json_schema`) para obtener un JSON estrictamente tipado con la lista de productos y totales.
- Implementar validaciones aritméticas deterministas (`cantidad * precio_unitario == subtotal` y `sum(subtotales) == gran_total`).
- Visualizar los datos consolidados en un DataFrame de Apache Spark.

---

## 2. Requerimientos Técnicos y Pasos a Seguir

Abre el archivo `clase04/05_ejercicios/hora2_starter.py` en tu Workspace de Databricks y completa los siguientes `# TODO`:

1. **TODO 1: Generación de la Orden de Compra Sintética:**
   - La orden incluye 4 ítems del catálogo oficial de dPhone (ej. dPhone D1, D1 Lite, D1 Pro, D1 Fold) con cantidades, precios unitarios oficiales y subtotales.
2. **TODO 2: Definición del JSON Schema Estricto:**
   - Crear el esquema JSON con propiedades para: `numero_orden` (str), `distribuidor` (str), `items` (array de objetos con `modelo`, `cantidad`, `precio_unitario_usd`, `subtotal_usd`) y `total_orden_usd` (float).
3. **TODO 3: Extracción Asistida por IA con `ai_query()`:**
   - Inyectar la tabla HTML extraída por `ai_parse_document()` en el prompt y solicitar la extracción garantizada con `responseFormat`.
4. **TODO 4: Validación Aritmética y Auditoría de Datos:**
   - Escribir una función en Python que verifique que cada subtotal sea matemáticamente consistente con su cantidad y precio, y que la suma de partidas coincida exactamente con el total declarado.

---

## 3. Criterios de Aceptación (Checklist)

- [ ] La imagen de la orden de compra se genera en memoria y se procesa mediante `ai_parse_document()` o su fallback canónico.
- [ ] El esquema JSON fuerza tipos numéricos para cantidades y precios.
- [ ] El modelo extrae con 100% de exactitud las 4 partidas de productos de dPhone.
- [ ] La validación aritmética comprueba que no existan discrepancias entre subtotales y el gran total.
- [ ] Los ítems validados se muestran en una tabla limpia de Spark.

---

## 4. Retos Extra para Avanzados

1. **Visión Directa con `files => content`:** Si tu workspace tiene habilitado un modelo de visión (como `databricks-llama-4-maverick`), realiza la extracción directamente desde la imagen binaria sin pasar por `ai_parse_document()`.
2. **Alertas de Descuento No Autorizado:** Agrega una regla que compare los precios unitarios de la orden contra los precios de lista del catálogo oficial (`dphone_catalogo_productos.csv`) y marque discrepancias mayores al 10%.
