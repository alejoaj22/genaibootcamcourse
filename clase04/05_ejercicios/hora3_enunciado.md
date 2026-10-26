# Laboratorio Práctico Hora 3: Construcción de un Eval Dataset y Evaluación Comparativa A/B con LLM-as-a-Judge

**Módulo 3: Soluciones con IA Generativa — Clase 4: HACER (Parte 2)**  
**Tiempo estimado:** 20 minutos (10 min en aula + 10 min de consolidación con Checkpoint 1)  
**Modalidad:** En los equipos del Reto Integrador  
**Entregables:** `hora3_starter.py` completado y verificado en Databricks  

---

## 1. Contexto del Problema y Objetivos de Aprendizaje

El desarrollo de aplicaciones con LLMs no puede depender de corazonadas ni de probar prompts manualmente con 1 o 2 ejemplos en la interfaz web. Para garantizar calidad de grado de producción, implementamos **Eval-Driven Development (EDD)**:
1. Diseñar un conjunto de evaluación de referencia (*Golden Eval Dataset*).
2. Definir métricas cuantitativas automáticas (Keyword Match, ROUGE-L).
3. Configurar un **Juez Evaluador Inteligente (LLM-as-a-Judge)** que aplique una rúbrica analítica y detecte alucinaciones.
4. Tomar decisiones de arquitectura con evidencia estadística.

### Objetivos:
- Construir un dataset de evaluación de **5 preguntas sobre dPhone**, incluyendo al menos una pregunta trampa no respondible.
- Definir dos variantes de prompt (Variante A simple vs Variante B estructurada con XML y Grounding).
- Implementar el cálculo de ROUGE-L mediante la subsecuencia común más larga (LCS).
- Implementar un LLM-as-a-Judge usando `databricks-gpt-oss-120b`.
- Generar una tabla comparativa y emitir una conclusión formal con regla de decisión.

---

## 2. Requerimientos Técnicos y Pasos a Seguir

Abre el archivo `clase04/05_ejercicios/hora3_starter.py` en tu Workspace de Databricks y completa los siguientes `# TODO`:

1. **TODO 1: Definición del Eval Dataset de 5 Casos:**
   - 4 preguntas sobre especificaciones o políticas de dPhone con su respuesta de referencia y palabras clave.
   - 1 pregunta no respondible cuya referencia exacta sea `"No encuentro esa información en la documentación proporcionada."`.
2. **TODO 2: Variante A (Prompt Básico) vs Variante B (Prompt Avanzado con XML y Grounding):**
   - Escribir ambas funciones generadoras de prompt.
3. **TODO 3: Implementación de Métrica ROUGE-L con LCS:**
   - Implementar la función de cálculo de ROUGE-L manual para medir solapamiento estructural sin librerías externas.
4. **TODO 4: Configuración del Evaluador LLM-as-a-Judge:**
   - Formular el prompt del juez con `databricks-gpt-oss-120b` exigiendo un JSON con `puntaje` (1 a 5) y `es_factualmente_correcto` (bool).
5. **TODO 5: Tabla Comparativa y Veredicto Final:**
   - Comparar el desempeño promedio de ambas variantes y emitir la conclusión técnica.

---

## 3. Criterios de Aceptación (Checklist)

- [ ] El eval dataset cuenta con exactamente 5 casos (4 respondibles y 1 no respondible).
- [ ] La función de ROUGE-L normaliza adecuadamente el texto en español (elimina puntuación y tildes).
- [ ] El Juez LLM emite evaluaciones con justificaciones cualitativas en JSON.
- [ ] La Variante B supera a la Variante A en el caso no respondible (se abstiene sin alucinar).
- [ ] La tabla final documenta ROUGE-L, coincidencia de keywords y puntaje del juez para cada variante.

---

## 4. Conexión Directa con el Reto Integrador (CHECKPOINT 1)

El eval dataset y el framework de evaluación que construyas en este ejercicio son la base técnica directa del **Checkpoint 1** de tu proyecto. En el Checkpoint 1 ampliarás este dataset a 10+ casos específicos de tu dominio de negocio.
