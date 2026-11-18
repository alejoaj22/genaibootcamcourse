# Ejercicio Hora 2 — Crear 2 funciones propias en Unity Catalog

| Campo | Valor |
|---|---|
| Hora | 2 · Lab: UC Functions como tools (notebook 2.3) |
| Modalidad | En parejas |
| Tiempo en clase | 8 minutos (+ 2 de verificación) |
| Archivos | `hora2_starter.py` (esqueleto con `# TODO`) · `hora2_solucion.py` (referencia) |
| Requisitos | `Includes/Lab_Setup_02` ejecutado (tablas `smartphones` e `inventory_stock`), permiso `CREATE FUNCTION`, cómputo serverless |

## Objetivo
Diseñar y registrar **dos herramientas gobernadas** para el agente dPhone, aplicando las buenas prácticas de la clase: nombre claro, descripción útil, parámetros tipados y descritos, validación de entradas y pruebas de casos límite.

## Qué debe hacer

### Función 1 — SQL con parámetro: `get_phones_under_price`
Debe devolver los teléfonos cuyo precio sea **menor o igual** a un presupuesto, ordenados del más caro al más barato, con las columnas `phone_id`, `model`, `color` y `price`.
- Parámetro: `max_price INT` (con `COMMENT`).
- La función debe tener un `COMMENT` que diga **qué hace**, **cuándo usarla** y **qué devuelve**.
- Evite usar para el parámetro el mismo nombre que una columna.

### Función 2 — Python: `estimate_delivery_date`
Debe calcular la fecha estimada de entrega sumando **días hábiles** (lunes a viernes) a partir de hoy y devolverla como texto `YYYY-MM-DD`.
- Parámetro: `business_days: int` (válido entre 1 y 30; fuera de rango devuelve un mensaje que empiece por `ERROR:`).
- Imports **dentro** de la función; sin parámetros por defecto; type hints y docstring con `Args` y `Returns`.

> Puede **renombrar** las funciones o inventar otras dos con el mismo nivel de complejidad (una SQL con parámetro y una Python). Si lo hace, cambie los nombres en las variables del starter para que la auditoría las encuentre.

## Pasos
1. Abra `hora2_starter.py` y ejecute las celdas de instalación y `%run`.
2. Complete los `TODO` del `COMMENT` y del cuerpo de la función SQL; ejecute la celda.
3. Complete el docstring y el cuerpo de la función Python; ejecute la celda de registro.
4. Ejecute las **3 pruebas** de cada función (caso normal, caso límite y caso inválido).
5. Ejecute la **auditoría automática**: ambas funciones deben salir en verde (✅).
6. Lea el `COMMENT` de cada función **como si fuera el LLM**: ¿sabría cuándo usarla y cuándo no?

## Entregable
- Dos funciones registradas en `<catálogo>.genai_course`.
- La tabla de auditoría con todos los criterios en `True`.
- Una frase: *"Una descripción ambigua podría hacer que el agente …"*.

## Criterios de evaluación (formativos)
| Criterio | Qué se espera |
|---|---|
| Descripción | `COMMENT`/docstring con qué hace, cuándo usarla y qué devuelve (≥ 40 caracteres, sin `TODO`). |
| Parámetros | Todos con descripción y tipo. |
| Robustez | Valida entradas fuera de rango y no falla ante casos límite. |
| Pruebas | 3 casos por función (normal, límite, inválido) con resultado correcto. |
| Gobernanza | La función queda en el esquema correcto y visible en Catalog Explorer. |

## Pistas
- En SQL, `WHERE price <= max_price` usa el parámetro; la columna `price` y el parámetro `max_price` tienen nombres distintos.
- En Python, `date.weekday()` devuelve 0 para lunes y 6 para domingo.
- Si `create_python_function` falla por falta de conexión, confirme que está en cómputo **serverless**.
- Cada función que cree en `genai_course` se convertirá automáticamente en herramienta del agente de la Hora 3 (comodín `*`).
