# Ejercicio 2 (Hora 2) — Ingestar documentos propios del reto en Bronze

**Tiempo:** 15 minutos · **Modalidad:** individual · **Archivos:** `hora2_starter.py`, `hora2_solucion.py`

## Contexto
Has ejecutado el notebook 1.1A con los documentos de dPhone. Ahora crearás tu propia tabla Bronze con documentos de tu reto.

## Instrucciones
1. Sube de 3 a 5 documentos de tu reto (PDF, DOCX, imágenes) a una carpeta nueva del volumen `dataset`, por ejemplo `<volumen>/docs_reto/` (Catalog → tu esquema → volumen `dataset` → *Upload to this volume*).
   - Si no tienes documentos propios, usa `docs` de dPhone con otro nombre de tabla.
2. Completa `hora2_starter.py` para:
   - Leer la carpeta con `binaryfile` y agregar `ingestion_time`.
   - Derivar `file_name`, `extension`, `size_kb` y `content_sha256`.
   - Detectar archivos vacíos o duplicados.
   - Guardar la tabla como `bronze_reto` (modo `overwrite`).
3. Verifica el resultado con `DESCRIBE DETAIL` y un `COUNT(*)`.
4. Responde por escrito (en una celda markdown): ¿Qué metadato adicional agregarías para tu reto y por qué?

## Entregable
- Tabla `bronze_reto` creada y notebook con las celdas ejecutadas.

## Criterios de revisión
- [ ] La lectura usa `format("binaryfile")` y la ruta se construye con `course.dataset_volume`.
- [ ] Existen las columnas de metadatos requeridas.
- [ ] Se validaron vacíos y duplicados.
- [ ] La tabla se guardó con `saveAsTable`.
- [ ] La reflexión sobre metadatos es concreta.

## Pistas
- `F.regexp_extract("path", r"([^/]+)$", 1)` obtiene el nombre del archivo.
- `F.sha2(F.col("content"), 256)` calcula la huella del contenido.
- Si la carpeta no existe, `dbutils.fs.mkdirs(ruta)` la crea.
