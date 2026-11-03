# DAMMG · Módulo VI · Aprendizaje supervisado

Material público del Módulo VI del *Diplomado en Aprendizaje de Máquina y Modelos Generativos* (Facultad de Ciencias, UNAM, 2026).

- **Sitio:** https://hesusg.github.io/dammg-supervisado/
- **[Menú de notebooks](notebooks/README.md)**: qué notebook va con cada sesión, cuándo se entrega y el botón para abrirlo en Colab.
- **Slides** (S1–S4), **widgets** interactivos y **notebooks** para Colab (sin soluciones). Cada notebook también se puede ver, de solo lectura, en el sitio.

Este repositorio contiene solo artefactos generados. Lo construye un script desde una fuente privada y no acepta *pull requests*.

Licencia: CC BY-NC-SA 4.0. Las fuentes de terceros (Andrew Ng, Stanford CS229, MIT OCW, Caltech CS156) se citan en cada slide.

## Datos

Los notebooks descargan solos los datos desde el sitio (`datos/`), con verificación sha256:

- `mibici_curso.parquet`: 40 estaciones de MiBici (Guadalajara), oct 2025–sep 2026, una fila por estación y hora. Lo usan los labs y las tareas.
- `mibici_panel_completo.parquet`: 469 estaciones, oct 2025–sep 2026 (opcional, para el proyecto).

Datos derivados de *MiBici Pública, datos abiertos* (Gobierno de Jalisco), agregados por estación y hora con fines académicos. No contienen datos personales.
