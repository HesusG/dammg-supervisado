# Proyecto final: analistas de MiBici

**DAMMG · Módulo VI · Aprendizaje supervisado** · 30 % de la calificación del módulo

## Qué van a hacer

En equipos de **2 o 3 personas**. Son el equipo de datos de MiBici. Responden **preguntas de negocio** con los modelos del módulo, siempre con los datos de MiBici.

**Preguntas de negocio**
1. **Mover bicis de noche.** ¿Cuántas bicis saldrán mañana a las 8 a.m. en cada estación, y con qué error? ¿Le ganan a "lo mismo que la semana pasada"? (regresión, MAE y RMSE)
2. **Alertas con presupuesto.** Con $N$ viajes de camión al día, ¿a qué estaciones y horas mandan alerta? ¿Cuántos desbalances atrapan y cuántos viajes son en balde? (clasificación, umbral, precision y recall)
3. **¿Dónde falla el modelo?** ¿En qué estaciones, horas o días se equivoca más? ¿Qué le recomiendan a operaciones? (errores por grupo, características derivadas)
4. **¿Qué modelo ponemos en producción?** Comparen al menos tres modelos del módulo (lineal o logística, árbol o bosque, SVM) y recomienden uno, con su métrica.
5. **¿Sigue sirviendo después de las vacaciones?** El modelo entrenado hasta junio, ¿funciona igual en agosto y septiembre?

**Teoría**
- **T1.** Si predicen siempre la misma constante $c$, demuestren que $\frac{1}{m}\sum_i (c - y^{(i)})^2$ es mínimo cuando $c = \bar y$.
  _Pista: repasen la derivada del Módulo III y el costo J de la sesión 1 (bloque B)._
- **T2.** Con un presupuesto fijo de $k$ alertas, muestren que $\text{precision} = TP/k$ y que $\text{recall} = \text{precision}\cdot k / P$, con $P$ los desbalances reales. ¿Qué quiere decir para el negocio?
  _Pista: repasen la matriz de confusión de la sesión 2; las k alertas son TP + FP._

El notebook trae andamiaje para las dos al final, en la sección **Teoría (T1 y T2)**.

Las preguntas caen en los criterios de la rúbrica: la 1 y la 2 en C2 a C4; la 3 y la 5 en C5 y C6; la 4 en C3; la teoría, en C5.

## Qué entregan

1. **Notebook** (`proyecto.ipynb`, a partir de la plantilla de esta carpeta) que corra de principio a fin con *Reiniciar y ejecutar todo*.
2. **Reporte de 2 páginas** en PDF (carta, letra de 11 pt o más; tablas y figuras cuentan dentro de las 2 páginas). Al final, una línea con lo que hizo cada integrante.

Una sola entrega por equipo en Moodle, con los nombres de todo el equipo en el notebook y en el reporte.

**Equipos:** regístrenlo en Moodle a más tardar el **martes 10 de noviembre**. Una persona guarda la copia del notebook en Drive y la comparte con el resto como editor.

**Fecha de entrega: domingo 22 de noviembre de 2026, 23:59 (hora de la Ciudad de México).** Tolerancia sin penalización hasta el **miércoles 25 de noviembre, 23:59**. Después de la tolerancia no se recibe. La retroalimentación por criterio llega en Moodle antes del 2 de diciembre.

## Datos: el recorte del módulo o el panel completo

La plantilla carga el **recorte del módulo**: las 40 estaciones más activas, de octubre de 2025 a septiembre de 2026, una fila por estación y hora. Son los mismos datos de los labs y las tareas.

Si quieren más, el notebook trae una celda opcional para el **panel completo** (`USAR_PANEL_COMPLETO = True`): las 469 estaciones de MiBici de octubre de 2025 a septiembre de 2026, unos 4.1 millones de filas. Es el mismo panel que usa el curso TC5061 (Tec de Monterrey), así que permite comparar con lo que se hace allá:

| | Este módulo | TC5061 |
|---|---|---|
| Regresión | `departures` a 24 h con MAE (L1, T1), contra el baseline «misma hora de la semana pasada» | **P1**: el mismo objetivo y la misma métrica, con Ridge y gradient boosting |
| Clasificación | evento \|`future_net`\| ≥ 7 (percentil 85 de train, ≈ 16 %), L2–L4 y T1–T2 | **P2**: `risk` = \|`future_net`\| ≥ 0.5 × capacidad (≈ 2 %), con PR-AUC |
| Partición | la misma que TC5061: train oct 2025–jun 2026, val jul, test ago–sep (columna `particion`) | train ≤ jun, val jul, test ago–sep (columna `particion`) |

Ideas de comparación: ¿un árbol o un SVM del módulo superan a una logística balanceada en `risk`? ¿Cambia el modelo ganador cuando el evento pasa del 16 % al 2 %? ¿Qué pasa con la precision a un mismo presupuesto de alertas?

Con 4 millones de filas, SVC y las rejillas grandes no terminan en Colab: submuestreen train (por ejemplo, unas estaciones o unas semanas) y díganlo en el reporte.

**Atribución.** Datos derivados de *MiBici Pública, datos abiertos* (Gobierno de Jalisco), agregados por estación y hora con fines académicos. No contienen datos personales.

## Qué debe contener

La plantilla está organizada con los mismos seis criterios de la rúbrica:

| Criterio | Peso | Lo mínimo que se espera |
|---|---|---|
| C1. Problema y datos | 10 % | Pregunta clara, variable objetivo, unidad de predicción y límites de los datos |
| C2. Partición honesta y baseline | 20 % | Partición acorde a los datos (temporal si hay tiempo), sin fugas, con un baseline razonable y su número |
| C3. Comparación de modelos | 20 % | Al menos 3 modelos, cada uno con hiperparámetros elegidos en validación, en una tabla |
| C4. Métrica justificada | 20 % | Métrica y umbral justificados por el costo de cada error o por un presupuesto; test reportado una vez (en regresión no hay umbral: justifiquen MAE contra RMSE por el costo del error) |
| C5. Lectura con la teoría | 20 % | Resultados explicados con sesgo/varianza **y** con el sesgo inductivo de cada modelo (VI.1) |
| C6. Limitaciones y riesgos | 10 % | Dos limitaciones y un riesgo (ético, operativo o de datos) con su mitigación |

Calificación: Σ peso × (nivel / 3) × 10, con niveles 3 (logrado), 2 (en desarrollo) y 1 (inicial). Si el notebook no corre de principio a fin se descuenta 1 punto (sobre 10) y tienen 48 h para corregirlo.

## Reglas de juego

- **Test se usa una vez**, al final, con el modelo y el umbral ya congelados (la plantilla trae la bandera `REPORTE_FINAL`). Si cambian algo después de ver test, díganlo en el reporte.
- Todo lo que se aprende de los datos (umbral del evento, escaladores, imputaciones) se ajusta **solo con train**.
- Para cada variable, pregúntense a qué hora exacta la conocerían en producción. En MiBici están prohibidas como entrada las columnas de la hora *t* (`departures`, `arrivals`, `net_flow`) y cualquier cosa que mire el futuro.
- Pueden usar IA para estudiar o depurar código. El análisis, las decisiones y la redacción son del equipo, y los números del reporte deben salir de su notebook.

## Carga estimada

Unas 5 horas por persona. Un plan razonable: sesión 1 (1.5 h) problema, datos, partición y baseline; sesión 2 (2 h) modelos y rejillas; sesión 3 (1.5 h) umbral, test, lectura y reporte.
