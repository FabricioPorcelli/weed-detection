# AgroVision-Edge — Brief de proyecto

## 1. Resumen

Sistema de **detección de malezas y plagas en cultivos** (foco inicial: soja, con
posibilidad de extender a maíz) mediante visión por computadora, usando un modelo
de la familia **YOLO**. El diferencial del proyecto no es solo "detectar objetos",
sino que el modelo final debe poder **correr en un dispositivo edge** (Raspberry
Pi, Jetson Nano/Orin Nano, o similar) sin depender de conexión a internet ni de
GPU de datacenter — que es como operan en la práctica sistemas de aplicación
selectiva de agroquímicos en el campo (tipo lo que hace DeepAgro con SprAI, o lo
que necesitaría una empresa como AgroPro para un pulverizador inteligente).

Este documento es la guía para que vos (agente de coding) estructures el repo,
propongas el plan de trabajo por fases, y vayas construyendo el proyecto conmigo
paso a paso. No asumas todo el trabajo de una — al final de cada fase, presentá
resultado y preguntá si seguimos.

## 2. Objetivo del proyecto

Entrenar y optimizar un modelo de detección de malezas/plagas que:

1. Tenga buena precisión (mAP@0.5 competitivo, documentado).
2. Sea **liviano**: candidato final < 10-15 MB, idealmente cuantizado a INT8.
3. Corra inferencia en **tiempo real o cercano a real** en hardware edge (target
   de referencia: <200ms por frame en una Raspberry Pi 4/5 o Jetson Nano — dejar
   el benchmark real documentado, no asumido).
4. Tenga un demo funcional (CLI o Streamlit) que reciba imagen/video y devuelva
   detecciones con bounding boxes y confianza.

## 3. Dataset

Fuente: [Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)
(Kaggle, autor ravirajsinh45).

- ~1300 imágenes de cultivo de sésamo con malezas, anotadas en formato YOLO. 
  Hay un archivo .txt por imagen con los valores: ```<class_id> <x_center> <y_center> <width> <height>```
  Todos los valores estan normalizados entre 0.0 y 1.0.
- Al iniciar el proyecto: explorar el dataset, contar clases reales, verificar
  balance de clases, y documentar en el README qué hay (no asumir que ya está
  perfectamente limpio — hacer un EDA básico primero).
- Dejar preparado el pipeline de datos para que en el futuro se pueda swappear
  este dataset por uno específico de soja/maíz sin reescribir todo (separar
  bien config de dataset del resto del código).

## 4. Stack técnico sugerido

- **Python 3.10+**
- **Ultralytics YOLO** (YOLOv8n o YOLOv11n — variante *nano*, pensada para edge,
  no las versiones grandes)
- **OpenCV** para preprocesamiento e inferencia sobre imagen/video
- **ONNX** como formato intermedio de exportación
- Exportación final a formato edge-friendly según el hardware target que
  definamos (ONNX Runtime, TFLite, o NCNN — evaluar cuál tiene mejor soporte
  para el dispositivo que uso como referencia de benchmark)
- **Streamlit** para el demo visual (mismo patrón que ya usé en otro proyecto
  de forecasting, mantener consistencia de portfolio)
- Cuantización **INT8** post-training como paso de optimización edge

## 5. Estructura de repo esperada

```
agrovision-edge/
├── README.md                # objetivo, resultados, cómo correrlo, benchmarks
├── data/
│   ├── raw/                 # dataset original (no versionar en git, .gitignore)
│   └── processed/           # splits train/val/test, YAML de config YOLO
├── notebooks/
│   └── 01_eda.ipynb         # exploración del dataset
├── src/
│   ├── train.py             # entrenamiento
│   ├── export.py            # exportación a ONNX / edge format
│   ├── quantize.py          # cuantización INT8
│   ├── benchmark.py         # mide latencia, tamaño de modelo, mAP
│   └── inference.py         # inferencia standalone (CLI)
├── app/
│   └── streamlit_app.py     # demo visual
├── models/                  # pesos entrenados (.pt, .onnx, cuantizado)
├── requirements.txt
└── .gitignore
```

## 6. Fases de trabajo (ir de a una, no saltar)

> **Regla para el agente:** cuando en una fase aparezca un bloque
> **[DECISIÓN]**, no elijas la opción por tu cuenta ni asumas un default.
> Presentá las opciones (con su trade-off en una línea cada una) y esperá
> respuesta de Fabricio antes de implementar esa parte. El resto de cada
> fase (lo que no está marcado como decisión) sí lo podés ejecutar directo.

### Fase 1 — Setup + EDA

1.1. Estructura de repo según sección 5, entorno virtual, `requirements.txt`
     inicial (ultralytics, opencv-python, matplotlib, pandas, streamlit).

1.2. Descargar el dataset de Kaggle y ubicarlo en `data/raw/`. Verificar
     integridad: cantidad de imágenes vs cantidad de archivos de anotación,
     que no falte ningún `.txt` por imagen.

1.3. EDA cuantitativo, documentado en `notebooks/01_eda.ipynb`:
   - Cantidad total de imágenes y de instancias (bounding boxes) por clase.
   - **[DECISIÓN] Granularidad de clases.** El dataset original distingue
     cultivo vs distintas malezas. Opciones:
     - **(a) Binaria** — "cultivo" vs "maleza" (todas las malezas juntas).
       Más simple, más robusta con pocos datos, pero menos informativa
       (no dice qué maleza es).
     - **(b) Multiclase original** — mantener las clases tal cual vienen
       en el dataset. Más informativo, pero con menos ejemplos por clase
       el modelo puede rendir peor.
     - **(c) Multiclase agrupada** — agrupar en 2-3 categorías intermedias
       si hay clases con muy pocos ejemplos. Punto medio.
   - Distribución de tamaño de las bounding boxes (objetos chicos son más
     difíciles de detectar — relevante porque malezas jóvenes son pequeñas
     en la imagen). Reportar en el notebook, no solo mencionar.
   - Resolución y aspect ratio de las imágenes originales.
   - Chequeo de duplicados o imágenes corruptas.
   - Visualización de una muestra con bounding boxes dibujados, para
     verificar a ojo que las anotaciones estén bien.

1.4. **[DECISIÓN] Estrategia de split train/val/test.** Opciones:
   - **(a) 70/20/10** — más datos para entrenar, menos para validar.
   - **(b) 80/10/10** — estándar más común en datasets chicos como este.
   - **(c) K-fold cross-validation** — más robusto estadísticamente con
     un dataset de solo ~1300 imágenes, pero más caro en tiempo de cómputo
     y más complejo de reportar en el README.

1.5. **[DECISIÓN] Manejo de desbalance de clases** (si el EDA del punto
     1.3 muestra que hay clases con muchos menos ejemplos que otras):
   - **(a) No corregir** — dejarlo así y reportarlo como limitación.
   - **(b) Oversampling** de las clases minoritarias.
   - **(c) Class weights** en la loss durante el entrenamiento (sin tocar
     el dataset físicamente).

### Fase 2 — Modelo baseline

2.1. **[DECISIÓN] Arquitectura base.** Opciones (ambas variante *nano*,
     pensadas para edge):
   - **(a) YOLOv8n** — más madura, más documentación y ejemplos
     disponibles, ecosistema de exportación (ONNX/TFLite/NCNN) muy probado.
   - **(b) YOLOv11n** — más nueva, en teoría mejor relación
     precisión/velocidad, pero menos "battle-tested" en export a formatos
     edge (más riesgo de encontrarte con bugs de compatibilidad).

2.2. **[DECISIÓN] Resolución de entrada (imgsz).** Afecta directo la
     latencia en edge:
   - **(a) 640x640** — resolución default de YOLO, mejor precisión,
     más lenta.
   - **(b) 416x416** — punto intermedio típico en proyectos edge.
   - **(c) 320x320** — más rápida, pero puede perder malezas chicas
     (relevante por lo visto en el EDA del punto 1.3).
   - Sugerencia de trabajo: entrenar el baseline en (a) primero para
     tener un techo de referencia, y comparar contra (b) o (c) recién en
     la Fase 3 cuando el foco pase a edge. Confirmar si te parece bien
     este orden o preferís ir directo a una resolución chica.

2.3. Transfer learning desde pesos preentrenados en COCO (no entrenar
     from scratch — con ~1300 imágenes no alcanza para converger bien
     desde cero). Esto no es una decisión abierta, es la práctica estándar.

2.4. Entrenamiento con métricas trackeadas por época: mAP50, mAP50-95,
     precision, recall. Guardar curvas de entrenamiento (loss, mAP) como
     imágenes versionadas, no solo logs de consola.

2.5. Matriz de confusión y AP por clase al final del baseline, para
     detectar si hay alguna clase que el modelo directamente no aprende
     (común con clases con pocos ejemplos).

2.6. **[DECISIÓN] Data augmentation.** YOLO trae augmentations por default
     (mosaic, flip, HSV shift, etc.). Opciones:
   - **(a) Dejar el default de Ultralytics** — rápido, ya viene tuneado
     para casos generales.
   - **(b) Ajustar manualmente** para simular condiciones reales de campo
     (más variación de brillo/contraste por luz solar directa, más
     rotación porque la cámara no siempre está perfectamente cenital).
     Requiere más tiempo de configuración pero puede ayudar a que el
     modelo generalice mejor fuera del dataset.

### Fase 3 — Optimización para edge

3.1. **[DECISIÓN] Hardware target de referencia.** Esto define qué formato
     de exportación tiene sentido, así que hay que definirlo antes de
     seguir:
   - **(a) Raspberry Pi 4/5** — solo CPU (ARM), sin GPU. Conviene exportar
     a ONNX Runtime o TFLite. Es el escenario más realista/barato para un
     proyecto de portfolio.
   - **(b) NVIDIA Jetson Nano/Orin Nano** — tiene GPU, se puede usar
     TensorRT para máxima velocidad. Más representativo de lo que usaría
     una empresa en producción a mayor escala, pero vos no tenés el
     hardware físico para probarlo (vas a tener que simular/estimar).
   - **(c) Genérico x86 de bajo consumo** (mini PC tipo Intel NUC) —
     opción intermedia, ONNX Runtime anda bien.
   - Nota: si no tenés ninguno de estos dispositivos físicamente, se puede
     simular la restricción de recursos en tu notebook (limitando threads/
     CPU) — igual conviene fijar la respuesta como "target de diseño" en
     el README aunque el benchmark real sea simulado.

3.2. Exportación del modelo baseline al formato elegido en 3.1 (ONNX como
     paso intermedio siempre, más el formato final específico).

3.3. **[DECISIÓN] Estrategia de cuantización.** Opciones:
   - **(a) Post-training quantization (PTQ) a INT8** — más simple y
     rápida de implementar, se aplica sobre el modelo ya entrenado.
   - **(b) Quantization-aware training (QAT)** — mejor retención de
     precisión, pero significa volver a entrenar con la cuantización
     simulada durante el training. Más trabajo, mejor resultado.
   - Sugerencia: arrancar con (a) porque es más rápido de validar, y
     evaluar (b) solo si la pérdida de mAP con PTQ es demasiado grande.
     Confirmar si te parece bien este criterio.

3.4. **[DECISIÓN] ¿Sumar pruning?** Reducir conexiones/canales redundantes
     del modelo además de cuantizar.
   - **(a) No** — mantener el proyecto enfocado en cuantización, que ya
     es una técnica de optimización sólida y suficiente para el alcance.
   - **(b) Sí** — sumar pruning estructurado como paso extra, más
     completo mostrar en el portfolio pero más tiempo de desarrollo.

3.5. Comparar en una tabla: modelo original vs cuantizado (vs podado si
     aplica 3.4) en tamaño (MB), mAP50, y mAP50-95. Este trade-off tiene
     que quedar explícito y visible, es parte central de la propuesta de
     valor del proyecto.

### Fase 4 — Benchmark de inferencia

4.1. Medir latencia promedio por frame (y FPS) para cada versión del
     modelo (baseline, cuantizado, y variantes de resolución si se
     probaron en 2.2), sobre el hardware target definido en 3.1 (real o
     simulado).

4.2. Reportar no solo el promedio sino variabilidad (min/max/percentil 95
     de latencia) — un solo número promedio esconde picos que en un
     sistema real de aplicación en campo importan.

4.3. Tabla final consolidada: modelo | tamaño | mAP50 | mAP50-95 |
     latencia promedio | FPS. Esta tabla es probablemente lo primero que
     va a mirar un técnico que revise el repo.

### Fase 5 — Demo

5.1. Script CLI (`src/inference.py`) que reciba una carpeta de imágenes y
     devuelva las mismas con bounding boxes dibujados + un CSV con las
     detecciones.

5.2. App Streamlit (`app/streamlit_app.py`):
   - Subida de imagen → detección → visualización con boxes y confianza.
   - **[DECISIÓN] ¿Sumar soporte de video, además de imagen estática?**
     - **(a) Solo imagen** — más simple, cubre el caso de uso principal.
     - **(b) Imagen + video** — más impresionante para un demo en vivo/
       entrevista, pero más trabajo (manejo de frames, posible latencia
       de UI).
   - Mostrar en la misma pantalla qué modelo se está usando (baseline vs
     cuantizado) para que en una demo se pueda mostrar el trade-off en
     vivo, no solo en una tabla del README.

### Fase 6 — Documentación final

6.1. README con: objetivo del proyecto, dataset usado (con la aclaración
     honesta de que es de sésamo, pensado como prueba de concepto
     extensible a soja/maíz), decisiones tomadas en cada [DECISIÓN] con
     una línea de justificación, tabla de resultados de la Fase 4,
     instrucciones para correr el demo, y limitaciones conocidas.

6.2. Sección explícita "por qué edge" conectando el proyecto con el caso
     de uso real de aplicación selectiva en campo (referencia a cómo
     operan sistemas tipo DeepAgro/SprAI).

6.3. Capturas de pantalla del demo funcionando, embebidas en el README
     (no solo descriptas en texto).

## 7. Criterios de éxito (para saber cuándo una fase está "terminada")

- Cada fase debe dejar algo corrible, no solo código a medio hacer.
- Las métricas (mAP, tamaño, latencia) tienen que quedar en un archivo o
  tabla versionada, no solo impresas en consola y perdidas.
- El README final tiene que poder leerlo un reclutador no técnico de RRHH
  y entender qué hace el proyecto y por qué importa en 30 segundos, y
  un técnico entender el detalle en 3 minutos.

## 8. Fuera de alcance (por ahora)

- No hace falta integrar con un dron o hardware físico real todavía.
- No hace falta desplegar en la nube — el foco es edge/local.
- No hace falta multi-clase de todas las malezas del dataset si complica
  el baseline; se puede arrancar binario (cultivo vs maleza) y escalar
  después si da el tiempo.