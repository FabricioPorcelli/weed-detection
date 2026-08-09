# AgroVision-Edge — Detección de malezas y plagas en cultivos

Sistema de detección de malezas/plagas en cultivos (foco inicial: soja, extensible a maíz) mediante visión por computadora con **YOLO**, optimizado para correr en **dispositivos edge** (Raspberry Pi, Jetson Nano/Orin) sin conexión a internet ni GPU de datacenter.

**Por qué edge importa:** la aplicación selectiva de agroquímicos en el campo (pulverizadores inteligentes, tipo DeepAgro/SprAI) ocurre en zonas sin conectividad. El modelo debe decidir en el dispositivo, en tiempo real: <200 ms por frame, <10–15 MB de peso, idealmente cuantizado a INT8.

## Estado del proyecto

- [x] Fase 1 — Setup + EDA: estructura, entorno, dataset, EDA, split 80/10/10 estratificado + `data.yaml` ✓
- [x] Fase 2 — Baseline: YOLOv8n @ 640, class weights, 30 épocas (mAP50=0.904, 6.0 MB) ✓
- [x] Fase 3 — Optimización edge: export ONNX FP32 + PTQ INT8, trade-off documentado ✓
- [ ] Fase 4 — Benchmark de inferencia: latencia por frame en hardware edge (o simulado)
- [ ] Fase 5 — Demo: CLI + app Streamlit
- [ ] Fase 6 — Documentación final

## Estructura del repo

```
weed-detection/
├── README.md                # objetivo, resultados, cómo correrlo, benchmarks
├── plan.md                  # brief del proyecto
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # dataset original (no versionado en git)
│   └── processed/           # splits train/val/test, YAML de config YOLO
├── notebooks/
│   ├── 01_eda.ipynb         # exploración del dataset (sin outputs commiteados; figuras en figures/)
│   ├── build_eda.py         # genera la notebook de EDA
│   └── figures/             # figuras versionadas del EDA
├── src/
│   ├── download_dataset.py  # descarga del dataset desde Kaggle
│   ├── make_splits.py       # split 80/10/10 estratificado + data.yaml
│   ├── train.py             # entrenamiento (Fase 2)
│   ├── export.py            # exportación a ONNX (FP32 + INT8 PTQ) (Fase 3)
│   ├── validate_onnx.py     # validación de mAP de modelos ONNX + tabla comparativa (Fase 3)
│   ├── quantize.py          # cuantización INT8 (pendiente)
│   ├── benchmark.py         # latencia, tamaño, mAP (pendiente)
│   └── inference.py         # inferencia standalone CLI (pendiente)
├── app/
│   └── streamlit_app.py     # demo visual (pendiente)
└── models/                  # pesos entrenados (no versionado en git)
```

## Setup

### 1. Entorno (conda, Python 3.10)

```bash
conda create -n agrovision python=3.10 -y
conda activate agrovision
pip install -r requirements.txt
```

### 2. Dataset

Fuente: [Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) (Kaggle, ~1300 imágenes de cultivo de sésamo con malezas, anotadas en formato YOLO).

Se necesita una cuenta de Kaggle:

1. Ir a https://www.kaggle.com/settings → *API* → **Create New Token** (descarga `kaggle.json`).
2. Ubicarlo en `~/.kaggle/kaggle.json`.
3. Descargar el dataset:

```bash
python src/download_dataset.py          # descarga y descomprime en data/raw/
python src/download_dataset.py --overwrite   # re-descarga si ya existe
```

Alternativa manual: descargar el zip desde la página de Kaggle y descomprimirlo en `data/raw/`.

## Resultados

### Baseline (Fase 2) — YOLOv8n @ 640

Entrenamiento en CPU (30 épocas, 1.49 h). Métricas y curvas versionadas en `reports/baseline/`, pesos en `models/baseline_best.pt` (no versionados).

| Métrica | Valor |
|---|---|
| mAP50 | **0.904** |
| mAP50-95 | 0.608 |
| Precision | 0.905 |
| Recall | 0.820 |
| Tamaño `best.pt` | **6.0 MB** |
| Parámetros | 3.0 M · 8.1 GFLOPs (fused) |

| Clase | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| crop | 0.800 | 0.826 | 0.877 | 0.617 |
| weed | 0.911 | 0.853 | 0.930 | 0.602 |

- `weed` (minoritaria) rinde mejor que `crop` → los **class weights** (`cls_pw=0.7`) equilibraron la pérdida sin dañar la clase mayoritaria.
- Convergencia rápida (mAP50 > 0.80 en época 6), consistente con el EDA (dataset mono-clase + boxes grandes).
- **6.0 MB ya cumple el target de tamaño edge** (<10–15 MB) sin cuantizar; la Fase 3 lo reducirá más vía INT8.
- Ver `reports/baseline/summary.md` para el análisis completo y la evolución por época.

## Optimización edge (Fase 3)

**Decisiones:**
- **3.1 Hardware target de diseño:** NVIDIA Jetson (Orin) → TensorRT en producción.
- **3.3 Cuantización:** Post-Training Quantization (PTQ) a INT8 sobre ONNX.
- **3.4 Pruning:** no (cuantización es suficiente para el alcance).

### Trade-off tamaño vs mAP (3.5)

Validado sobre el split `val` con `onnxruntime` (CPU). Ver `reports/optimization/comparison.csv` y `comparison.md`.

| modelo | formato | tamaño (MB) | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| baseline | `.pt` FP32 | 6.25 | **0.9034** | **0.6093** | 0.8556 | 0.8394 |
| baseline | `.onnx` FP32 | 12.37 | 0.9034 | 0.6093 | 0.8556 | 0.8394 |
| baseline | `.onnx` INT8 (PTQ) | **3.60** | 0.8850 | 0.5956 | 0.8672 | 0.8199 |

- **Onnx FP32 = .pt en mAP** (misma red, distinta serialización; el .onnx pesa más porque no está comprimido como torch.save).
- **PTQ INT8 pierde 1.84 pts de mAP50** (0.903 → 0.885) y 1.37 pts de mAP50-95 — dentro del rango aceptable (< 5-7 pts), **no requiere QAT**.
- **Recorte de tamaño: 6.25 → 3.60 MB = -42%** (vs el `.pt` original). El `.onnx` INT8 es **40% más liviano** y ya está en formato portable para ONNX Runtime / TensorRT.
- La inferencia INT8 en CPU mostró latencia similar a la FP32 (~60-70 ms/preprocess+inference en este CPU x86); el benchmark real de Jetson se completa en la Fase 4.

### Exportación + validación

```bash
python src/export.py                 # genera models/baseline.{onnx,_int8.onnx} desde best.pt
python src/validate_onnx.py           # valida .pt + .onnx + .onnx_int8 sobre val -> reports/optimization/
python src/validate_onnx.py --split test   # sobre test en vez de val
```

### Limitación conocida (TensorRT)

El **target de diseño es Jetson** pero no se dispone del hardware físico. Por eso:
- No se buildea el **engine de TensorRT** acá: los engines no son portables entre GPUs (se generan para el SM específico de la Jetson target), y no se dispone de hardware Jetson físico para generarlo ni medir su latencia real.
- El **path ONNX es el entregable portable** y benchmarkable en CPU acá; buildar el engine de TensorRT sería el paso correspondiente cuando se disponga de una Jetson, y queda como **TODO posterior**.
- El **benchmark real de latencia en Jetson** se documenta como limitación en la Fase 4, con sustitución por benchmark en CPU x86 limitando threads para recursos acotados.

## EDA — Hallazgos (Fase 1.3)

Ver `notebooks/01_eda.ipynb` (notebook re-ejecutable; figuras versionadas en `notebooks/figures/`). Para regenerar outputs:

```bash
python notebooks/build_eda.py        # reconstruye la notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

| Métrica | Valor |
|---|---|
| Imágenes totales | 1.300 |
| Resolución | 512×512 (uniforme, aspect 1:1) |
| Instancias totales | 2.072 |
| Bboxes/imagen (promedio) | 1.59 |
| crop (id 0) | 1.212 instancias · 635 imágenes |
| weed (id 1) | 860 instancias · 667 imágenes |
| Desbalance | 1.41:1 (crop mayoritario) |
| Imágenes con AMBAS clases | 2 (≈ mono-clase por imagen) |
| Objetos small (<32×32 px) | 29 / 2.072 = **1.40%** (crop 2.39%, weed 0%) |
| Duplicados / corruptas | 0 / 0 |

**Decisiones tomadas (Fase 1):**
- **1.3 Granularidad:** binaria `crop`/`weed` — el dataset no expone sub-clases de malezas; única opción realista.
- **1.4 Split:** 80/10/10 estratificado (1040 / 130 / 130).
- **1.5 Desbalance:** leve (1.41:1) → se aplicará **class weights** en la loss de Ultralytics (sin tocar el dataset físico).

**Observación clave:** cada imagen es prácticamente mono-clase (solo 2 imgs con ambas clases), con 1–2 boxes grandes que cubren casi toda la imagen. El dataset se comporta más como *clasificación a nivel imagen con box grosero* que como detección densa de objetos pequeños. El baseline debería aprender fácil; el desafío real de generalización vendrá con malezas jóvenes/pequeñas en campo (no representadas aquí, 98.6% de boxes son grandes). Esto también baja el riesgo de reducir `imgsz` en la Fase 3.

## Splits train/val/test (Fase 1.4)

Generados con `src/make_splits.py` — **80/10/10 estratificado** por composición de clases de cada imagen (stratum: `crop` / `weed` / `both`), seed 42, reproducible.

| Split | Imágenes | crop | weed | both |
|---|---|---|---|---|
| train | 1040 | 506 | 532 | 2 |
| val   | 129  | 63  | 66  | 0 |
| test  | 131  | 64  | 67  | 0 |
| **total** | **1300** | 633 | 665 | 2 |

Salida en `data/processed/` (formato Ultralytics YOLO): `images/{train,val,test}/`, `labels/{train,val,test}/` y `data.yaml` con clases `{0: crop, 1: weed}`. El script es dataset-agnostic (lee `classes.txt`), así se puede swappear el dataset sin reescribir.

```bash
python src/make_splits.py               # copia imgs+labels a data/processed/
python src/make_splits.py --symlink      # symlinks en vez de copiar
```

## Entrenamiento baseline (Fase 2)

Decisiones: arquitectura **YOLOv8n** (preentrenada COCO, transfer learning), `imgsz=640`, augmentation default de Ultralytics, y **class weights** (`cls_pw=0.7`) para el desbalance leve crop/weed. Métricas y curvas se versionan en `reports/<name>/` (CSV, confusion matrix, curvas P/R/F1/PR, `summary.md`); los pesos `.pt` quedan en `models/` (no versionados).

```bash
# CPU (default, ~3.3 min/epoch en Ryzen moderno)
python src/train.py --epochs 30 --patience 12 --name baseline

# GPU CUDA — mucho más rápido (requiere torch con soporte CUDA y GPU NVIDIA)
python src/train.py --epochs 30 --patience 12 --name baseline --device 0

# smoke test rápido (2 épocas) para validar el pipeline
python src/train.py --epochs 2 --name smoke
```

> **torch + CUDA:** `pip install -r requirements.txt` instala un build de torch con CUDA embebido (wheels recientes). En una PC NVIDIA con driver reciente debería detectar la GPU directamente (`torch.cuda.is_available()` → True). Si no, instalar el wheel de torch con el índice CUDA que corresponda al driver.
