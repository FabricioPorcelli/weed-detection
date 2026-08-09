# Benchmark de inferencia — Fase 4

Split: `test` · imgsz: 640 · iters: 3 · warmup: 5 frames · device: cpu

## Tabla consolidada (4.3)

| modelo | formato | tamaño (MB) | mAP50 | mAP50-95 | modo | threads | lat. mean (ms) | lat. p95 (ms) | lat. max (ms) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_best.pt | pt | 6.25 | 0.9034 | 0.6093 | full | 12 | 41.61 | 46.23 | 96.65 | 24.03 |
| baseline_best.pt | pt | 6.25 | 0.9034 | 0.6093 | edge | 2 | 40.29 | 43.12 | 50.01 | 24.82 |
| baseline.onnx | onnx | 12.37 | 0.9034 | 0.6093 | full | 12 | 75.83 | 103.38 | 138.18 | 13.19 |
| baseline.onnx | onnx | 12.37 | 0.9034 | 0.6093 | edge | 2 | 74.46 | 101.56 | 114.69 | 13.43 |
| baseline_int8.onnx | onnx | 3.60 | 0.8850 | 0.5956 | full | 12 | 86.21 | 120.06 | 170.66 | 11.60 |
| baseline_int8.onnx | onnx | 3.60 | 0.8850 | 0.5956 | edge | 2 | 86.75 | 117.91 | 151.32 | 11.53 |

## Notas (4.2 — variabilidad)
- Se reporta min/mean/median/p95/max/std además del promedio: un solo número promedio esconde picos relevantes en un sistema real de aplicación en campo.
- **modo `full`** = todos los threads disponibles (referencia del hardware de desarrollo).
- **modo `edge`** = threads limitados a 2 (simulación de restricción edge; target de diseño Jetson, sin hardware físico -> benchmark real pendiente).

## Lectura de los resultados (trade-off honesto)
- **En CPU x86 de desarrollo, `.pt` (PyTorch nativo) es el más rápido** (~24 FPS), porque PyTorch usa un path de inferencia altamente optimizado para CPU.
- **ONNX Runtime en CPU no acelera la inferencia** respecto a PyTorch, e incluso **INT8 es levemente más lento que FP32** (~12 vs ~13 FPS). Esto es esperable: la ventaja de INT8 se manifiesta en hardware edge con soporte INT8 nativo (Jetson/TensorRT, ARM con NEON/NNAPI, NPU), no en CPU x86 de escritorio. El cuantizado acá gana en **tamaño** (3.6 vs 12.4 MB) y en **porteabilidad edge**, no en latencia sobre esta CPU.
- La latencia `full` vs `edge` es casi idéntica → a batch=1 estos modelos pequeños no son CPU-bound; el cuello de botella es el acceso a memoria y el overhead del runtime, no los cores. Limitar threads sirve como proxy de "restricción edge" pero no reproduce fielmente un Jetson.

## Limitación conocida
- El target de diseño es NVIDIA Jetson (Orin). No se dispone del hardware físico, por lo que el benchmark real de TensorRT sobre Jetson queda como limitación documentada. Los números aquí son de CPU x86 con threads acotados como **aproximación conservadora** del régimen edge; una Jetson con TensorRT sería sustancialmente más rápida en inferencia (GPU + engine INT8 optimizado), por lo que estos valores son un **techo superior de latencia**, no una predicción de Jetson.
- **Conclusión honesta:** el benchmark acá valida que el pipeline de inferencia corre y mide variabilidad, pero **no demuestra el speedup de INT8** que es uno de los argumentos centrales del proyecto. Para validarlo hace falta el hardware target (Jetson) o, como paso posterior opcional, buildar el engine de TensorRT en una GPU NVIDIA disponible y medir INT8 vs FP32 ahí.
