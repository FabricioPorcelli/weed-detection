# Comparación de modelos — Fase 3 (3.5)

Split: `val` · imgsz: 640 · device: cpu

| modelo | tamaño (MB) | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| baseline.pt (FP32) | 6.248 | 0.9034 | 0.6093 | 0.8556 | 0.8394 |
| baseline.onnx (FP32) | 12.368 | 0.9034 | 0.6093 | 0.8556 | 0.8394 |
| baseline_int8.onnx (INT8 PTQ) | 3.602 | 0.8850 | 0.5956 | 0.8672 | 0.8199 |

_Latencia por frame se completa en la Fase 4._
