# Resumen del entrenamiento — baseline

- **Run:** `baseline` · YOLOv8n (preentrenada COCO, transfer learning)
- **Hardware:** CPU AMD Ryzen AI 5 340 (sin GPU usable) · torch 2.13.0
- **Épocas ejecutadas:** 30 (no hubo early stopping; patience=12 no se disparó)
- **Tiempo total:** 1.49 h (~3 min/epoch)
- **imgsz:** 640 · **augmentation:** default Ultralytics · **class weights:** `cls_pw=0.7` → `[crop=0.884, weed=1.116]`

## Métricas (validación, mejor epoch = 29)

| Métrica | Valor |
|---|---|
| mAP50 | **0.9039** |
| mAP50-95 | **0.6079** |
| Precision | 0.9047 |
| Recall | 0.8201 |

Validación con `best.pt` (re-avaluada al final del entrenamiento):

| Clase | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all  | 129 | 199 | 0.856 | 0.839 | 0.903 | 0.609 |
| crop | 63  | 115 | 0.800 | 0.826 | 0.877 | 0.617 |
| weed | 66  | 84  | 0.911 | 0.853 | 0.930 | 0.602 |

## Evolución del mAP50 por época

```
ep01 0.501   ep11 0.814   ep21 0.855
ep02 0.372 ↓ ep12 0.806   ep22 0.873
ep03 0.661   ep13 0.785 ↓ ep23 0.874
ep04 0.430 ↓ ep14 0.845   ep24 0.896
ep05 0.702   ep15 0.849   ep25 0.894
ep06 0.825   ep16 0.841   ep26 0.898
ep07 0.802   ep17 0.853   ep27 0.881 ↓
ep08 0.808   ep18 0.857   ep28 0.898
ep09 0.804   ep19 0.841   ep29 0.904 ★
ep10 0.841   ep20 0.856   ep30 0.903
```

## Tamaño del modelo

- `best.pt`: **6.0 MB** (3.0 M params, 8.1 GFLOPs en modo fused)

## Observaciones

- Convergencia **rápida** (mAP50 > 0.80 ya en época 6) — consistente con el EDA: dataset mayormente mono-clase por imagen, boxes grandes que cubren casi toda la imagen.
- **Oscilación alta en las primeras épocas** (0.50 → 0.37 → 0.66 → 0.43) por lr alto + few-shot + class weights; se estabiliza después de época ~10.
- `weed` rinde marginalmente mejor (mAP50 0.930) que `crop` (0.877) a pesar de ser la clase minoritaria → **los class weights funcionaron**, equilibrando la pérdida sin dañar la clase mayoritaria.
- El gap mAP50 (0.904) vs mAP50-95 (0.608) indica que las cajas están bien localizadas a IoU 0.5 pero decaen a IoUs más estrictos: las boxes son grandes y "groseras" (cobran casi toda la imagen),_hay margen de ajuste fino.
- No se disparó early stopping: en época 30 el mAP50 seguía en su máximo. Queda pendiente evaluar si más épocas aportan ganancia marginal o si es ruido.
- **El modelo ya cumple el criterio de tamaño** para edge (<10–15 MB): 6.0 MB sin cuantizar. La Fase 3 debería reducirlo más vía INT8.