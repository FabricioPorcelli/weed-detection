#!/usr/bin/env python3
"""Entrenamiento del modelo baseline — Fase 2.

Decisiones de la Fase 2:
  - 2.1 Arquitectura: YOLOv8n (preentrenado en COCO -> transfer learning, ver 2.3)
  - 2.2 Resolución: imgsz 640 (techo de referencia; comparar 416/320 en Fase 3)
  - 2.6 Augmentation: default de Ultralytics
  - 1.5 Desbalance: class weights via `cls_pw` (inverse-freq normalizado a media 1)

Métricas por época versionadas en reports/<name>/: results.csv + curvas
generadas por Ultralytics (results.png, confusion_matrix.png, PR/F1/P/R curves).

Uso:
  python src/train.py --epochs 50 --name baseline
  python src/train.py --epochs 2 --name smoke   # smoke test rápido
  python src/train.py --imgsz 416 --name baseline_416
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
REPORTS_DIR = ROOT / "reports"

ARTIFACTS = [
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "args.yaml",
]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8n.pt", help="pesos iniciales (default COCO nano)")
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", default=-1, help="-1 = auto-batch")
    p.add_argument("--cls-pw", type=float, default=0.7,
                   help="class weights power (0=off, 1=pura inverse-freq). 0.7 amortigua leve.")
    p.add_argument("--patience", type=int, default=15, help="early stopping")
    p.add_argument("--name", default="baseline", help="nombre del run (= carpeta en runs/detect/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu o 0 (gpu)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--exist-ok", action="store_true", help="sobreescribir run existente")
    return p.parse_args()


def version_artifacts(run_dir: Path, name: str):
    """Copia métricas y curvas a reports/<name>/ (versionado, no gitignored)."""
    out = REPORTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for art in ARTIFACTS:
        src = run_dir / art
        if src.exists():
            shutil.copy2(src, out / art)
            copied.append(art)
    # copiar pesos a models/ (no versionado en git, pero útil localmente)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    for w in ("best.pt", "last.pt"):
        src = run_dir / w
        if src.exists():
            shutil.copy2(src, models_dir / f"{name}_{w}")
    return out, copied


def summarize(run_dir: Path, out: Path):
    """Genera reports/<name>/summary.md con tabla de métricas finales."""
    import csv
    csv_path = run_dir / "results.csv"
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    last = rows[-1]
    best_map = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)
    md = ["# Resumen del entrenamiento\n",
          f"- Run: `{run_dir.name}`",
          f"- Épocas ejecutadas: {len(rows)}",
          f"- Mejor mAP50 ≈ {best_map:.4f}\n",
          "## Métricas de la última época\n",
          "| mAP50 | mAP50-95 | Precision | Recall |",
          "|---|---|---|---|",
          f"| {float(last.get('metrics/mAP50(B)',0)):.4f} "
          f"| {float(last.get('metrics/mAP50-95(B)',0)):.4f} "
          f"| {float(last.get('metrics/precision(B)',0)):.4f} "
          f"| {float(last.get('metrics/recall(B)',0)):.4f} |\n",
          "## Tamaño del modelo\n"]
    best_pt = run_dir / "best.pt"
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / 1e6
        md.append(f"- `best.pt`: **{size_mb:.2f} MB**\n")
    (out / "summary.md").write_text("\n".join(md))


def main():
    args = get_args()
    if not Path(args.data).exists():
        raise SystemExit(f"No existe {args.data}. Ejecutá src/make_splits.py primero.")

    from ultralytics import YOLO
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cls_pw=args.cls_pw,
        patience=args.patience,
        name=args.name,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        exist_ok=args.exist_ok,
        project=str(ROOT / "runs" / "detect"),
        verbose=True,
    )

    # run_dir = runs/detect/<name>
    run_dir = ROOT / "runs" / "detect" / args.name
    out, copied = version_artifacts(run_dir, args.name)
    summarize(run_dir, out)
    print(f"\nArtefactos versionados en: {out}")
    print(" Copiados:", ", ".join(copied))


if __name__ == "__main__":
    main()