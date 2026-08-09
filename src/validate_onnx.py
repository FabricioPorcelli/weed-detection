#!/usr/bin/env python3
"""Valida modelos ONNX (FP32 / INT8) sobre el split val y arma la tabla
comparativa de la Fase 3 -- 3.5 (tamaño + mAP), parte de Precision/Recall/mAP.

Calcula mAP50 y mAP50-95 por inferencia onnxruntime (CPU), reproducible.

Uso:
  python src/validate_onnx.py                 # valida .pt + .onnx + .onnx_int8 vs val
  python src/validate_onnx.py --split test    # sobre test en vez de val
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "optimization"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1, help="batch de inferencia (ONNX en CPU)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def model_row(name: str, path: Path, args) -> dict:
    """Carga el modelo (pt u onnx) y corre val() -> dict de métricas."""
    from ultralytics import YOLO
    if not path.exists():
        print(f"  [skip] {name}: no existe {path}")
        return {"modelo": name, "archivo": str(path), "tamaño_MB": "NA",
                "mAP50": "NA", "mAP50-95": "NA", "precision": "NA", "recall": "NA"}
    size_mb = path.stat().st_size / 1e6
    is_onnx = path.suffix == ".onnx"
    kwargs = dict(data=args.data, split=args.split, imgsz=args.imgsz,
                  batch=args.batch, device=args.device, verbose=False)
    if is_onnx:
        model = YOLO(str(path), task="detect")
    else:
        model = YOLO(str(path))
    print(f"  Validando {name} ({path.name}, {size_mb:.2f} MB)...")
    res = model.val(**kwargs)
    # `res` puede ser lista (pt) u objeto (onnx); .box expone métricas escalares
    box = res[0].box if isinstance(res, list) else res.box
    return {
        "modelo": name,
        "archivo": str(path),
        "tamaño_MB": f"{size_mb:.3f}",
        "mAP50": f"{box.map50:.4f}",
        "mAP50-95": f"{box.map:.4f}",
        "precision": f"{box.mp:.4f}",
        "recall": f"{box.mr:.4f}",
    }


def main():
    args = get_args()
    if not Path(args.data).exists():
        raise SystemExit(f"No existe {args.data}. Ejecutá src/make_splits.py primero.")

    targets = [
        ("baseline.pt (FP32)", MODELS_DIR / "baseline_best.pt"),
        ("baseline.onnx (FP32)", MODELS_DIR / "baseline.onnx"),
        ("baseline_int8.onnx (INT8 PTQ)", MODELS_DIR / "baseline_int8.onnx"),
    ]
    print(f"=== Validación sobre split '{args.split}' (imgsz={args.imgsz}, device={args.device}) ===")
    rows = []
    for name, p in targets:
        rows.append(model_row(name, p, args))

    # CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "comparison.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown tabla (sin latencia, se completa en Fase 4)
    md = ["# Comparación de modelos — Fase 3 (3.5)\n",
          f"Split: `{args.split}` · imgsz: {args.imgsz} · device: {args.device}\n",
          "| modelo | tamaño (MB) | mAP50 | mAP50-95 | Precision | Recall |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['modelo']} | {r['tamaño_MB']} | {r['mAP50']} | "
                   f"{r['mAP50-95']} | {r['precision']} | {r['recall']} |")
    md.append("\n_Latencia por frame se completa en la Fase 4._")
    md_path = REPORTS_DIR / "comparison.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"\n=== Tabla comparativa ===")
    for line in md:
        print(line)
    print(f"\nGuardado: {csv_path}\n          {md_path}")


if __name__ == "__main__":
    main()