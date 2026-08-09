#!/usr/bin/env python3
"""Exportación del modelo baseline a formatos edge — Fase 3.

Decisiones de la Fase 3:
  - 3.1 Hardware target de diseño: NVIDIA Jetson (Orin) — TensorRT en producción.
    Limitación conocida: sin hardware físico -> TensorRT engine queda como target
    de diseño documentado (no se builda ni benchmarkea acá). Path ONNX es el
    entregable portable y benchmarkable en CPU.
  - 3.3 Cuantización: Post-Training Quantization (PTQ) a INT8 sobre ONNX.
  - 3.4 Sin pruning.

Outputs (en models/):
  - baseline.onnx          (FP32, opset 12, batch dinamico)
  - baseline_int8.onnx     (INT8 PTQ, calibrado con split val)

Uso:
  python src/export.py                                # export FP32 + INT8
  python src/export.py --model models/baseline_best.pt
  python src/export.py --format onnx                 # solo FP32
  python src/export.py --format onnx-int8            # solo INT8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "optimization"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(MODELS_DIR / "baseline_best.pt"),
                   help=" pesos .pt a exportar (default: models/baseline_best.pt)")
    p.add_argument("--data", default=str(DATA_YAML),
                   help=" YAML del dataset (para calibración INT8)")
    p.add_argument("--format", choices=["onnx", "onnx-int8", "all"], default="all",
                   help=" qué exportar (default: all = FP32 + INT8)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--calib-frac", type=float, default=1.0,
                   help=" fracción de val usada para calibrar INT8 (0-1)")
    return p.parse_args()


def _move_artifact(path: Path, dst: Path) -> Path:
    """Mueve el artifact exportado a models/ con nombre canonico."""
    if not path.exists():
        raise FileNotFoundError(f"No se generó {path}")
    MODELS_DIR.mkdir(exist_ok=True)
    if path.resolve() != dst.resolve():
        if dst.exists():
            dst.unlink()
        shutil.move(str(path), str(dst))
    return dst


def export_fp32(model_path: str, imgsz: int, opset: int) -> Path:
    """Exporta a ONNX FP32 con batch dinamico, opset dado."""
    from ultralytics import YOLO
    m = YOLO(model_path)
    out = m.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=True,
                    simplify=True, half=False)
    src = Path(out)
    dst = MODELS_DIR / "baseline.onnx"
    return _move_artifact(src, dst)


def export_int8(model_path: str, data_yaml: str, imgsz: int, opset: int, frac: float) -> Path:
    """Exporta a ONNX INT8 (PTQ) usando Ultralytics (calibración sobre val).

    Ultralytics genera baseline_int8.onnx y borra el FP32 auxiliar. Lo renombramos
    a models/baseline_int8.onnx.
    """
    from ultralytics import YOLO
    m = YOLO(model_path)
    out = m.export(format="onnx", quantize=8, data=data_yaml, imgsz=imgsz,
                    opset=opset, dynamic=True, simplify=True, fraction=frac)
    src = Path(out)  # ya es *_int8.onnx
    dst = MODELS_DIR / "baseline_int8.onnx"
    return _move_artifact(src, dst)


def report_sizes(models: dict[str, Path]):
    """Imprime tamaños por consola (el CSV consolidado lo arma validate_onnx.py)."""
    print("\n--- Tamaños ---")
    for name, p in models.items():
        if p.exists():
            sz = p.stat().st_size
            print(f"  {name}: {sz/1e6:.3f} MB ({sz:,} bytes)")
    print(f"\nTabla comparativa (tamaño + mAP): correr `python src/validate_onnx.py`")


def main():
    args = get_args()
    if not Path(args.model).exists():
        raise SystemExit(f"No existe {args.model}. Entrená primero (Fase 2).")
    if not Path(args.data).exists():
        raise SystemExit(f"No existe {args.data}. Ejecutá src/make_splits.py primero.")

    produced: dict[str, Path] = {}
    if args.format in ("onnx", "all"):
        print("=== Exportando ONNX FP32 ===")
        produced["baseline.onnx (FP32)"] = export_fp32(args.model, args.imgsz, args.opset)
    if args.format in ("onnx-int8", "all"):
        print("\n=== Exportando ONNX INT8 (PTQ, calibración sobre val) ===")
        produced["baseline_int8.onnx (INT8 PTQ)"] = export_int8(
            args.model, args.data, args.imgsz, args.opset, args.calib_frac)

    report_sizes(produced)
    print("\nListo. Artefactos en models/.")


if __name__ == "__main__":
    main()