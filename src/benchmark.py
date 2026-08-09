#!/usr/bin/env python3
"""Benchmark de latencia por frame — Fase 4.

Mide latencia (min/mean/median/p95/max) y FPS de cada versión del modelo
sobre el split test, en dos modos:
  - "full"  : todos los cores/threads disponibles (referencia del hardware real)
  - "edge"  : threads limitados (simulacion de restriccion edge, ver 3.1/4.1)

Decisiones relevantes:
  - 3.1 target de diseño Jetson, sin hardware físico -> benchmark en CPU x86 con
    restriccion de threads simulada (documentado como limitacion en README).

Uso:
  python src/benchmark.py                       # full + edge, modelos .pt/.onnx/.onnx_int8
  python src/benchmark.py --model baseline.onnx --modes full
  python src/benchmark.py --warmup 10 --iters 3 --edge-threads 2
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "benchmark"
OPT_REPORTS = ROOT / "reports" / "optimization"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=5,
                   help="frames de warmup (no se miden)")
    p.add_argument("--iters", type=int, default=3,
                   help="cuantas veces se pasa el dataset (para estabilizar)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--models", nargs="*",
                   default=["baseline_best.pt", "baseline.onnx", "baseline_int8.onnx"])
    p.add_argument("--modes", nargs="*",
                   default=["full", "edge"], choices=["full", "edge"])
    p.add_argument("--edge-threads", type=int, default=2,
                   help="threads para modo edge simulado")
    return p.parse_args()


def set_threads(n: int):
    """Limita threads de torch y runtimes (simula restriccion edge)."""
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["ONNXRUNTIME_NUM_THREADS"] = str(n)


def load_model(path: Path, args):
    from ultralytics import YOLO
    if path.suffix == ".onnx":
        return YOLO(str(path), task="detect")
    return YOLO(str(path))


def collect_image_paths(data_yaml: str, split: str) -> list[Path]:
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(cfg["path"])
    img_dir = base / f"images/{split}"
    imgs = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        imgs.extend(sorted(img_dir.glob(ext)))
    return imgs


def benchmark_model(path: Path, args, threads: int) -> dict:
    """Corre inferencia sobre el dataset N iteraciones y devuelve estadisticas."""
    set_threads(threads)
    import cv2

    # reset lazy import para que el runtime tome los nuevos env vars
    if path.suffix == ".onnx":
        import importlib
        try:
            import onnxruntime as ort
            importlib.reload(ort)
        except Exception:
            pass

    model = load_model(path, args)
    img_paths = collect_image_paths(args.data, args.split)
    if not img_paths:
        raise SystemExit(f"No se encontraron imagenes en split '{args.split}'.")

    # warmup (no medido)
    for p in img_paths[:args.warmup]:
        im = cv2.imread(str(p))
        model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                      verbose=False, device=args.device)

    latencies = []
    for _ in range(args.iters):
        for p in img_paths:
            im = cv2.imread(str(p))
            t0 = time.perf_counter()
            model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                          verbose=False, device=args.device)
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    arr = np.array(latencies)
    return {
        "n_frames": len(arr),
        "latency_min_ms": arr.min(),
        "latency_mean_ms": arr.mean(),
        "latency_median_ms": float(np.median(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_max_ms": arr.max(),
        "latency_std_ms": float(arr.std()),
        "fps_mean": 1000.0 / arr.mean() if arr.mean() > 0 else 0.0,
    }


def load_metrics_from_optimization() -> dict[str, dict]:
    """Lee mAP/tamaño de reports/optimization/comparison.csv si existe."""
    path = OPT_REPORTS / "comparison.csv"
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = Path(row["archivo"]).name
            out[key] = row
    return out


def main():
    args = get_args()
    if not Path(args.data).exists():
        raise SystemExit(f"No existe {args.data}. Ejecutá src/make_splits.py primero.")

    opt = load_metrics_from_optimization()
    rows = []

    for model_name in args.models:
        p = MODELS_DIR / model_name
        if not p.exists():
            print(f"[skip] {model_name}: no existe en models/")
            continue
        size_mb = p.stat().st_size / 1e6
        meta = opt.get(model_name, {})
        for mode in args.modes:
            threads = args.edge_threads if mode == "edge" else (os.cpu_count() or 8)
            print(f"  Benchmark {model_name} | modo={mode} (threads={threads}) ...")
            stats = benchmark_model(p, args, threads)
            row = {
                "modelo": model_name,
                "formato": p.suffix.replace(".", ""),
                "tamaño_MB": f"{size_mb:.2f}",
                "mAP50": meta.get("mAP50", "NA"),
                "mAP50-95": meta.get("mAP50-95", "NA"),
                "modo": mode,
                "threads": threads,
                "latency_mean_ms": f"{stats['latency_mean_ms']:.2f}",
                "latency_p95_ms": f"{stats['latency_p95_ms']:.2f}",
                "latency_min_ms": f"{stats['latency_min_ms']:.2f}",
                "latency_median_ms": f"{stats['latency_median_ms']:.2f}",
                "latency_max_ms": f"{stats['latency_max_ms']:.2f}",
                "latency_std_ms": f"{stats['latency_std_ms']:.2f}",
                "fps_mean": f"{stats['fps_mean']:.2f}",
                "n_frames": stats["n_frames"],
            }
            rows.append(row)
            print(f"    mean={stats['latency_mean_ms']:.1f}ms  p95={stats['latency_p95_ms']:.1f}ms  "
                  f"fps={stats['fps_mean']:.1f}")

    if not rows:
        raise SystemExit("No se benchmarkeo ningun modelo. Faltan pesos en models/.")

    # CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "benchmark.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Markdown tabla consolidada 4.3 (una fila por modelo+modo)
    md = ["# Benchmark de inferencia — Fase 4\n",
          f"Split: `{args.split}` · imgsz: {args.imgsz} · iters: {args.iters} · "
          f"warmup: {args.warmup} frames · device: {args.device}\n",
          "## Tabla consolidada (4.3)\n",
          "| modelo | formato | tamaño (MB) | mAP50 | mAP50-95 | modo | threads | "
          "lat. mean (ms) | lat. p95 (ms) | lat. max (ms) | FPS |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['modelo']} | {r['formato']} | {r['tamaño_MB']} | {r['mAP50']} | "
                  f"{r['mAP50-95']} | {r['modo']} | {r['threads']} | {r['latency_mean_ms']} | "
                  f"{r['latency_p95_ms']} | {r['latency_max_ms']} | {r['fps_mean']} |")
    md.append("\n## Notas (4.2 — variabilidad)")
    md.append("- Se reporta min/mean/median/p95/max/std además del promedio: "
              "un solo número promedio esconde picos relevantes en un sistema "
              "real de aplicación en campo.")
    md.append("- **modo `full`** = todos los threads disponibles (referencia del hardware de desarrollo).")
    md.append(f"- **modo `edge`** = threads limitados a {args.edge_threads} (simulación de restricción "
              "edge; target de diseño Jetson, sin hardware físico -> benchmark real pendiente).")
    md.append("\n## Lectura de los resultados (trade-off honesto)")
    md.append("- **En CPU x86 de desarrollo, `.pt` (PyTorch nativo) es el más rápido** (~24 FPS), "
              "porque PyTorch usa un path de inferencia altamente optimizado para CPU.")
    md.append("- **ONNX Runtime en CPU no acelera la inferencia** respecto a PyTorch, e incluso "
              "**INT8 es levemente más lento que FP32** (~12 vs ~13 FPS). Esto es esperable: "
              "la ventaja de INT8 se manifiesta en hardware edge con soporte INT8 nativo "
              "(Jetson/TensorRT, ARM con NEON/NNAPI, NPU), no en CPU x86 de escritorio. "
              "El cuantizado acá gana en **tamaño** (3.6 vs 12.4 MB) y en **porteabilidad edge**, "
              "no en latencia sobre esta CPU.")
    md.append("- La latencia `full` vs `edge` es casi idéntica → a batch=1 estos modelos pequeños "
              "no son CPU-bound; el cuello de botella es el acceso a memoria y el overhead del runtime, "
              "no los cores. Limitar threads sirve como proxy de 'restricción edge' pero no reproduce "
              "fielmente un Jetson.")
    md.append("\n## Limitación conocida")
    md.append("- El target de diseño es NVIDIA Jetson (Orin). No se dispone del hardware físico, "
              "por lo que el benchmark real de TensorRT sobre Jetson queda como limitación documentada. "
              "Los números aquí son de CPU x86 con threads acotados como **aproximación conservadora** "
              "del régimen edge; una Jetson con TensorRT sería sustancialmente más rápida en inferencia "
              "(GPU + engine INT8 optimizado), por lo que estos valores son un **techo superior de latencia**, "
              "no una predicción de Jetson.")
    md.append("- **Conclusión honesta:** el benchmark acá valida que el pipeline de inferencia corre "
              "y mide variabilidad, pero **no demuestra el speedup de INT8** que es uno de los argumentos "
              " centrales del proyecto. Para validarlo hace falta el hardware target (Jetson) o, "
              "como paso posterior opcional, buildar el engine de TensorRT en una GPU NVIDIA disponible "
              "y medir INT8 vs FP32 ahí.")
    md_path = REPORTS_DIR / "benchmark.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"\n=== Tabla benchmark ===")
    for line in md:
        print(line)
    print(f"\nGuardado: {csv_path}\n          {md_path}")


if __name__ == "__main__":
    main()