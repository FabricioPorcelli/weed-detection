#!/usr/bin/env python3
"""Inferencia standalone (CLI) — Fase 5.1.

Recibe una carpeta de imágenes O un archivo de video y devuelve:
  - las imágenes/video con bounding boxes dibujados (en <out>/annotated/)
  - un CSV con todas las detecciones (image, class, conf, x1,y1,x2,y2)

Selector de modelo: baseline (.pt), .onnx FP32 o .onnx INT8 (PTQ).

Uso:
  python src/inference.py --source demo/demo_input.mp4 --model models/baseline_int8.onnx
  python src/inference.py --source data/processed/images/test --model models/baseline_best.pt
  python src/inference.py --source una_imagen.jpeg --conf 0.35
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MODEL = ROOT / "models" / "baseline_int8.onnx"
DEFAULT_OUT = ROOT / "demo" / "out"
CLASSES = ["crop", "weed"]
CLASS_COLOR = {0: (0, 200, 0), 1: (220, 30, 30)}   # crop=verde, weed=rojo (BGR)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   help="archivo de video, imagen, o carpeta de imágenes")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="pesos (.pt | .onnx)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-frames", action="store_true",
                   help="en modo video, guardar también los frames individuales")
    return p.parse_args()


def load_model(path: Path, device: str):
    from ultralytics import YOLO
    if path.suffix == ".onnx":
        return YOLO(str(path), task="detect")
    return YOLO(str(path))


def draw(im, boxes, classes_xyxy, confs):
    for (x1, y1, x2, y2), c, conf in zip(boxes, classes_xyxy, confs):
        col = CLASS_COLOR.get(int(c), (255, 255, 255))
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{CLASSES[int(c)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), col, -1)
        cv2.putText(im, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im


def iter_images(source: Path):
    if source.is_file() and source.suffix.lower() in (".jpg", ".jpeg", ".png"):
        yield source
    elif source.is_dir():
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            yield from sorted(source.glob(ext))
    else:
        raise SystemExit(f"Source inválido: {source}")


def process_image(path: Path, model, args, writer_csv, out_dir, frame_idx=None):
    im = cv2.imread(str(path))
    if im is None:
        return
    res = model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                        verbose=False, device=args.device)[0]
    names = res.names
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    draw(im, boxes, cls, confs)
    name = path.stem if frame_idx is None else f"frame_{frame_idx:05d}"
    cv2.imwrite(str(out_dir / f"{name}.jpg"), im)
    for (x1, y1, x2, y2), c, conf in zip(boxes, cls, confs):
        writer_csv.writerow({
            "image": path.name if frame_idx is None else f"frame_{frame_idx:05d}",
            "class_id": int(c),
            "class": names.get(int(c), str(int(c))),
            "conf": float(conf),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        })


def process_video(path: Path, model, args, writer_csv, out_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.imgsz // 32  # fallback
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_dir = out_dir / "frames" if args.save_frames else None
    if frames_dir:
        frames_dir.mkdir(exist_ok=True)
    annotated_path = out_dir / f"{path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

    idx = 0
    t0 = time.perf_counter()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        res = model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            verbose=False, device=args.device)[0]
        names = res.names
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        draw(im, boxes, cls, confs)
        writer.write(im)
        if frames_dir:
            cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.jpg"), im)
        for (x1, y1, x2, y2), c, conf in zip(boxes, cls, confs):
            writer_csv.writerow({
                "image": f"frame_{idx:05d}",
                "class_id": int(c),
                "class": names.get(int(c), str(int(c))),
                "conf": float(conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            })
        idx += 1
    cap.release()
    writer.release()
    dt = time.perf_counter() - t0
    print(f"Video: {idx} frames en {dt:.1f}s ({idx/dt:.1f} FPS procesados) -> {annotated_path}")
    print(f"  además: frames individuales en {frames_dir}" if frames_dir else "")


def main():
    args = get_args()
    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"No existe: {src}")
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"No existe el modelo: {model_path}. Generá con src/export.py.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, args.device)
    csv_path = out_dir / "detections.csv"
    fh = csv_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=["image", "class_id", "class", "conf",
                                            "x1", "y1", "x2", "y2"])
    writer.writeheader()

    print(f"Modelo: {model_path.name}  |  source: {src}  |  out: {out_dir}")
    if src.is_file() and src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        process_video(src, model, args, writer, out_dir)
    else:
        for i, p in enumerate(iter_images(src)):
            process_image(p, model, args, writer, out_dir, frame_idx=None)
            if (i + 1) % 25 == 0:
                print(f"  procesadas {i+1} imágenes...")
    fh.close()
    n = sum(1 for _ in csv_path.open()) - 1
    print(f"\nListo. Detecciones: {n} -> {csv_path}")
    print(f"Annotated en: {out_dir}")


if __name__ == "__main__":
    main()