#!/usr/bin/env python3
"""Arma un video de demo a partir de frames del dataset (Fase 5).

Simula captura continua tipo dispositivo montado en maquinaria: concatena N
imagenes del dataset en un .mp4 a fps dados, opcionalmente con cada frame
repetido K veces para dar sensacion de "paso de camara".

HONESTO: los frames provienen del propio dataset (mismo dominio que el
entrenamiento). Esto NO representa rendimiento en campo real; es solo para
demostrar el pipeline end-to-end (video -> deteccion -> video+CSV).

Uso:
  python src/make_demo_video.py                       # 30 frames de test -> demo.mp4
  python src/make_demo_video.py --n 50 --fps 10 --repeat 3 --split test
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
DEMO_DIR = ROOT / "demo"
DEFAULT_OUT = DEMO_DIR / "demo_input.mp4"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--n", type=int, default=30, help="cantidad de frames a incluir")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--repeat", type=int, default=2,
                   help="veces que se repite cada frame (suaviza el movimiento)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def collect_image_paths(data_yaml: str, split: str) -> list[Path]:
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(cfg["path"])
    img_dir = base / f"images/{split}"
    imgs = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        imgs.extend(sorted(img_dir.glob(ext)))
    return imgs


def main():
    args = get_args()
    if not Path(args.data).exists():
        raise SystemExit(f"No existe {args.data}. Ejecutá src/make_splits.py primero.")

    imgs = collect_image_paths(args.data, args.split)
    if not imgs:
        raise SystemExit(f"No hay imágenes en split '{args.split}'.")

    rng = random.Random(args.seed)
    n = min(args.n, len(imgs))
    selected = rng.sample(imgs, n)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    # leemos el tamaño del primer frame para configurar el writer
    first = cv2.imread(str(selected[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    written = 0
    for p in selected:
        im = cv2.imread(str(p))
        if im is None:
            continue
        # si difiere de tamaño, resize (por si hay variabilidad)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        for _ in range(args.repeat):
            writer.write(im)
            written += 1

    writer.release()
    print(f"Video demo creado: {out_path}")
    print(f"  frames source: {n}  |  frames totales: {written}  |  fps: {args.fps}")
    print(f"  duracion estimada: {written / args.fps:.1f}s")
    print("\nNOTA: frames provienen del propio dataset (mismo dominio que el entrenamiento).")
    print("No representa rendimiento en campo real; solo demuestra el pipeline.")


if __name__ == "__main__":
    main()