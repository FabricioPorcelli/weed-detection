#!/usr/bin/env python3
"""Descarga los pesos del modelo desde un GitHub Release (Fase 5/desploy).

Pensado para dos usos:
  1) Local / fresh-clone: bajar los pesos si no querés entrenar.
  2) Streamlit Cloud: la app lo invoca si `models/` está vacío (los pesos no
     se commitean, están gitignored) para auto-arrancar el demo.

El release debe contener como assets los archivos listados en MODEL_FILES
(baseline_best.pt, baseline.onnx, baseline_int8.onnx) generados por
`src/train.py` y `src/export.py`.

Para crear el release (reemplazar OWNER/REPO y tag):
    gh release create models-v1 models/baseline_best.pt models/baseline.onnx \
        models/baseline_int8.onnx \
        --repo FabricioPorcelli/weed-detection \
        --notes "Pesos del baseline + ONNX FP32/INT8 PTQ"

Uso:
    python src/download_models.py                  # descarga los 3 si faltan
    python src/download_models.py --force          # re-descarga siempre
    python src/download_models.py --tag models-v1
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"

# Tag por defecto del release; overrideable desde la app via variable de entorno
# MODEL_RELEASE_TAG (para poder cambiar sin tocar código).
DEFAULT_TAG = "models-v1"
DEFAULT_REPO = "FabricioPorcelli/weed-detection"

MODEL_FILES = ["baseline_best.pt", "baseline.onnx", "baseline_int8.onnx", "demo_input.mp4"]


def release_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def download_one(url: str, dst: Path, force: bool = False) -> bool:
    if dst.exists() and not force:
        print(f"  [skip] {dst.name} ya existe ({dst.stat().st_size/1e6:.2f} MB)")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {dst.name} <- {url}")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        print(f"  [FAIL] {dst.name}: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return False
    print(f"  [ok  ] {dst.name} ({dst.stat().st_size/1e6:.2f} MB)")
    return True


def ensure_models(models_dir: Path = MODELS_DIR, repo: str = DEFAULT_REPO,
                  tag: str | None = None, force: bool = False) -> list[Path]:
    """Descarga los modelos faltantes. Devuelve lista de paths existentes."""
    import os
    tag = tag or os.environ.get("MODEL_RELEASE_TAG", DEFAULT_TAG)
    repo = os.environ.get("MODEL_RELEASE_REPO", repo)
    models_dir.mkdir(exist_ok=True)
    existing = []
    for f in MODEL_FILES:
        dst = models_dir / f
        if not dst.exists() or force:
            download_one(release_url(repo, tag, f), dst, force)
        if dst.exists():
            existing.append(dst)
    return existing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--tag", default=None, help=f"release tag (default {DEFAULT_TAG})")
    ap.add_argument("--force", action="store_true", help="re-descargar aunque existan")
    args = ap.parse_args()
    existing = ensure_models(repo=args.repo, tag=args.tag, force=args.force)
    if not existing:
        print(f"\nNo se consiguió ningún modelo en {MODELS_DIR}.")
        sys.exit(1)
    print(f"\nModelos disponibles en {MODELS_DIR}: "
          + ", ".join(p.name for p in existing))


if __name__ == "__main__":
    main()