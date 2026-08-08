#!/usr/bin/env python3
"""Descarga el dataset de Crop and Weed Detection desde Kaggle a data/raw/.

Requiere:
  - Credenciales de Kaggle en ~/.kaggle/access_token
  - Paquete `kaggle` instalado (ver requirements.txt)

Uso:
  python src/download_dataset.py
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_REF = "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def check_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "access_token"
    if not kaggle_json.exists():
        sys.exit(
            "No se encontraron credenciales de Kaggle en ~/.kaggle/access_token.\n"
            "Creá el archivo con tu username y API key (https://www.kaggle.com/settings -> API)."
        )
    if not shutil.which("kaggle"):
        sys.exit(
            "El CLI de Kaggle no está instalado. Ejecutá: pip install kaggle"
        )


def download(target: Path, overwrite: bool = False) -> None:
    if target.exists() and not overwrite:
        print(f"El directorio {target} ya existe. Usá --overwrite para re-descargar.")
        return

    check_credentials()

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "dataset.zip"

    print(f"Descargando {DATASET_REF} ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_REF,
            "-p",
            str(target),
            "--unzip",
        ],
        check=True,
    )

    if zip_path.exists():
        zip_path.unlink()
    print(f"Dataset listo en: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Re-descargar aunque exista")
    args = parser.parse_args()

    download(RAW_DIR, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
