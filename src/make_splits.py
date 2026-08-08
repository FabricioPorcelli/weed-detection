#!/usr/bin/env python3
"""Arma los splits train/val/test y la config YOLO en data/processed/.

Estrategia (decisión 1.4): 80/10/10 estratificado por composición de clases
de cada imagen (crop / weed / both). Reproducible con seed.

Estructura de salida (formato Ultralytics YOLO):

    data/processed/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── data.yaml

Diseño dataset-agnostic: las clases salen de data/raw/classes.txt, así se
puede swappear el dataset sin reescribir este script.

Uso:
    python src/make_splits.py               # 80/10/10, seed 42
    python src/make_splits.py --seed 123
    python src/make_splits.py --symlink      # symlinks en vez de copiar
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DIR = ROOT / "data" / "raw" / "agri_data" / "data"
PROC_DIR = ROOT / "data" / "processed"
CLASSES_FILE = ROOT / "data" / "raw" / "classes.txt"

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def load_classes() -> list[str]:
    if not CLASSES_FILE.exists():
        sys.exit(f"No existe {CLASSES_FILE}. Ejecutá src/download_dataset.py primero.")
    names = [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]
    if not names:
        sys.exit(f"{CLASSES_FILE} vacío o sin clases válidas.")
    return names


def image_classes(stem: str, n_classes: int) -> str:
    """Devuelve un stratum reproducible: 'crop', 'weed' o 'both'."""
    txt = RAW_DIR / f"{stem}.txt"
    present = set()
    with open(txt) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 5:
                present.add(int(parts[0]))
    if len(present) >= 2:
        return "both"
    if len(present) == 1:
        c = next(iter(present))
        return names[c] if c < len(names) else f"cls{c}"
    return "empty"


def stratified_split(items: list[str], strata: list[str], ratios, seed):
    """Split estratificado por stratum según `ratios` (dict train/val/test)."""
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for stem, s in zip(items, strata):
        by_stratum[s].append(stem)

    import random
    rng = random.Random(seed)

    split_of: dict[str, str] = {}
    for s, stems in by_stratum.items():
        stems = sorted(stems)
        rng.shuffle(stems)
        n = len(stems)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        # test toma el resto; garantizando no quedarse en 0 por redondeo
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train += n_test
            n_test = 0
        split_of.update(dict(zip(stems[:n_train], ["train"] * n_train)))
        split_of.update(dict(zip(stems[n_train:n_train + n_val], ["val"] * n_val)))
        split_of.update(dict(zip(stems[n_train + n_val:], ["test"] * n_test)))
    return split_of


def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_data_yaml(classes: list[str]) -> Path:
    yml = {
        "path": str(PROC_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
    }
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "data.yaml"
    out.write_text(yaml.safe_dump(yml, sort_keys=False, allow_unicode=True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true",
                    help="Crear symlinks en lugar de copiar imgs/labels")
    args = ap.parse_args()

    global names
    names = load_classes()
    print(f"Clases ({len(names)}): {names}")

    # listado de imágenes
    imgs = sorted(p.stem for p in RAW_DIR.glob("*.jpeg"))
    print(f"Imágenes en raw: {len(imgs)}")

    # stratum por imagen
    strata = [image_classes(s, len(names)) for s in imgs]
    from collections import Counter
    print("Distribución por stratum:", dict(Counter(strata)))

    # split
    split_of = stratified_split(imgs, strata, SPLIT_RATIOS, args.seed)
    counts = Counter(split_of.values())
    print(f"Splits: {dict(counts)}")

    # preparar dirs
    for split in SPLIT_RATIOS:
        (PROC_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROC_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # copiar/symlink
    n_done = 0
    for stem, split in split_of.items():
        img_src = RAW_DIR / f"{stem}.jpeg"
        txt_src = RAW_DIR / f"{stem}.txt"
        link_or_copy(img_src, PROC_DIR / "images" / split / f"{stem}.jpeg", args.symlink)
        link_or_copy(txt_src, PROC_DIR / "labels" / split / f"{stem}.txt", args.symlink)
        n_done += 1
    print(f"Archivos colocados: {n_done} imgs + {n_done} labels")

    # per-stratum per-split report
    report = defaultdict(lambda: defaultdict(int))
    for stem, split in split_of.items():
        s = image_classes(stem, len(names))
        report[split][s] += 1
    print("\nReporte estratificado:")
    print(f"{'split':<8} " + " ".join(f"{k:<8}" for k in sorted({s for st in strata for s in [st]})) + "total")
    for split in ("train", "val", "test"):
        row = report[split]
        print(f"{split:<8} " + " ".join(f"{row.get(k, 0):<8}" for k in sorted(row)) + f"{counts[split]}")

    yml_path = write_data_yaml(names)
    print(f"\ndata.yaml -> {yml_path}")
    print(yml_path.read_text())


if __name__ == "__main__":
    main()