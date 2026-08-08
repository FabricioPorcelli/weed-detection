"""Construye notebooks/01_eda.ipynb (sin ejecutar) usando nbformat.

Uso:
  python notebooks/build_eda.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_PATH = Path(__file__).resolve().parent / "01_eda.ipynb"

cells = []

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("""# 01 — EDA: Crop and Weed Detection

Exploración cuantitativa del dataset **Crop and Weed Detection Data with Bounding Boxes** (Kaggle, `ravirajsinh45`), ubicado en `data/raw/`.

**Objetivo (Fase 1.3 del plan):**
1. Conteo total de imágenes y de instancias (bounding boxes) por clase.
2. Decisión de granularidad de clases (**(a) Binaria** — tal cual viene el dataset: `crop` vs `weed`).
3. Distribución de tamaño de las bounding boxes (objetos *small* = malezas jóvenes, relevantes para `imgsz`).
4. Resolución y aspect ratio de las imágenes originales.
5. Chequeo de duplicados e imágenes corruptas.
6. Visualización de una muestra con bounding boxes dibujados.

Dataset ya verificado (1.2): 1300 `.jpeg` + 1300 `.txt` (1:1, sin faltantes), formato YOLO normalizado en [0,1], resolución 512×512."""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Configuración general
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_DIR = Path("../data/raw/agri_data/data")
IMG_DIR = DATA_DIR
CLASSES = ["crop", "weed"]          # class_id 0 / 1 (de classes.txt)
CLASS_COLOR = {0: (0, 200, 0), 1: (220, 30, 30)}   # crop=verde, weed=rojo
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

assert DATA_DIR.exists(), f"No existe {DATA_DIR}. Ejecutá src/download_dataset.py primero."
print("Dataset dir:", DATA_DIR)"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("## 1. Carga de anotaciones (formato YOLO)"))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Parseo de todos los .txt -> DataFrame de instancias + DataFrame de imágenes
records = []
img_class_sets = {}
for txt_path in sorted(DATA_DIR.glob("*.txt")):
    if txt_path.name == "classes.txt":
        continue
    stem = txt_path.stem
    classes_in_img = set()
    n_lines = 0
    with open(txt_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, x, y, w, h = map(float, parts)
            records.append({
                "image": stem,
                "class_id": int(c),
                "class": CLASSES[int(c)],
                "xc": x, "yc": y, "w": w, "h": h,
            })
            classes_in_img.add(int(c))
            n_lines += 1
    img_class_sets[stem] = classes_in_img

df = pd.DataFrame(records)
print(f"Instancias totales: {len(df):,}")
print(f"Imágenes: {len(img_class_sets):,}")
df.head()"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("## 2. Conteos por clase"))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Instancias por clase
counts = df["class"].value_counts().rename("instancias")
ratio = counts.max() / counts.min()
print(counts.to_string())
print(f"\\nDesbalance (mayor/menor): {ratio:.2f}:1")

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
            palette={"crop": "#4CAF50", "weed": "#E53935"}, legend=False)
for i, v in enumerate(counts.values):
    ax.text(i, v + 15, f"{v:,}", ha="center", fontweight="bold")
ax.set_title("Instancias por clase")
ax.set_ylabel("Cantidad de bounding boxes")
ax.set_xlabel("Clase")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_class_counts.png", bbox_inches="tight")
plt.show()"""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Imágenes que contienen cada clase (una imagen puede tener ambas)
img_classes = pd.DataFrame(
    [(img, 0 in cs, 1 in cs) for img, cs in img_class_sets.items()],
    columns=["image", "has_crop", "has_weed"]
)
imgs_crop = img_classes["has_crop"].sum()
imgs_weed = img_classes["has_weed"].sum()
imgs_both = (img_classes["has_crop"] & img_classes["has_weed"]).sum()
print(f"Imágenes con crop : {imgs_crop:,}")
print(f"Imágenes con weed : {imgs_weed:,}")
print(f"Imágenes con ambas: {imgs_both:,}")
print(f"Bboxes por imagen (promedio): {len(df)/len(img_classes):.2f}")"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("""## 3. Distribución de tamaño de bounding boxes

Los objetos chicos son más difíciles de detectar. En field application las malezas jóvenes aparecen pequeñas en la imagen. Reportamos tamaños en **píxeles** (sobre 512×512) y marcamos el umbral de *small object* (<32×32 px, criterio de COCO)."""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Conversión a píxeles sobre 512x512
IMG_SIZE = 512
df["w_px"] = (df["w"] * IMG_SIZE).round().astype(int)
df["h_px"] = (df["h"] * IMG_SIZE).round().astype(int)
df["area_px"] = df["w_px"] * df["h_px"]
df["small"] = (df["w_px"] < 32) & (df["h_px"] < 32)   # COCO small-object umbral
df.head()"""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Distribución de ancho y alto (px) por clase
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes, ["w_px", "h_px"], ["Ancho (px)", "Alto (px)"]):
    sns.histplot(data=df, x=col, hue="class", bins=40, ax=ax,
                  palette={"crop": "#4CAF50", "weed": "#E53935"}, alpha=0.65)
    ax.axvline(32, color="black", ls="--", lw=1)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_bbox_size_dist.png", bbox_inches="tight")
plt.show()"""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Área (px^2) por clase (log scale por el rango amplio)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x="area_px", hue="class", bins=50, ax=ax, log_scale=True,
            palette={"crop": "#4CAF50", "weed": "#E53935"}, alpha=0.6)
ax.set_title("Distribución de área de bboxes (px², escala log)")
ax.set_xlabel("Área (px²)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_bbox_area_dist.png", bbox_inches="tight")
plt.show()"""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# % de objetos small (<32x32 px) por clase
small_by_class = df.groupby("class")["small"].agg(["sum", "mean", "count"])
small_by_class["pct_small"] = (small_by_class["mean"] * 100).round(2)
print("Objetos small (<32x32 px):")
print(small_by_class[["sum", "count", "pct_small"]])
print(f"\\nTotal small objects: {df['small'].sum()} ({df['small'].mean()*100:.2f}%)")"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("## 4. Resolución y aspect ratio de las imágenes"))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Verificamos resolución y aspect ratio sobre TODAS las imágenes
sizes = []
corrupt = []
for img_path in sorted(IMG_DIR.glob("*.jpeg")):
    try:
        with Image.open(img_path) as im:
            sizes.append((img_path.name, im.size, im.size[0] / im.size[1]))
    except Exception as e:
        corrupt.append((img_path.name, str(e)))

size_df = pd.DataFrame(sizes, columns=["file", "(w,h)", "aspect_ratio"])
print(f"Imágenes procesadas: {len(size_df)}")
print(f"Imágenes corruptas : {len(corrupt)}")
print("\\nResoluciones únicas:")
print(size_df["(w,h)"].value_counts())
print(f"\\nAspect ratio: min={size_df['aspect_ratio'].min():.3f}, "
      f"max={size_df['aspect_ratio'].max():.3f}, todo 1:1 = "
      f"{size_df['aspect_ratio'].nunique()==1}")"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("## 5. Chequeo de duplicados e imágenes corruptas"))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Duplicados exactos por hash MD5 ( imágenes idénticas a nivel byte )
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

hashes = {}
dupes = []
for img_path in sorted(IMG_DIR.glob("*.jpeg")):
    d = md5(img_path)
    if d in hashes:
        dupes.append((hashes[d], img_path.name))
    else:
        hashes[d] = img_path.name

print(f"Hashes únicos : {len(hashes)}")
print(f"Duplicados    : {len(dupes)}")
if dupes:
    print("Ejemplos de duplicados:")
    for a, b in dupes[:10]:
        print(f"  {a}  ==  {b}")"""))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Re-validación de formato YOLO: clases válidas {0,1}, coords en [0,1], 5 columnas
bad_class = df[~df["class_id"].isin([0, 1])]
bad_coords = df[(df["xc"] < 0) | (df["xc"] > 1) | (df["yc"] < 0) | (df["yc"] > 1)
              | (df["w"] < 0) | (df["w"] > 1) | (df["h"] < 0) | (df["h"] > 1)]
print(f"Instancias con class_id inválido: {len(bad_class)}")
print(f"Instancias con coords fuera de [0,1]: {len(bad_coords)}")
print(f"Imágenes .txt vacíos: {sum(1 for p in DATA_DIR.glob('*.txt') if p.name!='classes.txt' and p.stat().st_size==0)}")"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("## 6. Visualización de una muestra con bounding boxes"))

# --------------------------------------------------------------------------
cells.append(new_code_cell("""# Grilla de 9 imágenes al azar con boxes dibujados (verificación visual de anotaciones)
#
# Para cambiar el tamaño de texto de la etiqueta de cada box, ajustá LABEL_FONT_SIZE abajo.
from PIL import ImageFont
LABEL_FONT_SIZE = 18   # <-- subir/bajar este valor si la etiqueta sigue siendo chica/grande

def _load_font(size):
    # Proba fuentes comunes; si ninguna existe, cae al default de PIL.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

sample_stems = RNG.choice(sorted(img_class_sets), size=9, replace=False)

def draw_sample(stem, font):
    img_path = IMG_DIR / f"{stem}.jpeg"
    im = Image.open(img_path).convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    rows = df[df["image"] == stem]
    for _, r in rows.iterrows():
        w, h = im.size
        x1 = (r["xc"] - r["w"] / 2) * w
        y1 = (r["yc"] - r["h"] / 2) * h
        x2 = (r["xc"] + r["w"] / 2) * w
        y2 = (r["yc"] + r["h"] / 2) * h
        col = CLASS_COLOR[r["class_id"]]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        draw.text((x1 + 4, y1 + 4), r["class"], fill=col, font=font)
    return im

font = _load_font(LABEL_FONT_SIZE)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, stem in zip(axes.flat, sample_stems):
    ax.imshow(draw_sample(stem, font))
    ax.set_title(stem, fontsize=8)
    ax.axis("off")
plt.suptitle("Muestra con anotaciones (verde=crop, rojo=weed)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_sample_grid.png", bbox_inches="tight")
plt.show()"""))

# --------------------------------------------------------------------------
cells.append(new_markdown_cell("""## Resumen y conclusiones (insumos para decisiones de la Fase 1 y 2)

| Métrica | Valor |
|---|---|
| Imágenes totales | 1.300 |
| Resolución | 512×512 (uniforme, aspect 1:1) |
| Instancias totales | 2.072 |
| Bboxes/imagen (promedio) | 1.59 |
| crop (id 0) | 1.212 instancias |
| weed (id 1) | 860 instancias |
| Desbalance | 1.41:1 (crop mayoritario) |
| Imágenes con crop | 635 |
| Imágenes con weed | 667 |
| Imágenes con AMBAS clases | 2 (≈ mono-clase por imagen) |
| Objetos small (<32×32 px) | 29 / 2.072 = 1.40% (crop 2.39%, weed 0%) |
| Duplicados / corruptas | 0 / 0 |

**Conclusiones:**
- **Granularidad (1.3):** binaria `crop` / `weed` — el dataset no expone sub-clases de malezas; es la única opción realista.
- **Structura del dataset:** cada imagen es prácticamente mono-clase (solo 2 imgs con ambas clases), con 1–2 boxes grandes que cubren casi toda la imagen. Más cercano a *clasificación a nivel imagen con box grosero* que a detección densa de objetos pequeños. Implicancia: el baseline debería aprender fácil, y el desafío real vendrá de generalizar a malezas pequeñas en campo (no representadas aquí).
- **Split (1.4):** 80/10/10 estratificado (1300 imgs → 1040 / 130 / 130).
- **Desbalance (1.5):** leve (1.41:1). Se aplicará **class weights** en la loss de Ultralytics (no se toca el dataset físico).
- **Resolución de entrada (2.2):** imágenes a 512×512 y 98.6% de boxes grandes → entrenar baseline a 640 (techo de referencia); bajar a 416/320 en Fase 3 debería impactar poco en mAP dada la ausencia de objetos pequeños.
- **Dataset limpio:** 0 anotaciones faltantes, 0 coords fuera de [0,1], 0 duplicados, 0 corruptas."""))

# --------------------------------------------------------------------------
nb = new_notebook()
nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("Escrito:", NB_PATH)