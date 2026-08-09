"""AgroVision-Edge — Demo visual (Fase 5).

Características:
  - selector de modelo (baseline .pt | .onnx FP32 | .onnx INT8 PTQ) con métricas
    cargadas de reports/ (mAP, tamaño, latencia) para evidenciar el trade-off en vivo
  - pestañas Imagen / Video (DECISIÓN 5.2: soporta ambos)
  - sidebar con secciones y tooltips explicando cada parámetro
  - resumen por clase (crop/weed), distribución de confianza y latencia por frame
  - descarga de CSV y, en video, del .mp4 anotado
  - botón "Probar video demo" para cargar demo/demo_input.mp4
  - footer con limitaciones (no representa rendimiento en campo real)

Run:
  streamlit run app/streamlit_app.py
  streamlit run app/streamlit_app.py --server.fileWatcherType none   # si segfault
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

# --- workaround para segfault torch + Streamlit ---
# El inspector de modulos de Streamlit camina torch.classes.__path__ y provoca
# segfault al primer uso de torch dentro de la app. Vaciar __path__ antes de
# importar streamlit evita que el inspector lo recorra.
import torch
try:
    torch.classes.__path__ = []  # type: ignore[attr-defined]
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEMO_VIDEO = ROOT / "demo" / "demo_input.mp4"
BENCH_CSV = ROOT / "reports" / "benchmark" / "benchmark.csv"
OPT_CSV = ROOT / "reports" / "optimization" / "comparison.csv"

CLASSES = ["crop", "weed"]
CLASS_COLOR_BGR = {0: (0, 200, 0), 1: (220, 30, 30)}     # BGR para cv2
CLASS_COLOR_HEX = {0: "#2ecc40", 1: "#e51c23"}           # hex para legends mpls

MODEL_CHOICES = {
    "Baseline (.pt FP32)": "baseline_best.pt",
    "ONNX FP32": "baseline.onnx",
    "ONNX INT8 (PTQ)": "baseline_int8.onnx",
}


# ----------------------- helpers de modelo -----------------------

@st.cache_resource(show_spinner="Cargando modelo…")
def load_model_cached(name: str):
    """Carga y cachea el modelo (un solo YOLO por sesión; evita reinstanciar torch)."""
    from ultralytics import YOLO
    p = MODELS_DIR / name
    if not p.exists():
        return None, p
    if p.suffix == ".onnx":
        return YOLO(str(p), task="detect"), p
    return YOLO(str(p)), p


def model_stats(name: str) -> dict:
    info = {}
    if BENCH_CSV.exists():
        for r in csv.DictReader(BENCH_CSV.open()):
            if Path(r["modelo"]).name == name and r["modo"] == "edge":
                info["latency_ms"] = float(r["latency_mean_ms"])
                info["fps"] = float(r["fps_mean"])
                break
    if OPT_CSV.exists():
        for r in csv.DictReader(OPT_CSV.open()):
            if Path(r["archivo"]).name == name:
                info["mAP50"] = float(r["mAP50"])
                info["mAP50-95"] = float(r["mAP50-95"])
                info["size_MB"] = float(r["tamaño_MB"])
                break
    return info


# ----------------------- helpers de inferencia -----------------------

def run_inference(model, im_bgr, imgsz=640, conf=0.25, iou=0.7):
    t0 = time.perf_counter()
    res = model.predict(im_bgr, imgsz=imgsz, conf=conf, iou=iou,
                        verbose=False, device="cpu")[0]
    dt = (time.perf_counter() - t0) * 1000
    boxes = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    return boxes, cls, confs, dt, res.names


def draw(im, boxes, cls, confs):
    for (x1, y1, x2, y2), c, conf in zip(boxes, cls, confs):
        col = CLASS_COLOR_BGR.get(int(c), (255, 255, 255))
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{CLASSES[int(c)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (int(x1), int(y1) - th - 6),
                      (int(x1) + tw + 6, int(y1)), col, -1)
        cv2.putText(im, label, (int(x1) + 3, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im


def detections_df(boxes, cls, confs, names, frame=None) -> pd.DataFrame:
    rows = []
    for (x1, y1, x2, y2), c, cf in zip(boxes, cls, confs):
        row = {"class": names.get(int(c), str(int(c))), "conf": float(cf),
               "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
        if frame is not None:
            row["frame"] = frame
        rows.append(row)
    return pd.DataFrame(rows)


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["class", "count", "avg_conf"])
    g = df.groupby("class")["conf"].agg(["count", "mean"]).reset_index()
    g.columns = ["class", "count", "avg_conf"]
    g["avg_conf"] = g["avg_conf"].round(3)
    return g


# ----------------------- sidebar -----------------------

def render_sidebar():
    st.sidebar.markdown(
        "## ⚙️ Configuración\n"
        "<small>Ajustá el modelo y los parámetros de detección.</small>",
        unsafe_allow_html=True)

    st.sidebar.markdown("##### Modelo")
    label = st.sidebar.selectbox(
        "Variante", list(MODEL_CHOICES.keys()),
        help="Baseline = PyTorch nativo (más rápido en CPU desktop). "
             "ONNX FP32 = formato portable. ONNX INT8 = cuantizado, "
             "más chico y portable para edge (misma mAP, casi sin pérdida).")
    model_file = MODEL_CHOICES[label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Detección")
    conf = st.sidebar.slider(
        "Confianza mínima", 0.05, 0.95, 0.25, 0.05,
        help="Umbral de score: sólo se muestran detecciones con confianza ≥ a este valor. "
             "Subirlo → menos falsos positivos pero posibles falsos negativos.")
    iou = st.sidebar.slider(
        "IoU NMS", 0.10, 0.95, 0.70, 0.05,
        help="Umbral de supresión de no-máximos (Intersection-over-Union). "
             "Bajarlo fusiona más boxes solapados; subirlo deja más boxes superpuestos.")
    imgsz = st.sidebar.select_slider(
        "Resolución de entrada (px)", [320, 416, 640], value=640,
        help="Tamaño al que se redimensiona la imagen antes de inferir. "
             "Menor → más rápido en edge, pero puede perder objetos pequeños "
             "(en este dataset los objetos son grandes, bajar suele costar poco).")

    st.sidebar.markdown("---")
    info = model_stats(model_file)
    st.sidebar.markdown("##### Métricas del modelo (modo edge)")
    if info:
        c1, c2 = st.sidebar.columns(2)
        c1.metric("mAP50", f"{info.get('mAP50', 0):.3f}")
        c2.metric("mAP50-95", f"{info.get('mAP50-95', 0):.3f}")
        c1.metric("Tamaño", f"{info.get('size_MB', 0):.2f} MB")
        c2.metric("Latencia", f"{info.get('latency_ms', 0):.1f} ms")
        st.sidebar.caption(f"FPS ≈ {info.get('fps', 0):.1f} (edge simulado, CPU threads=2)")
    else:
        st.sidebar.caption("Corré `python src/benchmark.py` para llenar métricas.")
    st.sidebar.caption(f"Archivo: `{model_file}`")

    return label, model_file, conf, iou, imgsz


# ----------------------- tabs -----------------------

def render_legend():
    st.markdown("**Leyenda:** &nbsp; "
                f"<span style='color:{CLASS_COLOR_HEX[0]}'>■</span> crop &nbsp; "
                f"<span style='color:{CLASS_COLOR_HEX[1]}'>■</span> weed",
                unsafe_allow_html=True)


def tab_image(model, model_meta, conf, iou, imgsz):
    st.subheader("Detección sobre imagen estática")
    col_up, col_demo = st.columns([3, 1])
    upload = col_up.file_uploader(
        "Subí una imagen (.jpg / .png)", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="img_up")
    if upload is None:
        st.info("Esperando una imagen…")
        return

    im_bgr = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if im_bgr is None:
        st.error("No se pudo decodificar la imagen.")
        return

    boxes, cls, confs, dt, names = run_inference(model, im_bgr, imgsz, conf, iou)
    annotated = draw(im_bgr.copy(), boxes, cls, confs)

    render_legend()
    show_annotated = st.toggle("Mostrar anotada", value=True,
                                help="Alternar entre imagen original y con detecciones.")
    st.image((annotated if show_annotated else im_bgr)[:, :, ::-1],
             channels="RGB", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detecciones", len(boxes))
    m2.metric("Latencia (frame)", f"{dt:.1f} ms")
    fps = 1000 / dt if dt > 0 else 0
    m3.metric("FPS (este frame)", f"{fps:.1f}")
    m4.metric("Resolución", f"{imgsz}px")

    df = detections_df(boxes, cls, confs, names)
    if df.empty:
        st.warning("Sin detecciones por encima del umbral de confianza.")
        return

    left, right = st.columns([2, 3])
    with left:
        st.markdown("##### Resumen por clase")
        st.dataframe(class_summary(df), hide_index=True, use_container_width=True)
        st.download_button("⬇️ Descargar detecciones (CSV)",
                           df.to_csv(index=False).encode(),
                           "detections.csv", "text/csv", use_container_width=True)
    with right:
        st.markdown("##### Tabla de detecciones")
        st.dataframe(df, hide_index=True, use_container_width=True)


def tab_video(model, model_meta, conf, iou, imgsz):
    st.subheader("Detección sobre video (frame a frame)")
    col_up, col_demo = st.columns([3, 1])
    upload = col_up.file_uploader(
        "Subí un video (.mp4)", type=["mp4"], label_visibility="collapsed", key="vid_up")
    use_demo = col_demo.button("🎬 Probar video demo",
                                help="Carga demo/demo_input.mp4 (frames del dataset, "
                                "no representa rendimiento en campo real).",
                                disabled=not DEMO_VIDEO.exists())

    source = None
    src_name = "demo_input.mp4"
    if upload is not None:
        tmp = Path("/tmp/streamlit_in.mp4")
        tmp.write_bytes(upload.getvalue())
        source, src_name = tmp, upload.name
    elif use_demo:
        source = DEMO_VIDEO
    else:
        st.info("Esperando un video, o pulsá **Probar video demo**…")
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.caption(f"**{src_name}** — {total} frames · {fps_src:.0f} fps · {w}×{h}px")

    render_legend()
    stframe = st.empty()
    progress = st.progress(0.0, text="Procesando…")
    status = st.empty()

    rows = []
    latencies = []
    out_path = Path("/tmp/streamlit_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))

    t0 = time.perf_counter()
    idx = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        boxes, cls, confs, dt, names = run_inference(model, im, imgsz, conf, iou)
        latencies.append(dt)
        im_out = draw(im, boxes, cls, confs)
        writer.write(im_out)
        stframe.image(im_out[:, :, ::-1], channels="RGB",
                      caption=f"frame {idx + 1}/{total or '?'} · {dt:.1f} ms")
        for _b, c, cf in zip(boxes, cls, confs):
            rows.append({"frame": idx, "class": names.get(int(c), str(int(c))),
                         "conf": float(cf)})
        idx += 1
        if total > 0:
            progress.progress(idx / total, text=f"Procesando frame {idx}/{total}")

    cap.release()
    writer.release()
    dt_total = time.perf_counter() - t0
    progress.progress(1.0, text="✅ Listo")
    status.empty()

    if idx == 0:
        st.error("No se leyeron frames.")
        return

    lat = np.array(latencies)
    df = pd.DataFrame(rows)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frames", idx)
    m2.metric("Detecciones", len(df))
    m3.metric("Tiempo total", f"{dt_total:.1f} s")
    m4.metric("FPS proceso", f"{idx / dt_total:.1f}")
    m5.metric("Latencia media", f"{lat.mean():.1f} ms")
    st.caption(f"Latencia: min {lat.min():.1f} · mediana {np.median(lat):.1f} · "
               f"p95 {np.percentile(lat,95):.1f} · max {lat.max():.1f} ms")

    if not df.empty:
        left, right = st.columns([2, 3])
        with left:
            st.markdown("##### Resumen por clase")
            st.dataframe(class_summary(df), hide_index=True, use_container_width=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Detecciones (CSV)",
                                df.to_csv(index=False).encode(),
                                "detections.csv", "text/csv", use_container_width=True)
            if out_path.exists():
                d2.download_button("⬇️ Video anotado (MP4)",
                                    out_path.read_bytes(), "annotated.mp4",
                                    "video/mp4", use_container_width=True)
        with right:
            st.markdown("##### Primeras 50 detecciones")
            st.dataframe(df.head(50), hide_index=True, use_container_width=True)


# ----------------------- main -----------------------

def main():
    st.set_page_config(page_title="AgroVision-Edge demo", page_icon="🌱",
                       layout="wide", initial_sidebar_state="expanded")
    # hero / header
    st.markdown(
        "<h1 style='margin-bottom:0'>🌱 AgroVision-Edge</h1>"
        "<p style='font-size:1.1rem;color:#6b7280;margin-top:0'>"
        "Detección de <b>cultivo vs maleza</b> con YOLO optimizado para edge</p>",
        unsafe_allow_html=True)

    with st.expander("ℹ️ Sobre este demo", expanded=False):
        st.markdown(
            "Demo del pipeline end-to-end: imagen/video → detección → visualización + CSV. "
            "El **selector de modelo** (sidebar) deja ver el trade-off en vivo entre "
            "*baseline* y *cuantizado INT8*.\n\n"
            "**Disclaimer:** el modelo se entrenó en un dataset de sésamo (no en soja/maíz) "
            "y el video demo usa frames del propio dataset — **no representa rendimiento en campo real**."
        )

    label, model_file, conf, iou, imgsz = render_sidebar()
    model, model_path = load_model_cached(model_file)
    if model is None:
        st.error(f"No existe el modelo `{model_path}`. Generá los pesos con "
                 "`python src/export.py` y `python src/train.py`.")
        st.stop()

    model_meta = MODEL_CHOICES[label]
    tab_img, tab_vid = st.tabs(["🖼️ Imagen", "🎞️ Video"])
    with tab_img:
        tab_image(model, model_meta, conf, iou, imgsz)
    with tab_vid:
        tab_video(model, model_meta, conf, iou, imgsz)

    st.markdown("---")
    st.caption(
        "AgroVision-Edge · Portfolio · "
        "Ver README para detalle de trade-offs y limitaciones.")


if __name__ == "__main__":
    main()