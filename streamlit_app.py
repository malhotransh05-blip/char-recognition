import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from PIL import Image, ImageOps, ImageFilter

# ------------------------------------------------------------------
# SETTINGS - change only if your file names are different
# ------------------------------------------------------------------
MODEL_PATH = "xgb_model.json"        # your downloaded XGBoost model
LABELS_PATH = "label_encoder.pkl"    # your downloaded LabelEncoder
# If the model is too big for GitHub, upload it as a GitHub Release asset
# and paste its direct download link here (otherwise leave empty).
MODEL_URL = ""

COLS = [f"pixel_{i}" for i in range(576)]

st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️")


# ------------------------------------------------------------------
# Load model + labels once (cached across visitors)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = MODEL_PATH
    if not os.path.exists(path) and MODEL_URL:
        path = os.path.join("/tmp", os.path.basename(MODEL_PATH))
        if not os.path.exists(path):
            urllib.request.urlretrieve(MODEL_URL, path)
    booster = xgb.Booster()
    booster.load_model(path)
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)
    return booster, [str(c) for c in le.classes_]


# ------------------------------------------------------------------
# Preprocessing (identical to your notebook)
# ------------------------------------------------------------------
def otsu_threshold(gray_u8):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(sigma_b))


def preprocess_image(file, margin=0.10, thicken=0.015):
    img = ImageOps.exif_transpose(Image.open(file)).convert("L")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    g = np.array(img, dtype=np.uint8)

    thr = otsu_threshold(g)
    ink = g < thr
    if ink.mean() > 0.5:
        ink = ~ink
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return None

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)) + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    bg = int(np.median(g[~ink])) if (~ink).any() else 255
    canvas = Image.new("L", (side, side), color=bg)
    canvas.paste(img, (-(cx - side // 2), -(cy - side // 2)))

    k = int(side * thicken) | 1
    if k >= 3:
        canvas = canvas.filter(ImageFilter.MinFilter(k))

    small = canvas.resize((24, 24), Image.Resampling.LANCZOS)
    arr = np.array(small, dtype=np.float32)
    paper = np.median(arr[arr >= np.percentile(arr, 50)])
    arr = np.clip(arr / (paper + 1e-8), 0, 1)
    lo = arr.min()
    return np.clip((arr - lo) / (1 - lo + 1e-8), 0, 1) ** 1.5


def predict_top3(file, booster, classes):
    arr = preprocess_image(file)
    if arr is None:
        return None, None
    df_in = pd.DataFrame(arr.reshape(1, -1), columns=COLS)
    raw = booster.predict(xgb.DMatrix(df_in))[0]
    if not np.isclose(np.sum(raw), 1.0):
        e = np.exp(raw - np.max(raw))
        probs = e / e.sum()
    else:
        probs = raw
    top = np.argsort(probs)[::-1][:3]
    return arr, [(classes[i], float(probs[i])) for i in top]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("✍️ Handwritten Character Recognition")
st.markdown(
    """
**Tips for best accuracy**
1. Use a sharply focused, close-up photo
2. Crop out extra background
3. Write **one clear, dark character** on plain light paper
"""
)

booster, classes = load_model()

tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Take photo"])
with tab_upload:
    uploaded = st.file_uploader("Upload a character image", type=["png", "jpg", "jpeg", "webp"])
with tab_camera:
    snapped = st.camera_input("Take a photo of the character")

file = uploaded or snapped

if file is not None:
    st.image(file, caption="Your image", width=250)
    with st.spinner("Thinking..."):
        arr, results = predict_top3(file, booster, classes)

    if results is None:
        st.error("No dark strokes found. Try a darker, better-lit, cropped photo.")
    else:
        st.subheader("Top 3 guesses")
        cols = st.columns(3)
        for col, (ch, p) in zip(cols, results):
            col.metric(label=f"{p:.1%} confidence", value=ch)
        st.progress(results[0][1])
        with st.expander("What the model actually sees (24×24)"):
            st.image(arr, width=160, clamp=True)
