"""
Streamlit app publik untuk klasifikasi Cats vs Dogs.
Model auto-load dari Hugging Face: aisyahnoviani16/anjingKucing/model_vgg16_clean.keras
User cukup upload gambar, app akan prediksi & tampilkan hasil.
"""

import os, io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

#CONFIG
HF_REPO = "aisyahnoviani16/anjingKucing"
HF_FILENAME = "model_vgg16_clean.keras"
IMAGE_SIZE = (128, 128)
CLASS_MAP = {0: "Cat", 1: "Dog"}

#STREAMLIT CONFIG
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐱🐶")
st.markdown("""

<style>
.stApp { background: linear-gradient(180deg, #fffaf8 0%, #fff0f6 100%); }
.block-container { border-radius: 14px; box-shadow: 0 6px 22px rgba(168,47,103,0.08); padding-top: 4rem; }
.stButton>button { background: linear-gradient(90deg,#ff6fa3,#ff3b84); color:white; font-weight:700; border-radius:10px; }
.stProgress > div > div > div { background-color: #ff6fa3; }
</style>
""", unsafe_allow_html=True)

st.title("🐱🐶 Cats vs Dogs Classifier")
st.caption("Upload gambar kucing atau anjing, model akan memprediksi otomatis.")

# LOAD MODEL
@st.cache_resource
def load_model_from_hf():
    local_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="model")
    return tf.keras.models.load_model(local_path, compile=False)

with st.spinner(" Loading model dari Hugging Face..."):
    model = load_model_from_hf()

st.success("✅ Model siap digunakan! Silakan upload gambar.")

#HELPERS
def preprocess_pil_image(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_batch(model, images):
    results = []
    for img in images:
        x = preprocess_pil_image(img)
        pred = model.predict(x, verbose=0)
        if pred.shape[1] == 1:  # sigmoid
            prob = float(pred[0][0])
            idx = 1 if prob >= 0.5 else 0
            label = CLASS_MAP[idx]
            score = prob if idx == 1 else 1 - prob
            probs = [1 - prob, prob]  # [Cat, Dog]
        else:  # softmax
            probs = tf.nn.softmax(pred, axis=1).numpy()[0]
            idx = int(np.argmax(probs))
            label = CLASS_MAP[idx]
            score = float(probs[idx])
        results.append({"label": label, "score": score, "probs": probs})
    return results

# MAIN UI
st.subheader("Cara Pakai !")
st.markdown("""
1. Klik **Upload** untuk memilih 1 atau lebih gambar.  
2. Tunggu sebentar, model akan memprediksi otomatis.  
3. Lihat hasil prediksi & confidence score.  
4. (Opsional) Download hasil dalam format CSV.  
""")

uploaded_files = st.file_uploader(
    "📂 Upload gambar (jpg/png)", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

if uploaded_files:
    imgs, names = [], []
    for f in uploaded_files:
        pil = Image.open(f)
        imgs.append(pil.copy())
        names.append(f.name)

    results = predict_batch(model, imgs)
    st.success(f"{len(imgs)} gambar berhasil diprediksi!")

    cols = st.columns(min(3, len(imgs)))
    pred_rows = []
    for i, (img, name, res) in enumerate(zip(imgs, names, results)):
        with cols[i % len(cols)]:
            st.image(img, caption=name, use_column_width=True)
            st.markdown(f"**Prediksi:** `{res['label']}` — Confidence: **{res['score']*100:.1f}%**")
            st.progress(int(res['score']*100))

            # # Visualisasi bar chart confidence
            # fig, ax = plt.subplots(figsize=(2.5,1.5))
            # ax.bar(CLASS_MAP.values(), res["probs"], color=["#6fa8dc","#ff6fa3"])
            # ax.set_ylim(0,1)
            # ax.set_ylabel("Probabilitas")
            # st.pyplot(fig)

            pred_rows.append({"filename": name, "label": res['label'], "confidence": res['score']})

    df_out = pd.DataFrame(pred_rows)
    st.download_button("💾 Download hasil (CSV)", df_out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

else:
    st.info("Upload gambar untuk memulai prediksi.")

#Disclaimer
st.write("---")
st.caption("⚠️ Aplikasi ini dibuat untuk tujuan demo/edukasi. Hasil prediksi tidak menggantikan penilaian profesional.")
