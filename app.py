"""
Application web Streamlit — Classification d'images sportives par CNN
Modèle : MobileNetV2 + Transfer Learning (sport_classifier.keras)
Classes : basketball, football, tennis
"""

import io
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sport_classifier.keras")
IMG_SIZE   = (224, 224)
CLASSES    = ["basketball", "football", "tennis"]

SPORT_META = {
    "basketball": {"label": "Basketball", "emoji": "🏀", "color": "#F97316"},
    "football":   {"label": "Football",   "emoji": "⚽", "color": "#22C55E"},
    "tennis":     {"label": "Tennis",     "emoji": "🎾", "color": "#EAB308"},
}

# ─────────────────────────────────────────────
# Chargement du modèle (mis en cache)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle CNN...")
def charger_modele():
    return tf.keras.models.load_model(MODEL_PATH)

model = charger_modele()

# ─────────────────────────────────────────────
# Page config & style
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SportVision CNN — Classification de Sports",
    page_icon="🏅",
    layout="wide",
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .stApp { background: transparent; }
    h1 { color: #f8fafc !important; }
    .result-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    .pred-name { font-size: 2.2rem; font-weight: 700; color: #f8fafc; }
    .pred-conf { font-size: 1.1rem; color: #94a3b8; }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white; font-weight: 600; border: none; border-radius: 10px;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────
st.title("🏅 SportVision CNN")
st.subheader("Classification automatique d'images sportives — Football · Basketball · Tennis")
st.caption("Modèle MobileNetV2 entraîné par transfer learning — 224×224 px")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Importer une image")
    image_file = st.file_uploader(
        "Glissez-déposez ou sélectionnez une image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if image_file is not None:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Image importée", use_container_width=True)

        # Prétraitement identique à l'entraînement
        arr = img.resize(IMG_SIZE)
        arr = img_to_array(arr)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        with st.spinner("Analyse en cours..."):
            preds = model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(preds))
        classe = CLASSES[idx]
        meta = SPORT_META[classe]
        confiance = float(preds[idx]) * 100
        probabilites = {CLASSES[i]: float(preds[i]) * 100 for i in range(len(CLASSES))}
    else:
        st.info("Importez une image pour lancer la classification.")
        probabilites = None
        classe = None
        confiance = 0.0
        meta = None

with col_right:
    st.markdown("### 🔍 Résultat de la classification")

    if probabilites is None:
        st.markdown(
            """
            <div class="result-card">
                <div style="font-size:3rem;">🖼️</div>
                <div class="pred-name">En attente</div>
                <div class="pred-conf">Importez une image pour voir la prédiction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:2rem;">{meta['emoji']}</div>
                <div class="pred-name">{meta['label']}</div>
                <div class="pred-conf">Confiance : <b>{confiance:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Probabilités par catégorie")
        for cls in CLASSES:
            m = SPORT_META[cls]
            pct = probabilites[cls]
            st.progress(min(pct / 100, 1.0), text=f"{m['emoji']} {m['label']} — {pct:.2f}%")

        st.markdown("#### 📋 Détail")
        st.dataframe(
            {
                "Sport": [SPORT_META[c]["label"] for c in CLASSES],
                "Probabilité (%)": [f"{probabilites[c]:.2f}" for c in CLASSES],
                "Statut": ["✅ Prédit" if c == classe else "" for c in CLASSES],
            },
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Projet académique — CNN MobileNetV2 · Test Accuracy : 83.33% · "
    "Entraînement et évaluation disponibles dans le notebook Jupyter."
)
