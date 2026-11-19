"""
Test de la fonction d'inférence 'predire_sport' sur des photographies de test réelles.
"""

import os
import sys
import tempfile
import pathlib
import numpy as np
from PIL import Image

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sport_classifier.keras"
TEST_DIR = BASE_DIR / "dataset" / "test"

# Chargement du modèle entraîné
print(f"Chargement du modèle depuis : {MODEL_PATH}")
model = keras.models.load_model(str(MODEL_PATH))
class_names = ["basketball", "football", "tennis"]

def predire_sport(image_path, model_inst=model, classes=class_names):
    image_path = pathlib.Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {image_path}")
        
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    
    probs = model_inst.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(probs)
    classe_predite = classes[pred_idx]
    confiance = probs[pred_idx] * 100
    
    return {
        "image": image_path.name,
        "classe_predite": classe_predite,
        "confiance_pct": round(float(confiance), 2),
        "probabilites": {c: round(float(p) * 100, 2) for c, p in zip(classes, probs)}
    }

print("\n--- Test sur un échantillon de chaque discipline ---")
for cat in class_names:
    test_files = list((TEST_DIR / cat).glob("*.jpg"))
    if test_files:
        sample = test_files[0]
        res = predire_sport(sample)
        print(f"Image [{cat.upper()}] : {res['image']}")
        print(f"  -> Prédiction : {res['classe_predite'].upper()} avec {res['confiance_pct']}% de confiance")
        print(f"  -> Détail des probabilités : {res['probabilites']}\n")
