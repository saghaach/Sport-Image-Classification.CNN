"""
Script d'exécution complète du projet CNN de Classification des Sports
Génère tous les résultats : entraînement, évaluation, visualisations, prédictions
"""

import os
import sys
import tempfile
import warnings

# Configuration environnement
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

print("=" * 65)
print("   CLASSIFICATION AUTOMATIQUE DES SPORTS PAR CNN")
print("   MobileNetV2 + Transfer Learning")
print("=" * 65)

# ─────────────────────────────────────────────
# PARTIE 2 — Importation des bibliothèques
# ─────────────────────────────────────────────
print("\n[PARTIE 2] Importation des bibliothèques...")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json
import shutil
import random
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score, f1_score)
from PIL import Image

print(f"  ✓ TensorFlow     : {tf.__version__}")
print(f"  ✓ Keras          : {keras.__version__}")
print(f"  ✓ NumPy          : {np.__version__}")
print(f"  ✓ Pandas         : {pd.__version__}")
print(f"  ✓ Matplotlib     : {matplotlib.__version__}")
print(f"  ✓ Seaborn        : {sns.__version__}")
print(f"  ✓ Scikit-learn   : OK")
print(f"  ✓ Pillow         : {Image.__version__}")

# Graine pour reproductibilité
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
print(f"  ✓ Graine aléatoire : {SEED}")

# Détection GPU
gpus = tf.config.list_physical_devices('GPU')
device = "GPU" if gpus else "CPU"
print(f"  ✓ Dispositif : {device}")

# ─────────────────────────────────────────────
# CHEMINS DU PROJET
# ─────────────────────────────────────────────
BASE_DIR    = Path("C:/Users/DELL/Desktop/Sport_Classification_CNN")
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "validation"
TEST_DIR    = DATASET_DIR / "test"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES    = ['basketball', 'football', 'tennis']
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 10
MODEL_PATH = MODELS_DIR / "sport_classifier.keras"

# ─────────────────────────────────────────────
# PARTIE 3 — Exploration des données
# ─────────────────────────────────────────────
print("\n[PARTIE 3] Exploration des données...")

def compter_images(split_dir):
    counts = {}
    total = 0
    for cls in CLASSES:
        cls_path = split_dir / cls
        if cls_path.exists():
            imgs = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
            counts[cls] = len(imgs)
            total += len(imgs)
        else:
            counts[cls] = 0
    return counts, total

train_counts, train_total = compter_images(TRAIN_DIR)
val_counts,   val_total   = compter_images(VAL_DIR)
test_counts,  test_total  = compter_images(TEST_DIR)

print(f"\n  Distribution du dataset :")
print(f"  {'Classe':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
print(f"  {'-'*47}")
for cls in CLASSES:
    t = train_counts[cls] + val_counts[cls] + test_counts[cls]
    print(f"  {cls:<15} {train_counts[cls]:>8} {val_counts[cls]:>8} {test_counts[cls]:>8} {t:>8}")
grand_total = train_total + val_total + test_total
print(f"  {'TOTAL':<15} {train_total:>8} {val_total:>8} {test_total:>8} {grand_total:>8}")

# Graphique distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribution du Dataset par Classe et par Split", fontsize=14, fontweight='bold')
splits = ['Entraînement', 'Validation', 'Test']
counts_list = [train_counts, val_counts, test_counts]
colors = ['#2196F3', '#FF9800', '#4CAF50']

for ax, split, counts, color in zip(axes, splits, counts_list, colors):
    bars = ax.bar(CLASSES, [counts[c] for c in CLASSES], color=color, edgecolor='black', alpha=0.85)
    ax.set_title(f"{split}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Discipline sportive")
    ax.set_ylabel("Nombre d'images")
    ax.set_ylim(0, max(train_counts.values()) + 5)
    for bar, val in zip(bars, [counts[c] for c in CLASSES]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "distribution_dataset.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✓ Graphique distribution sauvegardé")

# ─────────────────────────────────────────────
# PARTIE 4 — Prétraitement & Augmentation
# ─────────────────────────────────────────────
print("\n[PARTIE 4] Prétraitement et augmentation des données...")

def verifier_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def charger_dataset(split_dir):
    images, labels = [], []
    corrompues = 0
    for idx, cls in enumerate(CLASSES):
        cls_path = split_dir / cls
        if not cls_path.exists():
            continue
        fichiers = (list(cls_path.glob("*.jpg")) + 
                    list(cls_path.glob("*.jpeg")) + 
                    list(cls_path.glob("*.png")))
        for f in fichiers:
            if not verifier_image(f):
                corrompues += 1
                continue
            try:
                img = load_img(f, target_size=IMG_SIZE)
                arr = img_to_array(img)
                arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                images.append(arr)
                labels.append(idx)
            except Exception as e:
                corrompues += 1
    if corrompues > 0:
        print(f"  ⚠ {corrompues} image(s) corrompue(s) ignorée(s)")
    return np.array(images), np.array(labels)

print("  Chargement train...")
X_train, y_train = charger_dataset(TRAIN_DIR)
print(f"  ✓ Train : {X_train.shape[0]} images")

print("  Chargement validation...")
X_val, y_val = charger_dataset(VAL_DIR)
print(f"  ✓ Validation : {X_val.shape[0]} images")

print("  Chargement test...")
X_test, y_test = charger_dataset(TEST_DIR)
print(f"  ✓ Test : {X_test.shape[0]} images")

# Augmentation uniquement sur train
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

# Datasets TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (train_dataset
                 .shuffle(buffer_size=len(X_train), seed=SEED)
                 .batch(BATCH_SIZE)
                 .map(lambda x, y: (data_augmentation(x, training=True), y))
                 .prefetch(tf.data.AUTOTUNE))

val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

print("  ✓ Datasets TensorFlow créés avec augmentation")

# ─────────────────────────────────────────────
# PARTIE 5 — Architecture CNN (MobileNetV2)
# ─────────────────────────────────────────────
print("\n[PARTIE 5] Construction du modèle MobileNetV2...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3), name="input_image")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(len(CLASSES), activation='softmax', name="classification")(x)

model = keras.Model(inputs, outputs, name="Sport_MobileNetV2")

total_params = model.count_params()
trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"  ✓ Modèle créé : MobileNetV2 + Tête de classification")
print(f"  ✓ Paramètres totaux    : {total_params:,}")
print(f"  ✓ Paramètres à entraîner : {trainable_params:,}")

# ─────────────────────────────────────────────
# PARTIE 6 — Compilation
# ─────────────────────────────────────────────
print("\n[PARTIE 6] Compilation du modèle...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("  ✓ Optimiseur : Adam (lr=0.001)")
print("  ✓ Perte      : Sparse Categorical Crossentropy")
print("  ✓ Métrique   : Accuracy")

# ─────────────────────────────────────────────
# PARTIE 7 — Entraînement
# ─────────────────────────────────────────────
print("\n[PARTIE 7] Entraînement du modèle...")
print(f"  Epochs max : {EPOCHS} | Batch size : {BATCH_SIZE}")
print(f"  Dispositif : {device}")
print("  " + "-" * 50)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

debut = datetime.now()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
duree = (datetime.now() - debut).seconds

print(f"\n  ✓ Entraînement terminé en {duree//60}m {duree%60}s")
print(f"  ✓ Meilleure val_accuracy : {max(history.history['val_accuracy']):.4f}")
print(f"  ✓ Modèle sauvegardé dans : {MODEL_PATH}")

# ─────────────────────────────────────────────
# PARTIE 8 — Visualisation des résultats
# ─────────────────────────────────────────────
print("\n[PARTIE 8] Génération des graphiques de performance...")

hist = history.history
epochs_range = range(1, len(hist['accuracy']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Performance du Modèle CNN — Classification des Sports", fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, hist['accuracy'], 'b-o', linewidth=2, markersize=5, label='Entraînement')
axes[0].plot(epochs_range, hist['val_accuracy'], 'r-s', linewidth=2, markersize=5, label='Validation')
axes[0].set_title("Précision (Accuracy) par Époque", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Précision")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)
axes[0].set_xticks(list(epochs_range))

# Loss
axes[1].plot(epochs_range, hist['loss'], 'b-o', linewidth=2, markersize=5, label='Entraînement')
axes[1].plot(epochs_range, hist['val_loss'], 'r-s', linewidth=2, markersize=5, label='Validation')
axes[1].set_title("Perte (Loss) par Époque", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Perte")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(epochs_range))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Graphique accuracy/loss sauvegardé")

# ─────────────────────────────────────────────
# PARTIE 9 — Évaluation sur le jeu de test
# ─────────────────────────────────────────────
print("\n[PARTIE 9] Évaluation sur le jeu de test...")

# Chargement du meilleur modèle
best_model = keras.models.load_model(str(MODEL_PATH))

# Prédictions
y_pred_proba = best_model.predict(test_dataset, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Métriques
test_acc   = accuracy_score(y_test, y_pred)
test_prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
test_rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
test_f1    = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │   RÉSULTATS SUR LE JEU DE TEST  │")
print(f"  ├─────────────────────────────────┤")
print(f"  │  Accuracy   : {test_acc*100:6.2f}%           │")
print(f"  │  Précision  : {test_prec*100:6.2f}%           │")
print(f"  │  Rappel     : {test_rec*100:6.2f}%           │")
print(f"  │  F1-Score   : {test_f1*100:6.2f}%           │")
print(f"  └─────────────────────────────────┘")

# Rapport de classification
report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(RESULTS_DIR / "classification_report.csv")
print(f"\n  Rapport de classification par classe :")
print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor='gray')
plt.title("Matrice de Confusion — Jeu de Test", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Étiquette Réelle", fontsize=12)
plt.xlabel("Étiquette Prédite", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Matrice de confusion sauvegardée")

# ─────────────────────────────────────────────
# PARTIE 10 — Prédiction d'une nouvelle image
# ─────────────────────────────────────────────
print("\n[PARTIE 10] Fonction de prédiction...")

def predire_sport(image_path, afficher=False):
    """
    Prédit la discipline sportive d'une image.
    
    Args:
        image_path (str): Chemin vers l'image à classifier
        afficher (bool): Afficher l'image avec matplotlib (True en Jupyter)
    
    Returns:
        dict: Classe prédite, confiance et probabilités
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    
    preds = best_model.predict(arr, verbose=0)[0]
    idx_pred = np.argmax(preds)
    classe_predite = CLASSES[idx_pred]
    confiance = float(preds[idx_pred]) * 100
    
    probs = {cls: round(float(preds[i]) * 100, 2) for i, cls in enumerate(CLASSES)}
    
    return {
        'classe': classe_predite,
        'confiance': round(confiance, 2),
        'probabilites': probs
    }

# Test sur des images de chaque classe
print("\n  Test de prédiction sur des images réelles :")
print(f"  {'Image':<35} {'Réel':<12} {'Prédit':<12} {'Confiance':>10}")
print(f"  {'-'*72}")

images_pred = []
labels_pred = []
preds_pred  = []

for cls in CLASSES:
    cls_path = TEST_DIR / cls
    if cls_path.exists():
        fichiers = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        if fichiers:
            f = fichiers[0]
            res = predire_sport(str(f))
            correct = "✓" if res['classe'] == cls else "✗"
            print(f"  {f.name:<35} {cls:<12} {res['classe']:<12} {res['confiance']:>8.1f}%  {correct}")
            images_pred.append(f)
            labels_pred.append(cls)
            preds_pred.append(res)

# Visualisation des prédictions
print("\n  Génération du graphique de prédictions...")
n_images = min(9, len(images_pred) * 3)
fichiers_all = []
labels_all = []

for cls in CLASSES:
    cls_path = TEST_DIR / cls
    if cls_path.exists():
        fichiers = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
        for f in fichiers[:3]:
            fichiers_all.append(f)
            labels_all.append(cls)

n = min(9, len(fichiers_all))
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
fig.suptitle("Prédictions du Modèle CNN sur le Jeu de Test", fontsize=14, fontweight='bold')

axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i in range(n):
    f = fichiers_all[i]
    cls_reel = labels_all[i]
    res = predire_sport(str(f))
    
    img = Image.open(f).resize((224, 224))
    correct = res['classe'] == cls_reel
    couleur = '#4CAF50' if correct else '#F44336'
    
    axes_flat[i].imshow(img)
    axes_flat[i].set_title(
        f"Réel : {cls_reel}\nPrédit : {res['classe']} ({res['confiance']:.1f}%)",
        color=couleur, fontweight='bold', fontsize=9
    )
    axes_flat[i].axis('off')
    
    # Bordure colorée
    for spine in axes_flat[i].spines.values():
        spine.set_edgecolor(couleur)
        spine.set_linewidth(3)

for i in range(n, len(axes_flat)):
    axes_flat[i].axis('off')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Graphique de prédictions sauvegardé")

# ─────────────────────────────────────────────
# SAUVEGARDE DES MÉTRIQUES
# ─────────────────────────────────────────────
metrics_summary = {
    "date_execution": datetime.now().isoformat(),
    "device": device,
    "epochs_realises": len(hist['accuracy']),
    "meilleure_val_accuracy": float(max(hist['val_accuracy'])),
    "test_accuracy": float(test_acc),
    "test_precision": float(test_prec),
    "test_recall": float(test_rec),
    "test_f1": float(test_f1),
    "rapport_par_classe": {
        cls: {
            "precision": round(report[cls]['precision'], 4),
            "recall": round(report[cls]['recall'], 4),
            "f1-score": round(report[cls]['f1-score'], 4),
            "support": int(report[cls]['support'])
        }
        for cls in CLASSES if cls in report
    },
    "dataset": {
        "train": train_counts,
        "validation": val_counts,
        "test": test_counts,
        "total": grand_total
    }
}

with open(RESULTS_DIR / "metrics_summary.json", "w", encoding='utf-8') as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
print(f"\n  ✓ Métriques sauvegardées dans results/metrics_summary.json")

# ─────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("   EXÉCUTION TERMINÉE — RÉSUMÉ FINAL")
print("=" * 65)
print(f"\n  Modèle         : MobileNetV2 + Transfer Learning")
print(f"  Dataset        : {grand_total} images ({', '.join([f'{v} {k}' for k,v in train_counts.items()])} en train)")
print(f"  Epochs         : {len(hist['accuracy'])}/{EPOCHS}")
print(f"  Dispositif     : {device}")
print(f"\n  ── MÉTRIQUES DE TEST ──")
print(f"  Test Accuracy  : {test_acc*100:.2f}%")
print(f"  Test F1-Score  : {test_f1*100:.2f}%")
print(f"  Test Précision : {test_prec*100:.2f}%")
print(f"  Test Rappel    : {test_rec*100:.2f}%")

print(f"\n  ── FICHIERS GÉNÉRÉS ──")
fichiers_generes = [
    "results/accuracy_loss.png",
    "results/confusion_matrix.png",
    "results/predictions.png",
    "results/classification_report.csv",
    "results/metrics_summary.json",
    "results/distribution_dataset.png",
    "models/sport_classifier.keras"
]
for f in fichiers_generes:
    path = BASE_DIR / f
    taille = path.stat().st_size / 1024 if path.exists() else 0
    statut = f"✓ ({taille:.0f} KB)" if path.exists() else "✗ MANQUANT"
    print(f"  {statut:>12}  {f}")

print("\n" + "=" * 65)
print("  Projet prêt pour présentation académique !")
print("=" * 65)
