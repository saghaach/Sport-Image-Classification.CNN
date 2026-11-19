"""
Script complet d'entraînement et d'évaluation du modèle de classification des sports.
Exécute le pipeline de bout en bout et génère les artefacts de résultats réels :
- models/sport_classifier.keras
- results/accuracy_loss.png
- results/confusion_matrix.png
- results/predictions.png
- results/classification_report.csv
"""

import os
import sys
import tempfile
import pathlib
import json
import random
import numpy as np
import pandas as pd
from PIL import Image

# Configuration du cache Matplotlib
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Reproductibilité
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, applications, optimizers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

tf.random.set_seed(RANDOM_SEED)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Adapté pour ~72 images d'entraînement (4-5 batches par époque)
EPOCHS = 10

def main():
    print("=========================================================")
    print("  LANCEMENT DE L'ENTRAÎNEMENT DU CLASSIFICATEUR DE SPORTS")
    print("=========================================================")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    
    # 1. Chargement des données
    print("\n--- Chargement des ensembles de données ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Classes détectées : {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Construction de l'architecture MobileNetV2 avec Transfer Learning
    print("\n--- Construction du modèle MobileNetV2 Transfer Learning ---")
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", name="aug_flip"),
        layers.RandomRotation(0.1, name="aug_rot"),
        layers.RandomZoom(0.1, name="aug_zoom"),
    ], name="data_augmentation")

    base_model = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    print(f"Backbone MobileNetV2 chargé. Nombre de couches gelées : {len(base_model.layers)}")

    inputs = layers.Input(shape=(224, 224, 3), name="entree_image")
    x = data_augmentation(inputs)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout_reg")(x)
    outputs = layers.Dense(len(class_names), activation='softmax', name="tete_dense_softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Sport_Classifier_MobileNetV2")
    
    # Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. Callbacks & Entraînement
    checkpoint_file = MODELS_DIR / "sport_classifier.keras"
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(checkpoint_file), monitor='val_loss', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    ]

    print(f"\n--- Début de l'entraînement ({EPOCHS} époques maximales) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    print(f"\n✓ Meilleur modèle sauvegardé avec succès dans : {checkpoint_file}")

    # 4. Tracé des courbes d'apprentissage
    print("\n--- Génération des courbes d'apprentissage (results/accuracy_loss.png) ---")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'o-', color='#1f77b4', linewidth=2.5, label="Entraînement (Train)")
    plt.plot(epochs_range, val_acc, 's--', color='#ff7f0e', linewidth=2.5, label="Validation (Val)")
    plt.title("Exactitude (Accuracy) par Époque", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Époque", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'o-', color='#1f77b4', linewidth=2.5, label="Entraînement (Train)")
    plt.plot(epochs_range, val_loss, 's--', color='#d62728', linewidth=2.5, label="Validation (Val)")
    plt.title("Perte (Cross-Entropy Loss) par Époque", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Époque", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = RESULTS_DIR / "accuracy_loss.png"
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique d'apprentissage sauvegardé sous : {curves_path}")

    # 5. Évaluation rigoureuse sur le jeu de test
    print("\n--- Évaluation finale sur le jeu de test indépendant ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"✓ Test Accuracy : {test_acc * 100:.2f} %")
    print(f"✓ Test Loss     : {test_loss:.4f}")

    y_true_list = []
    y_pred_probs_list = []
    test_images_collector = []

    for images_batch, labels_batch in test_ds:
        preds = model.predict(images_batch, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels_batch.numpy())
        test_images_collector.append(images_batch.numpy())

    y_true = np.argmax(np.concatenate(y_true_list, axis=0), axis=1)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Rapport de classification
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    print("\n=== RAPPORT DE CLASSIFICATION DU TEST SET ===")
    print(df_report.round(4))

    report_csv_path = RESULTS_DIR / "classification_report.csv"
    df_report.to_csv(report_csv_path)
    print(f"✓ Rapport de classification exporté dans : {report_csv_path}")

    # Matrice de confusion
    print("\n--- Génération de la matrice de confusion (results/confusion_matrix.png) ---")
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Matrice de Confusion Normalisée (Jeu de Test)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Discipline Prédite", fontsize=11, fontweight='bold')
    plt.ylabel("Discipline Réelle", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Matrice de confusion sauvegardée sous : {cm_path}")

    # Échantillon de prédictions visuelles
    print("\n--- Génération des visualisations de prédictions (results/predictions.png) ---")
    all_test_images = np.concatenate(test_images_collector, axis=0)
    nb_samples = min(6, len(all_test_images))

    plt.figure(figsize=(14, 8))
    for i in range(nb_samples):
        img = all_test_images[i].astype("uint8")
        true_lbl = class_names[y_true[i]]
        pred_lbl = class_names[y_pred[i]]
        conf = y_pred_probs[i][y_pred[i]] * 100
        
        color = "green" if true_lbl == pred_lbl else "red"
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Prédit : {pred_lbl.capitalize()} ({conf:.1f}%)\nVrai : {true_lbl.capitalize()}", 
                  fontsize=11, fontweight='bold', color=color)
        plt.axis("off")

    plt.suptitle("Échantillon de Prédictions sur le Jeu de Test Indépendant", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    preds_path = RESULTS_DIR / "predictions.png"
    plt.savefig(preds_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prédictions sauvegardées sous : {preds_path}")

    # Résumé final JSON pour injecter dans le rapport académique
    metrics_summary = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_avg_precision": float(report_dict["macro avg"]["precision"]),
        "macro_avg_recall": float(report_dict["macro avg"]["recall"]),
        "macro_avg_f1": float(report_dict["macro avg"]["f1-score"]),
        "class_metrics": {
            c: {
                "precision": float(report_dict[c]["precision"]),
                "recall": float(report_dict[c]["recall"]),
                "f1_score": float(report_dict[c]["f1-score"]),
                "support": int(report_dict[c]["support"])
            }
            for c in class_names
        }
    }
    summary_path = RESULTS_DIR / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Métriques réelles enregistrées dans : {summary_path}")
    print("\n=========================================================")
    print("  ENTRAÎNEMENT ET ÉVALUATION TERMINÉS AVEC SUCCÈS !")
    print("=========================================================")

if __name__ == "__main__":
    main()
