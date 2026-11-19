# Classification Automatique d'Images selon les Disciplines Sportives (CNN)

Projet académique de Deep Learning (Niveau M2 / 5ème année d'école d'ingénieurs).

---

## 1. Présentation Générale

Ce projet implémente un système complet de vision par ordinateur et d'apprentissage profond (*Deep Learning*) capable de reconnaître automatiquement la discipline sportive représentée sur une photographie. Le système est conçu autour d'une architecture à base de Réseaux de Neurones Convolutifs (**CNN**) exploitant le mécanisme d'apprentissage par transfert (**Transfer Learning**) avec le modèle pré-entraîné **MobileNetV2**.

Le système classifie les images parmi trois disciplines majeures :
- **Football** (Association Football / Soccer)
- **Basketball**
- **Tennis**

Le modèle analyse et extrait automatiquement les caractéristiques visuelles clés : ballons spécifiques (rond, strié, feutre jaune), équipements (paniers, raquettes, filets), tenues des joueurs, et environnements (pelouse verte, parquet intérieur, court en terre battue ou dur).

---

## 2. Objectifs Pédagogiques & Techniques

- **Conception et implémentation d'un pipeline de Deep Learning complet** : de la collecte et du prétraitement des données jusqu'au déploiement d'une interface d'inférence.
- **Maîtrise du Transfer Learning** : réutilisation des représentations visuelles de haut niveau apprises sur le dataset de référence *ImageNet* via MobileNetV2.
- **Régularisation et optimisation** : mise en place de la data augmentation (retournements, rotations, zooms modérés), de couches de *Dropout* et de callbacks dynamiques (*EarlyStopping*, *ReduceLROnPlateau*, *ModelCheckpoint*).
- **Évaluation scientifique rigoureuse** : mesure des performances sur un jeu de test indépendant et jamais vu (*Accuracy*, *Precision*, *Recall*, *F1-score*, Matrice de confusion normalisée).
- **Démonstration interactive** : possibilité de charger une nouvelle image locale ou de tester via des widgets interactifs.

---

## 3. Technologies et Dépendances

Le projet est développé exclusivement en **Python 3.11** sous forme d'un **Jupyter Notebook** interactif et modulaire :

- **Deep Learning / Frameworks** : TensorFlow 2.18, Keras 3
- **Calcul scientifique & Tableaux** : NumPy, Pandas
- **Machine Learning & Métriques** : Scikit-Learn
- **Visualisation graphique** : Matplotlib, Seaborn
- **Traitement d'images** : Pillow (PIL)
- **Environnement & Interactivité** : Jupyter Notebook, ipykernel, ipywidgets

---

## 4. Structure du Projet

```text
Sport_Classification_CNN/
│
├── notebooks/
│   └── classification_sports.ipynb      # Notebook Jupyter complet (12 parties)
│
├── dataset/
│   ├── train/                           # Données d'entraînement (70%)
│   │   ├── football/
│   │   ├── basketball/
│   │   └── tennis/
│   │
│   ├── validation/                      # Données de validation (15%)
│   │   ├── football/
│   │   ├── basketball/
│   │   └── tennis/
│   │
│   └── test/                            # Données de test indépendantes (15%)
│       ├── football/
│       ├── basketball/
│       └── tennis/
│
├── models/
│   └── sport_classifier.keras           # Modèle entraîné sauvegardé au format Keras natif
│
├── results/
│   ├── accuracy_loss.png                # Courbes d'entraînement et validation (loss/accuracy)
│   ├── confusion_matrix.png             # Matrice de confusion sur le jeu de test
│   ├── predictions.png                  # Visualisation des prédictions avec scores de confiance
│   └── classification_report.csv        # Rapport de classification tabulaire (précision, rappel, f1)
│
├── scripts/
│   ├── prepare_dataset.py               # Script automatisé de téléchargement et structuration
│   └── run_training.py                  # Script d'entraînement autonome (CPU/GPU)
│
├── requirements.txt                     # Liste des dépendances pip
├── README.md                            # Documentation utilisateur et guide de démonstration
└── rapport_projet.md                    # Rapport académique complet et détaillé
```

---

## 5. Installation et Configuration

### Prérequis
- Python 3.11 ou version compatible installée sur le système.
- Accès terminal (PowerShell sous Windows ou Bash sous Linux/macOS).

### Étape 1 : Cloner ou ouvrir le dossier du projet
```powershell
cd c:\Users\DELL\Desktop\Sport_Classification_CNN
```

### Étape 2 : Installer les dépendances
```powershell
pip install -r requirements.txt
```

---

## 6. Préparation du Jeu de Données (Dataset)

Le dataset est constitué de photographies réelles issues de sources ouvertes vérifiées (Wikimedia Commons & HuggingFace Gerry Piosenka Sports Dataset).

### Téléchargement automatique
Un script Python préconfiguré permet de créer la structure et de télécharger les images authentiques vérifiées :
```powershell
python scripts/prepare_dataset.py
```

### Répartition du jeu de données
Afin d'éviter tout surapprentissage et garantir une rigueur académique, les données sont strictement réparties selon les proportions suivantes :
- **70% Entraînement** (`dataset/train/`) : utilisé pour ajuster les poids du classificateur.
- **15% Validation** (`dataset/validation/`) : utilisé pour monitorer la généralisation et ajuster le taux d'apprentissage.
- **15% Test** (`dataset/test/`) : strictement isolé et réservé à l'évaluation finale.

Une graine aléatoire fixe (`seed=42`) est appliquée pour assurer la parfaite reproductibilité des découpages.

---

## 7. Utilisation du Notebook Jupyter

### Étape 1 : Démarrer le serveur Jupyter
Depuis le répertoire du projet, exécutez :
```powershell
jupyter notebook notebooks/classification_sports.ipynb
```
ou simplement :
```powershell
jupyter notebook
```
puis naviguez dans le dossier `notebooks/` et ouvrez `classification_sports.ipynb`.

### Étape 2 : Exécution ordonnée des cellules
Pour une exécution fluide et conforme :
1. Dans le menu de Jupyter, cliquez sur **Kernel -> Restart & Clear Output**.
2. Exécutez les cellules séquentiellement du début à la fin (**Cell -> Run All** ou `Maj + Entrée` sur chaque cellule).

### Les 12 Parties du Notebook :
- **PARTIE 1 — Introduction** : Contexte, problématique, objectifs, théorie des CNN et du Transfer Learning.
- **PARTIE 2 — Importation des bibliothèques** : Chargement de TensorFlow, Keras, Matplotlib, détection GPU/CPU et fixation des graines aléatoires.
- **PARTIE 3 — Exploration des données** : Comptage par classe, visualisation d'échantillons réels, graphique de distribution et validation d'intégrité des fichiers.
- **PARTIE 4 — Prétraitement et Data Augmentation** : Redimensionnement $224 \times 224$, batching dynamique (32), normalisation MobileNetV2 $[-1, 1]$, augmentation pour l'entraînement (rotations légères, retournements horizontaux, zooms).
- **PARTIE 5 — Architecture CNN & Transfer Learning** : Intégration du backbone MobileNetV2 pré-entraîné sur ImageNet, GlobalAveragePooling2D, Dropout (0.3), couche Dense Softmax (3 classes).
- **PARTIE 6 — Compilation du modèle** : Optimiseur Adam ($\text{lr}=10^{-4}$), fonction de perte Categorical Crossentropy, métrique Accuracy.
- **PARTIE 7 — Entraînement** : Entraînement sur 10 époques avec callbacks intelligents (*EarlyStopping*, *ModelCheckpoint*, *ReduceLROnPlateau*).
- **PARTIE 8 — Visualisation des résultats** : Tracé haute résolution des courbes d'accuracy et de loss sauvegardées dans `results/accuracy_loss.png`.
- **PARTIE 9 — Évaluation sur le jeu de test** : Matrice de confusion Seaborn, rapport de classification complet (Precision, Recall, F1-Score).
- **PARTIE 10 — Prédiction d'une nouvelle image** : Fonction Python `predire_sport(image_path)` avec affichage de la photo et des probabilités par classe.
- **PARTIE 11 — Démonstration interactive** : Interface interactive `ipywidgets` avec bouton de prédiction et sélecteur de fichier (avec solution de repli locale).
- **PARTIE 12 — Conclusion académique** : Bilan technique, analyse critique des performances, limites observées et perspectives d'amélioration.

---

## 8. Tester une Nouvelle Image

Vous pouvez tester n'importe quelle image locale de sport (JPEG ou PNG) grâce à la fonction `predire_sport` :

```python
from notebooks.classification_sports import predire_sport  # ou directement dans le notebook

# Exemple :
predire_sport("dataset/test/tennis/tennis_01.jpg")
```

Sortie générée :
- Affichage de l'image analysée.
- Classe prédite (ex. : **Tennis**).
- Score de confiance principal (ex. : **98.42%**).
- Barres de probabilités détaillées pour chacune des 3 classes :
  - Football : 0.85%
  - Basketball : 0.73%
  - Tennis : 98.42%

---

## 9. Guide de Présentation devant le Professeur

1. **Introduction rapide (2 minutes)** :
   - Présenter le sujet : classification supervisée multi-classe de disciplines sportives.
   - Justifier le choix du **Transfer Learning avec MobileNetV2** (efficacité computationnelle adaptée aux stations de travail étudiantes, profondeur et richesse des représentations d'ImageNet).
2. **Exploration du Dataset (1 minute)** :
   - Montrer la distribution équilibrée des classes (Football, Basketball, Tennis).
   - Expliquer l'importance du cloisonnement strict : 70% Train, 15% Val, 15% Test.
3. **Architecture et Stratégie d'Entraînement (2 minutes)** :
   - Expliquer pourquoi le backbone est gelé (*frozen*) lors de la première phase.
   - Montrer l'action de la *Data Augmentation* pour éviter le surapprentissage.
   - Détailler le rôle des callbacks (*EarlyStopping* pour stopper au pic de validation, *ReduceLROnPlateau* pour affiner la convergence).
4. **Analyse des Résultats et Métriques (2 minutes)** :
   - Commenter la matrice de confusion : identifier les confusions éventuelles et expliquer leurs causes physiques (couleur des maillots, angles de vue, dynamique des joueurs).
   - Présenter l'accuracy finale et les scores F1 par discipline.
5. **Démonstration en direct (1 minute)** :
   - Exécuter la fonction `predire_sport` sur une image de test jamais vue.
   - Montrer la distribution des probabilités softmax.

---

## 10. Résolution des Problèmes Fréquents (Troubleshooting)

- **Erreur de mémoire vive (OOM - Out of Memory)** :
  - *Solution* : Réduire la taille de batch (`BATCH_SIZE = 16` au lieu de 32) dans la cellule de prétraitement.
- **Absence de GPU détecté** :
  - *Comportement normal* : Le code détecte automatiquement si un GPU CUDA est absent et bascule sur le CPU de manière transparente. Grâce à la compacité de MobileNetV2, l'entraînement s'effectue en quelques minutes seulement sur CPU.
- **Problème d'affichage des widgets `ipywidgets`** :
  - *Solution* : La cellule de la Partie 11 intègre automatiquement un système de repli (*fallback*) standard utilisant un chemin de fichier local sans bloquer l'exécution.
- **Images corrompues** :
  - Le pipeline de chargement intègre une routine de vérification Pillow (`PIL.Image.verify()`) éliminant automatiquement tout fichier altéré.
