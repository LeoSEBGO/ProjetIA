# 🔮 Analyse et Prédiction de Maladies Cardiaques

Ce projet utilise le Machine Learning pour prédire les risques de maladies cardiaques à partir de données médicales. Il comprend une interface web interactive développée avec Streamlit.

## 📋 Description

Le projet vise à :
- Analyser les facteurs de risque des maladies cardiaques
- Entraîner différents modèles de classification
- Fournir une interface utilisateur intuitive pour la prédiction
- Visualiser les performances des modèles

## 🚀 Fonctionnalités

- **Analyse Exploratoire des Données** : Visualisations interactives et futuristes des données
- **Modèles de Classification** : Implémentation de plusieurs algorithmes (Régression Logistique, KNN, SVM, etc.)
- **Interface Web Interactive** : Application Streamlit avec design futuriste
- **Évaluation des Modèles** : Métriques détaillées et visualisations comparatives
- **Prédiction en Temps Réel** : Interface intuitive pour les nouvelles prédictions

## 📦 Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_DOSSIER]
```

2. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 🎮 Utilisation

1. Entraîner les modèles :
```bash
python model_training.py
```

2. Lancer l'application :
```bash
streamlit run app.py
```

## 📊 Structure du Projet

```
.
├── app.py              # Application Streamlit principale
├── data_loader.py      # Fonctions de chargement et prétraitement des données
├── model_training.py   # Entraînement et évaluation des modèles
├── requirements.txt    # Dépendances du projet
├── best_model.pkl      # Meilleur modèle sauvegardé
├── model_results.pkl   # Résultats des performances des modèles
└── scaler.pkl          # StandardScaler sauvegardé
```

## 🧠 Modèles Implémentés

- Régression Logistique
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost

## 📈 Métriques d'Évaluation

- Accuracy (Précision globale)
- Precision (Précision)
- Recall (Rappel)
- F1-Score
- AUC-ROC

## 🎨 Design

L'interface utilise un design futuriste avec :
- Thème sombre
- Accents néon
- Visualisations interactives
- Animations fluides
- Interface responsive

## 👥 Auteurs

- Développé sous la supervision du Dr Arthur Sawadogo
- IFOAD - Institut de Formation Ouverte et à Distance

## 🙏 Remerciements

- Dataset Heart Disease UCI
- Streamlit pour l'interface web
- Scikit-learn pour les algorithmes de Machine Learning
- Plotly pour les visualisations interactives 

## 📝 Variables Utilisées

1. `age` : Âge du patient
2. `sex` : Sexe (0 = Femme, 1 = Homme)
3. `cp` : Type de douleur thoracique
4. `trestbps` : Pression artérielle au repos
5. `chol` : Cholestérol sérique
6. `fbs` : Glycémie à jeun
7. `restecg` : Résultats électrocardiographiques
8. `thalach` : Fréquence cardiaque maximale
9. `exang` : Angine induite par l'exercice
10. `oldpeak` : Dépression ST induite par l'exercice
11. `slope` : Pente du segment ST
12. `ca` : Nombre de vaisseaux colorés
13. `thal` : Thalassémie 
