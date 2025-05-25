import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Chargement des données
print("Chargement des données...")
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                 names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

# Nettoyage des données
print("Nettoyage des données...")
df = df.replace('?', np.nan)
df = df.dropna()
df = df.astype(float)
df['target'] = (df['target'] > 0).astype(int)

# Séparation des features et de la target
X = df.drop('target', axis=1)
y = df['target']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde du scaler
joblib.dump(scaler, 'scaler.pkl')

# Initialisation des modèles
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Entraînement et évaluation des modèles
results = {}
best_score = 0
best_model = None

print("Entraînement des modèles...")
for name, model in models.items():
    print(f"\nEntraînement de {name}...")
    
    # Entraînement du modèle
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Stockage des résultats
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Mise à jour du meilleur modèle
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Sauvegarde des résultats
print("\nSauvegarde des résultats...")
joblib.dump(results, 'model_results.pkl')

# Sauvegarde du meilleur modèle
print("Sauvegarde du meilleur modèle...")
joblib.dump(best_model, 'best_model.pkl')

# Affichage des résultats
print("\nRésultats des modèles :")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

print(f"\nMeilleur modèle : {best_model.__class__.__name__} avec une précision de {best_score:.3f}") 