import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    """
    Charge le dataset Heart Disease UCI
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)
    
    # Nettoyage des données
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Conversion des colonnes en types numériques
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Binarisation de la variable cible
    df['target'] = (df['target'] > 0).astype(int)
    
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Prétraite les données pour l'entraînement
    """
    # Séparation des features et de la cible
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_feature_names():
    """
    Retourne les noms des features avec leurs descriptions
    """
    return {
        'age': 'Âge',
        'sex': 'Sexe (1=Homme, 0=Femme)',
        'cp': 'Type de douleur thoracique',
        'trestbps': 'Pression artérielle au repos (mm Hg)',
        'chol': 'Cholestérol sérique (mg/dl)',
        'fbs': 'Glycémie à jeun > 120 mg/dl',
        'restecg': 'Électrocardiogramme au repos',
        'thalach': 'Fréquence cardiaque maximale',
        'exang': 'Angine induite par exercice',
        'oldpeak': 'Dépression ST à l\'effort',
        'slope': 'Pente du segment ST',
        'ca': 'Nombre de vaisseaux principaux',
        'thal': 'Thalassémie'
    }

if __name__ == "__main__":
    # Test du chargement des données
    df = load_data()
    print("Shape du dataset:", df.shape)
    print("\nAperçu des données:")
    print(df.head())
    print("\nInformations sur les données:")
    print(df.info()) 