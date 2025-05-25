import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

def analyze_data(df):
    """
    Effectue l'analyse exploratoire des données
    """
    # 1. Distribution de l'âge
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', hue='target', multiple="stack")
    plt.title('Distribution de l\'âge par présence de maladie cardiaque')
    plt.savefig('age_distribution.png')
    plt.close()

    # 2. Différence de maladie cardiaque entre les sexes
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sex', hue='target')
    plt.title('Maladie cardiaque par sexe')
    plt.xlabel('Sexe (0=Femme, 1=Homme)')
    plt.ylabel('Nombre de patients')
    plt.savefig('sex_distribution.png')
    plt.close()

    # 3. Type de douleur thoracique vs maladie cardiaque
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cp', hue='target')
    plt.title('Type de douleur thoracique vs Maladie cardiaque')
    plt.xlabel('Type de douleur thoracique')
    plt.ylabel('Nombre de patients')
    plt.savefig('chest_pain_distribution.png')
    plt.close()

    # 4. Moyennes des variables numériques par présence de maladie
    numeric_cols = ['trestbps', 'chol', 'thalach']
    means = df.groupby('target')[numeric_cols].mean()
    print("\nMoyennes par présence de maladie cardiaque:")
    print(means)

    # 5. Matrice de corrélation
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # 6. Statistiques descriptives
    print("\nStatistiques descriptives:")
    print(df.describe())

if __name__ == "__main__":
    # Chargement et analyse des données
    df = load_data()
    analyze_data(df) 