import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import plotly.figure_factory as ff

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Maladie Cardiaque - IFOAD",
    page_icon="❤️",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 18px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    .stSelectbox, .stSlider {
        background-color: #262730;
        color: white;
    }
    h1, h2, h3 {
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre et description
st.title("Prédiction de Maladie Cardiaque")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #262730; border-radius: 10px; margin: 20px 0;'>
        <h2>Projet IFOAD</h2>
        <p>Développé sous la supervision du Dr Arthur Sawadogo</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une section", ["Prédiction", "Analyse des Données", "Performance des Modèles"])

if page == "Prédiction":
    st.header("🎯 Prédiction de Maladie Cardiaque")
    
    # Création de deux colonnes pour les inputs
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Âge", 20, 100, 50)
        sex = st.selectbox("Sexe", ["Homme", "Femme"])
        cp = st.selectbox("Type de douleur thoracique", 
                         ["Typique", "Atypique", "Non-angineuse", "Asymptomatique"])
        trestbps = st.slider("Pression artérielle au repos (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholestérol sérique (mg/dl)", 100, 600, 200)
        
    with col2:
        fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", ["Non", "Oui"])
        restecg = st.selectbox("Électrocardiogramme au repos", 
                              ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire"])
        thalach = st.slider("Fréquence cardiaque maximale", 60, 202, 150)
        exang = st.selectbox("Angine induite par exercice", ["Non", "Oui"])
        oldpeak = st.slider("Dépression ST à l'effort", 0.0, 6.2, 0.0)
        slope = st.selectbox("Pente du segment ST", ["Montante", "Plateau", "Descendante"])
        ca = st.slider("Nombre de vaisseaux colorés (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassémie", ["Normal", "Fixe", "Réversible"])
    
    # Conversion des inputs
    sex = 1 if sex == "Homme" else 0
    cp = ["Typique", "Atypique", "Non-angineuse", "Asymptomatique"].index(cp)
    fbs = 1 if fbs == "Oui" else 0
    restecg = ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire"].index(restecg)
    exang = 1 if exang == "Oui" else 0
    slope = ["Montante", "Plateau", "Descendante"].index(slope)
    thal = ["Normal", "Fixe", "Réversible"].index(thal)
    
    if st.button("Lancer la Prédiction", key="predict"):
        # Création du DataFrame pour la prédiction (13 variables dans l'ordre)
        data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        try:
            # Chargement du modèle
            model = joblib.load('best_model.pkl')
            # Prédiction
            prediction = model.predict(data)
            probability = model.predict_proba(data)[0][1]
            
            # Affichage du résultat avec style
            st.markdown("""
                <div style='text-align: center; padding: 20px; background-color: #262730; border-radius: 10px; margin: 20px 0;'>
                    <h3>Résultat de la Prédiction</h3>
                </div>
                """, unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.error("Risque de maladie cardiaque détecté")
            else:
                st.success("Aucun risque de maladie cardiaque détecté")
                
            # Affichage de la probabilité avec une jauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilité de maladie cardiaque (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
            
        except FileNotFoundError:
            st.warning("Le modèle n'a pas encore été entraîné. Veuillez d'abord exécuter model_training.py")

elif page == "Analyse des Données":
    st.header("Analyse des Données")
    try:
        # Chargement des données
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                        names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
        
        # Nettoyage des données
        df = df.replace('?', np.nan)
        df = df.dropna()
        df = df.astype(float)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Visualisations
        st.subheader("Distribution de l'âge par sexe")
        fig = px.histogram(df, x="age", color="sex", 
                          title="Distribution de l'âge par sexe",
                          labels={"age": "Âge", "sex": "Sexe"},
                          color_discrete_map={0: "pink", 1: "blue"})
        st.plotly_chart(fig)
        
        st.subheader("Corrélation entre les variables")
        corr = df.corr()
        fig = px.imshow(corr, 
                       title="Matrice de corrélation",
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")

elif page == "Performance des Modèles":
    st.header("📈 Performance des Modèles")
    try:
        # Chargement des résultats si disponibles
        results = joblib.load('model_results.pkl')
        
        # Affichage des métriques
        st.subheader("Comparaison des modèles")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df)
        
        # Graphique des performances
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparaison des Performances des Modèles',
            xaxis_title='Modèles',
            yaxis_title='Score',
            barmode='group',
            template='plotly_dark'
        )
        st.plotly_chart(fig)
        
    except FileNotFoundError:
        st.warning("Les résultats des modèles ne sont pas encore disponibles. Veuillez d'abord exécuter model_training.py") 