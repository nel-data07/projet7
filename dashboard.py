import streamlit as st
import requests

# URL de l'API déployée
API_URL = "https://projet7-gszq.onrender.com/predict"

# Vérification de l'API
st.header("Vérification de l'API")
try:
    response = requests.get(API_URL.replace('/predict', '/'))
    if response.status_code == 200:
        st.success("L'API est accessible.")
    else:
        st.warning("L'API est en ligne, mais l'URL semble incorrecte.")
except Exception as e:
    st.error(f"Erreur de connexion : {e}")

# Test de prédiction
st.header("Tester une prédiction")
if st.button("Tester"):
    test_data = [{
        "CODE_GENDER": 1,
        "FLAG_OWN_CAR": 0,
        "CNT_CHILDREN": 2,
        "AMT_INCOME_TOTAL": 200000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000
    }]
    try:
        response = requests.post(API_URL, json=test_data)
        if response.status_code == 200:
            st.success("Prédiction réussie.")
            st.json(response.json())
        else:
            st.error(f"Erreur API : {response.status_code}")
    except Exception as e:
        st.error(f"Erreur : {e}")

# Formulaire pour les utilisateurs
st.sidebar.header("Entrée des paramètres client")
CODE_GENDER = st.sidebar.selectbox("Genre (0=Femme, 1=Homme)", [0, 1])
FLAG_OWN_CAR = st.sidebar.selectbox("Possède une voiture (0=Non, 1=Oui)", [0, 1])
CNT_CHILDREN = st.sidebar.slider("Nombre d'enfants", 0, 10, 2)
AMT_INCOME_TOTAL = st.sidebar.number_input("Revenu total annuel (€)", min_value=0, step=1000, value=200000)
AMT_CREDIT = st.sidebar.number_input("Montant du crédit demandé (€)", min_value=0, step=1000, value=500000)
AMT_ANNUITY = st.sidebar.number_input("Montant de l'annuité (€)", min_value=0, step=100, value=25000)
AMT_GOODS_PRICE = st.sidebar.number_input("Prix des biens (€)", min_value=0, step=1000, value=450000)

# Préparer les données utilisateur
data = {
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "CNT_CHILDREN": CNT_CHILDREN,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE
}

# Envoyer les données pour la prédiction
if st.button("Envoyer les données"):
    try:
        response = requests.post(API_URL, json=[data])
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error(f"Erreur API : {response.status_code}")
    except Exception as e:
        st.error(f"Erreur : {e}")
