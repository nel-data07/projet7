import streamlit as st
import requests

import streamlit as st

# Effacer le cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# URL de l'API déployée
API_URL = "https://projet7-gszq.onrender.com/predict"

st.header("Test de l'API")

# Tester une prédiction
st.subheader("Entrer les données pour la prédiction")

CODE_GENDER = st.selectbox("Genre (0=Femme, 1=Homme)", [0, 1])
FLAG_OWN_CAR = st.selectbox("Possède une voiture (0=Non, 1=Oui)", [0, 1])
CNT_CHILDREN = st.slider("Nombre d'enfants", 0, 10, 0)
AMT_INCOME_TOTAL = st.number_input("Revenu total annuel (€)", min_value=0, value=0)
AMT_CREDIT = st.number_input("Montant du crédit demandé (€)", min_value=0, value=0)
AMT_ANNUITY = st.number_input("Montant de l'annuité (€)", min_value=0, value=0)
AMT_GOODS_PRICE = st.number_input("Prix des biens (€)", min_value=0, value=0)

# Préparer les données pour l'API
data = [{
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "CNT_CHILDREN": CNT_CHILDREN,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE
}]

if st.button("Envoyer les données à l'API"):
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            st.success(f"Probabilité de non-remboursement : {predictions[0]:.2f}")
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")
