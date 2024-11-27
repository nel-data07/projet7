import streamlit as st
import requests
import pandas as pd

# URL de l'API déployée
API_URL = "https://projet7-gszq.onrender.com"  # Remplacez par l'URL correcte

# Titre du dashboard
st.title("Dashboard de Prédiction de Non-Remboursement de Crédit")

# Description du projet
st.markdown("""
Ce tableau de bord permet de :
- Simuler un scoring client basé sur les données entrées.
- Envoyer ces données à une API pour obtenir une prédiction.
- Visualiser les résultats sous forme de probabilité de non-remboursement.
""")

# Entrée des données utilisateur
st.sidebar.header("Entrée des paramètres client")

# Collecte des données via le sidebar
CODE_GENDER = st.sidebar.selectbox("Genre (0=Femme, 1=Homme)", [0, 1])
FLAG_OWN_CAR = st.sidebar.selectbox("Possède une voiture (0=Non, 1=Oui)", [0, 1])
CNT_CHILDREN = st.sidebar.slider("Nombre d'enfants", 0, 10, 2)
AMT_INCOME_TOTAL = st.sidebar.number_input("Revenu total annuel (€)", min_value=0, step=1000, value=200000)
AMT_CREDIT = st.sidebar.number_input("Montant du crédit demandé (€)", min_value=0, step=1000, value=500000)
AMT_ANNUITY = st.sidebar.number_input("Montant de l'annuité (€)", min_value=0, step=100, value=25000)
AMT_GOODS_PRICE = st.sidebar.number_input("Prix des biens (€)", min_value=0, step=1000, value=450000)

# Préparer les données pour l'envoi à l'API
data = [
    {
        "CODE_GENDER": CODE_GENDER,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "CNT_CHILDREN": CNT_CHILDREN,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    }
]

# Afficher les données collectées
st.write("Données collectées :")
st.json(data)

# Bouton pour envoyer les données à l'API
if st.button("Envoyer les données à l'API"):
    try:
        # Appel API
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            # Obtenir les résultats
            predictions = response.json().get("predictions", [])
            st.success("Prédiction réussie !")
            st.write("Probabilité de non-remboursement :", predictions)

            # Affichage des résultats
            if predictions[0] >= 0.5:
                st.error("Résultat : Risque élevé de non-remboursement")
            else:
                st.success("Résultat : Risque faible de non-remboursement")
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {str(e)}")

# Footer
st.markdown("Développé avec Streamlit et Flask - Projet 7")

