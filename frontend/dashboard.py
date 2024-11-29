import streamlit as st
import requests

# Effacer le cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# Récupérer l'URL de l'API depuis une variable d'environnement
API_URL = "https://projet7-1.onrender.com"

# Titre principal du dashboard
st.title("Simulation de Risque de Crédit")

# Description introductive
st.write(
    """
    Bienvenue dans l'outil de simulation de risque de crédit. 
    Veuillez entrer les informations ci-dessous pour estimer la probabilité de non-remboursement d'un crédit. 
    - **Montant de l'annuité** : Montant annuel remboursé (capital + intérêts).  
    - **Prix des biens** : Montant total des biens financés avec le crédit.
    """
)

# Formulaire pour saisir les données
st.subheader("Entrez les informations pour la simulation")

CODE_GENDER = st.selectbox("Genre (0 = Femme, 1 = Homme)", [0, 1])
FLAG_OWN_CAR = st.selectbox("Possède une voiture (0 = Non, 1 = Oui)", [0, 1])
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
            prob = predictions[0]
            st.success(f"Probabilité de non-remboursement : {prob:.2f}")
            
            # Indiquer si le crédit est accepté ou refusé
            if prob > 0.5:
                st.error("Résultat : Crédit REFUSÉ (Risque élevé)")
            else:
                st.success("Résultat : Crédit ACCEPTÉ (Risque faible)")
        else:
            st.error(f"Erreur API : {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")
