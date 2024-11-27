import streamlit as st
import requests

# URL de l'API déployée
API_URL = "https://projet7-gszq.onrender.com/predict"  # Remplacez par votre URL d'API

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

# Colonnes manquantes avec valeurs par défaut
default_columns = {
    'FLAG_OWN_REALTY': 0, 'REGION_POPULATION_RELATIVE': 0.0, 'DAYS_BIRTH': 0,
    'DAYS_EMPLOYED': 0, 'DAYS_REGISTRATION': 0, 'DAYS_ID_PUBLISH': 0, 'OWN_CAR_AGE': 0,
    'FLAG_MOBIL': 1, 'FLAG_EMP_PHONE': 0, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1,
    'FLAG_PHONE': 0, 'FLAG_EMAIL': 0, 'CNT_FAM_MEMBERS': 1, 'REGION_RATING_CLIENT': 1,
    'REGION_RATING_CLIENT_W_CITY': 1, 'HOUR_APPR_PROCESS_START': 10, 'REG_REGION_NOT_LIVE_REGION': 0,
    'REG_REGION_NOT_WORK_REGION': 0, 'LIVE_REGION_NOT_WORK_REGION': 0,
    'REG_CITY_NOT_LIVE_CITY': 0, 'REG_CITY_NOT_WORK_CITY': 0, 'LIVE_CITY_NOT_WORK_CITY': 0,
    'EXT_SOURCE_1': 0.0, 'EXT_SOURCE_2': 0.0, 'EXT_SOURCE_3': 0.0,
    'APARTMENTS_AVG': 0.0, 'BASEMENTAREA_AVG': 0.0, 'YEARS_BEGINEXPLUATATION_AVG': 0.0,
    'YEARS_BUILD_AVG': 0.0, 'COMMONAREA_AVG': 0.0, 'ELEVATORS_AVG': 0.0,
    'ENTRANCES_AVG': 0.0, 'FLOORSMAX_AVG': 0.0, 'FLOORSMIN_AVG': 0.0,
    'LANDAREA_AVG': 0.0, 'LIVINGAPARTMENTS_AVG': 0.0, 'LIVINGAREA_AVG': 0.0,
    'NONLIVINGAPARTMENTS_AVG': 0.0, 'NONLIVINGAREA_AVG': 0.0,
    'DAYS_EMPLOYED_PERC': 0.0, 'INCOME_CREDIT_PERC': 0.0, 'INCOME_PER_PERSON': 0.0,
    'ANNUITY_INCOME_PERC': 0.0, 'PAYMENT_RATE': 0.0
}

# Préparer les données utilisateur
data = {
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "CNT_CHILDREN": CNT_CHILDREN,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    **default_columns  # Ajouter toutes les colonnes manquantes
}

# Convertir en liste pour compatibilité avec l'API
data_list = [data]

# Afficher les données collectées
st.write("Données collectées :")
st.json(data_list)

# Bouton pour envoyer les données à l'API
if st.button("Envoyer les données à l'API"):
    try:
        # Appel API
        response = requests.post(API_URL, json=data_list)
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
