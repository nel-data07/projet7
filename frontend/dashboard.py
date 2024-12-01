import streamlit as st
import requests
import pandas as pd

# Effacer le cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# Configuration des pages du dashboard
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

# Style CSS pour améliorer l'apparence
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .description {
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Description introductive
st.title("Bienvenue dans l'outil de simulation de risque de crédit")
st.write(
    """
    <p class="description">
    Cet outil permet d'estimer la probabilité de non-remboursement d'un crédit en fonction des informations du client.
    </p>
    <h3>Fonctionnalités :</h3>
    <ul>
        <li>Clients existants : Sélectionnez un ID client pour visualiser les prédictions associées.</li>
        <li>Nouveaux clients : Remplissez les informations nécessaires pour simuler une nouvelle demande de crédit.</li>
    </ul>
    <h3>Définitions :</h3>
    <ul>
        <li><b>Montant de l'annuité</b> : Montant annuel remboursé (capital + intérêts).</li>
        <li><b>Prix des biens</b> : Montant total des biens financés avec le crédit.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Menu de navigation
menu = st.sidebar.selectbox(
    "Menu", ["Prédictions Client Existant", "Créer Nouveau Client"]
)

if menu == "Prédictions Client Existant":
    st.subheader("Simulation de Risque de Crédit - Client Existant")
    st.write("### Sélectionnez un client existant pour prédire le risque.")

    try:
        # Charger les IDs clients depuis l'API
        response = requests.get("https://projet7-1.onrender.com/get_client_ids")
        if response.status_code == 200:
            client_ids = response.json().get("client_ids", [])
        else:
            st.error(f"Erreur lors de la récupération des IDs clients : {response.status_code}")
            client_ids = []
    except Exception as e:
        st.error(f"Erreur : {e}")
        client_ids = []

    # Interface pour sélectionner un ID client
    if client_ids:
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
        if st.button("Prédire"):
            try:
                response = requests.post(
                    "https://projet7-1.onrender.com/predict_client",
                    json={"SK_ID_CURR": selected_id},
                )
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction")
                    shap_values = data.get("shap_values")
                    feature_names = data.get("feature_names")

                    if prediction > 0.5:
                        st.error(f"Crédit REFUSÉ (Risque élevé - {prediction:.2f})")
                    else:
                        st.success(f"Crédit ACCEPTÉ (Risque faible - {prediction:.2f})")

                    # Affichage des valeurs SHAP
                    if shap_values and feature_names:
                        st.subheader("Facteurs influençant la décision du modèle")
                        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values})
                        st.bar_chart(shap_df.set_index("Feature"))
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Erreur : {e}")
    else:
        st.warning("Aucun ID client disponible.")

elif menu == "Créer Nouveau Client":
    st.subheader("Simulation de Risque de Crédit - Nouveau Client")
    st.write("### Remplissez les informations suivantes pour prédire le risque.")

    # Formulaire d'entrée des données
    CODE_GENDER = st.selectbox("Genre (0 = Femme, 1 = Homme)", [0, 1])
    FLAG_OWN_CAR = st.selectbox("Possède une voiture (0 = Non, 1 = Oui)", [0, 1])
    CNT_CHILDREN = st.slider("Nombre d'enfants", 0, 10, 0)
    AMT_INCOME_TOTAL = st.number_input("Revenu total annuel (€)", min_value=0, value=0)
    AMT_CREDIT = st.number_input("Montant du crédit demandé (€)", min_value=0, value=0)
    AMT_ANNUITY = st.number_input("Montant de l'annuité (€)", min_value=0, value=0)
    AMT_GOODS_PRICE = st.number_input("Prix des biens (€)", min_value=0, value=0)

    # Préparer les données pour l'API
    data = {
        "CODE_GENDER": CODE_GENDER,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "CNT_CHILDREN": CNT_CHILDREN,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    }

    if st.button("Valider"):
        try:
            response = requests.post(
                "https://projet7-1.onrender.com/predict_client", json=data
            )
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction")
                shap_values = data.get("shap_values")
                feature_names = data.get("feature_names")

                if prediction > 0.5:
                    st.error(f"Crédit REFUSÉ (Risque élevé - {prediction:.2f})")
                else:
                    st.success(f"Crédit ACCEPTÉ (Risque faible - {prediction:.2f})")

                if shap_values and feature_names:
                    st.subheader("Facteurs influençant la décision du modèle")
                    shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values})
                    st.bar_chart(shap_df.set_index("Feature"))
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f"Erreur : {e}")
