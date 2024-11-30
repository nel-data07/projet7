import streamlit as st
import requests
import pandas as pd

# Effacer le cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# Récupérer l'URL de l'API
API_URL = "https://projet7-1.onrender.com"

# Configuration des pages du dashboard
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

# Menu de navigation
menu = st.sidebar.selectbox("Menu", ["Prédictions Client Existant", "Créer Nouveau Client"])

# Page 1 : Prédictions pour un client existant
if menu == "Prédictions Client Existant":
    st.title("Simulation de Risque de Crédit - Client Existant")
    
    # Charger les IDs clients
    try:
        # Exemple d'appel pour récupérer les IDs clients existants (remplacez par votre endpoint API si disponible)
        response = requests.get(f"{API_URL}/get_client_ids")
        if response.status_code == 200:
            client_ids = response.json().get("client_ids", [])
        else:
            st.error(f"Erreur lors de la récupération des IDs clients : {response.status_code}")
            client_ids = []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des IDs clients : {e}")
        client_ids = []

    # Afficher une liste déroulante pour choisir un client
    if client_ids:
        selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
        if st.button("Prédire"):
            # Envoyer une requête à l'API pour obtenir les prédictions et les valeurs SHAP
            try:
                response = requests.post(f"{API_URL}/predict_client", json={"SK_ID_CURR": selected_id})
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction", None)
                    shap_values = data.get("shap_values", None)
                    feature_names = data.get("feature_names", None)

                    # Afficher la prédiction
                    if prediction is not None:
                        if prediction > 0.5:
                            st.error(f"Résultat : Crédit REFUSÉ (Risque élevé - {prediction:.2f})")
                        else:
                            st.success(f"Résultat : Crédit ACCEPTÉ (Risque faible - {prediction:.2f})")

                    # Afficher les valeurs SHAP si disponibles
                    if shap_values and feature_names:
                        st.subheader("Facteurs influençant la décision du modèle")
                        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values})
                        st.bar_chart(shap_df.set_index("Feature"))
                    else:
                        st.warning("Valeurs SHAP indisponibles.")
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Erreur lors de l'appel API : {e}")
    else:
        st.warning("Aucun ID client disponible.")

# Page 2 : Créer un nouveau client et obtenir une prédiction
elif menu == "Créer Nouveau Client":
    st.title("Simulation de Risque de Crédit - Nouveau Client")

    # ID auto-incrémenté
    try:
        response = requests.get(f"{API_URL}/get_next_client_id")
        if response.status_code == 200:
            next_id = response.json().get("next_id", None)
        else:
            st.error(f"Erreur lors de la récupération du prochain ID client : {response.status_code}")
            next_id = "Inconnu"
    except Exception as e:
        st.error(f"Erreur lors de la récupération du prochain ID client : {e}")
        next_id = "Inconnu"

    st.write(f"ID client auto-généré : **{next_id}**")

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
        "SK_ID_CURR": next_id,
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
            response = requests.post(f"{API_URL}/predict", json=data)
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                prob = predictions[0]

                # Affichage de la probabilité
                st.success(f"Probabilité de non-remboursement : {prob:.2f}")

                # Indique si le crédit est accepté ou refusé
                if prob > 0.5:
                    st.error("Résultat : Crédit REFUSÉ (Risque élevé)")
                else:
                    st.success("Résultat : Crédit ACCEPTÉ (Risque faible)")
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")
