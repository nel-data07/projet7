import streamlit as st
import requests
import pandas as pd

# Effacer le cache au démarrage
st.cache_data.clear()
st.cache_resource.clear()

# URL de l'API
API_URL = "https://projet7-1.onrender.com"

# Configuration de la page
st.set_page_config(
    page_title="Simulation de Risque de Crédit",
    page_icon="💳",
    layout="wide"
)

# Description introductive
st.write(
    """
    # Simulation de Risque de Crédit
    Visualisez les prédictions et facteurs influençant la décision pour un client existant.
    """
)

# Menu : Prédictions pour un client existant
st.title("Prédictions pour un Client Existant")

# Charger les IDs clients
try:
    response = requests.get(f"{API_URL}/get_client_ids")
    if response.status_code == 200:
        client_ids = response.json().get("client_ids", [])
    else:
        st.error(f"Erreur lors de la récupération des IDs clients : {response.status_code}")
        client_ids = []
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs clients : {e}")
    client_ids = []

# Afficher une liste déroulante pour sélectionner un client
if client_ids:
    selected_id = st.selectbox("Choisissez un ID client (SK_ID_CURR)", client_ids)
    if st.button("Prédire"):
        try:
            # Appel à l'API pour récupérer les prédictions
            response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_id})
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("probability_of_default", None)
                shap_values = data.get("shap_values", [])
                feature_names = data.get("feature_names", [])

                # Affichage de la probabilité de défaut
                if prediction is not None:
                    optimal_threshold = 0.08  # Seuil pour la décision
                    if prediction > optimal_threshold:
                        st.error(f"Résultat : Crédit REFUSÉ (Probabilité de défaut : {prediction:.2f})")
                    else:
                        st.success(f"Résultat : Crédit ACCEPTÉ (Probabilité de défaut : {prediction:.2f})")

                # Affichage des 15 principales valeurs SHAP
                if shap_values and feature_names:
                    st.subheader("Top 15 des facteurs influençant la décision")
                    # Convertir les données en DataFrame et trier par valeur SHAP absolue
                    shap_df = pd.DataFrame({
                        "feature": feature_names,
                        "shap_value": shap_values
                    })
                    
                    # Trier les 15 principales caractéristiques par valeur absolue de SHAP
                    shap_df = shap_df.reindex(shap_df["shap_value"].abs().sort_values(ascending=False).index).head(15)

                    # Afficher le barplot horizontal
                    st.bar_chart(shap_df.set_index("feature"))
                else:
                    st.warning("Valeurs SHAP indisponibles.")
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.write(response.json())
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")
else:
    st.warning("Aucun ID client disponible.")
