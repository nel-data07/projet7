import os
import logging
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import shap

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Initialiser une application Flask
app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Chemins des fichiers nécessaires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_lgb_no.pkl")
CLIENTS_DATA_PATH = os.path.join(BASE_DIR, "clients_data.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.txt")

# Vérifications et chargements initiaux
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Modèle ou fichier des features introuvable.")

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Charger les features nécessaires
with open(FEATURES_PATH, "r") as f:
    required_features = f.read().strip().split(",")

# Charger les données clients
if os.path.exists(CLIENTS_DATA_PATH):
    clients_data = pd.read_csv(CLIENTS_DATA_PATH)
    logging.info(f"Fichier clients_data.csv chargé avec succès. Nombre de clients : {len(clients_data)}")
else:
    clients_data = pd.DataFrame()
    logging.warning("Le fichier clients_data.csv est introuvable ou vide.")

@app.route("/", methods=["GET"])
def index():
    """Endpoint racine pour vérifier l'état de l'API"""
    return jsonify({"message": "API en ligne", "status": "success"}), 200

@app.route("/get_client_ids", methods=["GET"])
def get_client_ids():
    """Récupérer la liste des IDs clients disponibles"""
    if clients_data.empty:
        return jsonify({"client_ids": []}), 200
    return jsonify({"client_ids": clients_data["SK_ID_CURR"].tolist()}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Faire une prédiction pour un client donné"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))

        # Trouver les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr]
        if client_data.empty:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Préparer les données pour la prédiction
        data_for_prediction = client_data[required_features]
        logging.info(f"Données prêtes pour la prédiction :\n{data_for_prediction}")

        # Prédiction avec le modèle
        predictions = model.predict_proba(data_for_prediction)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive
        logging.info(f"Probabilité de défaut de paiement : {probability_of_default}")

        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Informations descriptives du client
        client_info = client_data.iloc[0].to_dict()

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default": probability_of_default,
            "shap_values": shap_values.tolist(),
            "feature_names": required_features,
            "client_info": client_info
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_global_importance", methods=["GET"])
def get_global_importance():
    """Calculer les importances globales des caractéristiques"""
    try:
        if clients_data.empty:
            return jsonify({"error": "Les données clients sont vides ou indisponibles."}), 404

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(clients_data[required_features])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        global_importances = pd.DataFrame({
            "Feature": required_features,
            "Global Importance": abs(shap_values).mean(axis=0)
        }).sort_values(by="Global Importance", ascending=False)

        return jsonify({
            "status": "success",
            "global_importances": global_importances.to_dict(orient="records")
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors du calcul des importances globales : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_with_custom_values", methods=["POST"])
def predict_with_custom_values():
    """Faire une prédiction avec des valeurs modifiées par l'utilisateur"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))

        # Trouver les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr].copy()
        if client_data.empty:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Mise à jour des valeurs si elles sont fournies dans la requête
        for key, value in data.items():
            if key in client_data.columns and value is not None:
                client_data[key] = value

        # Préparer les données pour la prédiction
        data_for_prediction = client_data[required_features]
        logging.info(f"Données prêtes pour la prédiction avec valeurs personnalisées :\n{data_for_prediction}")

        # Prédiction avec le modèle
        predictions = model.predict_proba(data_for_prediction)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive
        logging.info(f"Probabilité de défaut de paiement avec valeurs personnalisées : {probability_of_default}")

        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Informations descriptives du client
        client_info = client_data.iloc[0].to_dict()

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default": probability_of_default,
            "shap_values": shap_values.tolist(),
            "feature_names": required_features,
            "client_info": client_info
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction avec valeurs personnalisées : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_next_client_id", methods=["GET"])
def get_next_client_id():
    """Renvoie le prochain ID client disponible."""
    if not clients_data.empty:
        max_id = clients_data["SK_ID_CURR"].max()
    else:
        max_id = 100000  # ID initial par défaut
    next_id = max_id + 1
    return jsonify({"next_id": next_id}), 200

@app.route("/predict_new_client", methods=["POST"])
def predict_new_client():
    """Faire une prédiction pour un nouveau client avec des valeurs par défaut"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()

        # Créer une ligne avec des valeurs par défaut
        default_client = {col: 0 for col in required_features}
        
        # Remplacer les colonnes avec les données fournies
        for key, value in data.items():
            if key in default_client:
                default_client[key] = value

        # Convertir en DataFrame
        new_client_df = pd.DataFrame([default_client])

        # Prédiction avec le modèle
        predictions = model.predict_proba(new_client_df)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive

        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(new_client_df)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Retourner la réponse
        return jsonify({
            "probability_of_default": probability_of_default,
            "shap_values": shap_values.tolist(),
            "feature_names": required_features
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction pour un nouveau client : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
