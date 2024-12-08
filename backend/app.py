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
MODEL_PATH = "best_model_lgb_no.pkl"
CLIENTS_DATA_PATH = "/Users/nelly/Desktop/projet 7/clients_data.csv"
FEATURES_PATH = "selected_features.txt"

# Charger les fichiers nécessaires
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Modèle ou fichier des features introuvable.")

model = joblib.load(MODEL_PATH)
logging.info("Modèle chargé avec succès.")

with open(FEATURES_PATH, "r") as f:
    required_features = f.read().strip().split(",")

clients_data = pd.read_csv(CLIENTS_DATA_PATH) if os.path.exists(CLIENTS_DATA_PATH) else pd.DataFrame()

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API en ligne", "status": "success"}), 200

@app.route("/get_client_ids", methods=["GET"])
def get_client_ids():
    if clients_data.empty:
        return jsonify({"client_ids": []}), 200
    return jsonify({"client_ids": clients_data["SK_ID_CURR"].tolist()}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr]
        if client_data.empty:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Vérification si plusieurs lignes existent et utilisation de la première
        if len(client_data) > 1:
            logging.warning(f"Plusieurs lignes trouvées pour le client {sk_id_curr}. Utilisation de la première.")
        client_data = client_data.iloc[0:1]

        # Préparer les données pour la prédiction
        data_for_prediction = client_data[required_features]
        logging.info(f"Données prêtes pour la prédiction :\n{data_for_prediction}")

        # Prédiction avec le modèle
        predictions = model.predict_proba(data_for_prediction)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive (défaut de paiement)
        logging.info(f"Probabilité de défaut de paiement : {probability_of_default}")

        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default": probability_of_default,  # Garder seulement la probabilité de défaut
            "shap_values": shap_values.tolist(),
            "feature_names": required_features
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
