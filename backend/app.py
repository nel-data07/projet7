import os
import logging
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import shap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Désactive l'interface graphique

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Initialiser une application Flask
app = Flask(__name__)
CORS(app)

# Chemins des fichiers nécessaires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_lgb_no.pkl")
CLIENTS_DATA_PATH = os.path.join(BASE_DIR, "clients_data.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.txt")

# Activer les logs
logging.basicConfig(level=logging.INFO)
print("Répertoire courant :", os.getcwd())
print("Chemin attendu pour clients_data.csv :", CLIENTS_DATA_PATH)
print("Fichier présent :", os.path.exists(CLIENTS_DATA_PATH))

# Vérifier si les fichiers nécessaires existent
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Modèle ou fichier des features introuvable.")

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Charger les features nécessaires
with open(FEATURES_PATH, "r") as f:
    required_features = f.read().strip().split(",")

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API en ligne", "status": "success"}), 200

@app.route("/get_client_ids", methods=["GET"])
def get_client_ids():
    client_ids = []
    for chunk in load_data_in_chunks(CLIENTS_DATA_PATH):
        client_ids.extend(chunk["SK_ID_CURR"].tolist())
    return jsonify({"client_ids": client_ids}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))

        # Parcourir les chunks pour trouver les données du client
        client_data = None
        for chunk in load_data_in_chunks(CLIENTS_DATA_PATH):
            filtered_data = chunk[chunk["SK_ID_CURR"] == sk_id_curr]
            if not filtered_data.empty:
                client_data = filtered_data.iloc[0]  # Récupère la première occurrence
                break

        if client_data is None:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Préparer les données pour la prédiction
        data_for_prediction = pd.DataFrame([client_data])[required_features]
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

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default": probability_of_default,
            "shap_values": shap_values.tolist(),
            "feature_names": required_features
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
