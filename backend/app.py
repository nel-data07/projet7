import os
import logging
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import warnings
import shap

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Initialiser une application Flask uniquement si elle n'existe pas déjà
if "app" not in globals():
    app = Flask(__name__)
    CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Chemins des fichiers nécessaires
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "best_model_lgb_bal.pkl"))
FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "selected_features.txt"))
CLIENTS_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "clients_data.csv"))

# Vérification des chemins
if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(FEATURES_PATH):
    logging.error(f"Fichier des colonnes introuvable : {FEATURES_PATH}")
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

# Charger le modèle
start_time = time.time()
model = joblib.load(MODEL_PATH)
logging.info(f"Modèle chargé en {time.time() - start_time:.2f} secondes.")

# Charger les colonnes utilisées pour l'entraînement
with open(FEATURES_PATH, "r") as f:
    required_columns = f.read().strip().split(",")

# Valeurs par défaut pour les colonnes nécessaires
default_values = {col: 0.0 for col in required_columns}

# Charger les données clients
if os.path.exists(CLIENTS_DATA_PATH):
    clients_data = pd.read_csv(CLIENTS_DATA_PATH, delimiter=",")
    # Forcer les types des colonnes
    clients_data = clients_data.astype({
        "SK_ID_CURR": int,
        "CODE_GENDER": int,
        "FLAG_OWN_CAR": int,
        "CNT_CHILDREN": int,
        "AMT_INCOME_TOTAL": float,
        "AMT_CREDIT": float,
        "AMT_ANNUITY": float,
        "AMT_GOODS_PRICE": float
    })
    logging.info("Chargement du fichier clients_data avec typage forcé :")
    logging.info(clients_data.dtypes)
else:
    clients_data = pd.DataFrame(columns=["SK_ID_CURR"] + required_columns)

@app.route('/', methods=['GET'])
def index():
    """Route principale pour vérifier si l'API est en ligne."""
    return jsonify({
        "message": "API en ligne",
        "status": "success",
        "routes": ["/", "/predict", "/get_client_ids", "/get_next_client_id", "/predict_client"]
    }), 200

@app.route('/get_client_ids', methods=['GET'])
def get_client_ids():
    """Retourne les IDs des clients existants."""
    try:
        if clients_data.empty:
            logging.warning("Les données clients sont vides.")
            return jsonify({"error": "Aucun client trouvé."}), 404
        client_ids = clients_data["SK_ID_CURR"].tolist()
        logging.info(f"Clients trouvés : {client_ids}")
        return jsonify({"client_ids": client_ids}), 200
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des clients : {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_next_client_id', methods=['GET'])
def get_next_client_id():
    """Retourne le prochain ID client disponible."""
    try:
        if clients_data.empty:
            logging.warning("Les données clients sont vides.")
            return jsonify({"error": "Aucun client trouvé."}), 404
        
        # Calculer le prochain ID client
        max_id = clients_data["SK_ID_CURR"].max()
        next_id = int(max_id + 1) if not pd.isnull(max_id) else 100001  # Conversion explicite en int natif
        logging.info(f"Prochain ID client généré : {next_id}")
        return jsonify({"next_id": int(next_id)})  # S'assurer que next_id est de type natif
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du prochain ID client : {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_client', methods=['POST'])
def predict_client():
    """Obtenir les prédictions et valeurs SHAP pour un client existant."""
    try:
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))  # Convertir l'ID client en entier
        logging.info(f"Vérification de SK_ID_CURR : {sk_id_curr}")
        logging.info(f"Colonnes disponibles dans clients_data : {clients_data.columns.tolist()}")

        # Vérification de l'existence de l'ID client
        if sk_id_curr not in clients_data["SK_ID_CURR"].values:
            return jsonify({"error": f"Client avec ID {sk_id_curr} introuvable."}), 404

        # Récupérer les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr]
        logging.info(f"Données récupérées pour SK_ID_CURR={sk_id_curr} :\n{client_data}")

        client_data = client_data.reset_index(drop=True)

        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_columns = [col for col in required_columns if col not in client_data.columns]
        if missing_columns:
            logging.error(f"Colonnes manquantes dans client_data : {missing_columns}")
            return jsonify({"error": f"Colonnes manquantes : {missing_columns}"}), 400

        # Créer un DataFrame sans 'SK_ID_CURR' pour la prédiction
        data_for_prediction = client_data.drop(columns=['SK_ID_CURR'], errors='ignore')

        # Filtrer uniquement les colonnes nécessaires
        data_for_prediction = data_for_prediction[required_columns]
        logging.info(f"Données finales envoyées au modèle :\n{data_for_prediction}")

        # Prédiction et valeurs SHAP
        predictions = model.predict_proba(data_for_prediction)
        logging.info(f"Probabilités prédites par le modèle : {predictions}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des classes multiples pour les SHAP values
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe 1
        else:
            shap_values = shap_values[0]

        return jsonify({
            "prediction": predictions[0][1],  # Probabilité pour la classe positive
            "shap_values": shap_values.tolist(),
            "feature_names": required_columns
        }), 200
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction pour le client : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
