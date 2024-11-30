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

app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Chemin absolu vers le fichier modèle
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "best_model_lgb_bal.pkl"))
FEATURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "selected_features.txt"))
CLIENTS_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "clients_data.csv"))  # Exemple de fichier client

# Vérification des chemins
if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(FEATURES_PATH):
    logging.error(f"Fichier des colonnes introuvable : {FEATURES_PATH}")
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

# Charger le modèle au format .pkl
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
    clients_data = pd.read_csv(CLIENTS_DATA_PATH)
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
    """Obtenir la liste des IDs clients existants."""
    ids = clients_data["SK_ID_CURR"].tolist()
    return jsonify({"client_ids": ids}), 200

@app.route('/get_next_client_id', methods=['GET'])
def get_next_client_id():
    """Obtenir le prochain ID client auto-incrémenté."""
    if not clients_data.empty:
        next_id = int(clients_data["SK_ID_CURR"].max()) + 1
    else:
        next_id = 100001  # Exemple de point de départ pour les IDs
    return jsonify({"next_id": next_id}), 200

@app.route('/predict_client', methods=['POST'])
def predict_client():
    """Obtenir les prédictions et valeurs SHAP pour un client existant."""
    try:
        data = request.get_json()
        sk_id_curr = data.get("SK_ID_CURR")

        if sk_id_curr not in clients_data["SK_ID_CURR"].values:
            return jsonify({"error": f"Client avec ID {sk_id_curr} introuvable."}), 404

        # Récupérer les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr].iloc[:, 1:]  # Supprime SK_ID_CURR
        client_data = client_data.reset_index(drop=True)  # Réinitialiser l'index

        # Prédiction et valeurs SHAP
        prediction = model.predict_proba(client_data)[:, 1][0]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(client_data)[1][0]  # SHAP pour la classe positive

        return jsonify({
            "prediction": prediction,
            "shap_values": shap_values.tolist(),
            "feature_names": required_columns
        }), 200
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction pour le client : {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Route pour effectuer des prédictions pour de nouveaux clients."""
    try:
        total_start_time = time.time()

        # Lecture des données reçues
        input_data = request.get_json()
        logging.info(f"Données reçues : {input_data}")

        if not input_data or not isinstance(input_data, list):
            return jsonify({'error': "Les données doivent être une liste d'objets JSON"}), 400

        # Transformer les données en DataFrame
        df = pd.DataFrame(input_data)

        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Colonne manquante ajoutée : {col} avec la valeur par défaut {default_values[col]}")
                df[col] = default_values[col]
            else:
                df[col] = df[col].fillna(default_values[col])

        # S'assurer que les types des colonnes sont corrects
        df = df.astype({col: float for col in required_columns})

        # Filtrer uniquement les colonnes nécessaires pour le modèle
        df = df[required_columns]
        df = df.reset_index(drop=True)  # Réinitialiser l'index

        # Prédiction
        predict_start_time = time.time()
        predictions = model.predict_proba(df)[:, 1]  # Prédictions avec probabilités pour la classe positive
        logging.info(f"Prédiction effectuée en {time.time() - predict_start_time:.2f} secondes.")

        # Sauvegarder les données clients
        if "SK_ID_CURR" in df.columns:
            global clients_data
            clients_data = pd.concat([clients_data, df], ignore_index=True).drop_duplicates("SK_ID_CURR")
            clients_data = clients_data.reset_index(drop=True)  # Réinitialiser l'index
            clients_data.to_csv(CLIENTS_DATA_PATH, index=False)

        # Retourner les résultats
        total_time = time.time() - total_start_time
        logging.info(f"Requête complète traitée en {total_time:.2f} secondes.")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
