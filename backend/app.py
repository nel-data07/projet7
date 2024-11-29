import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import time  # Pour mesurer le temps d'exécution

app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Charger le modèle
MODEL_PATH = "./backend/lightgbm_model.txt"
FEATURES_PATH = "./backend/selected_features.txt"

# Vérification des chemins
if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(FEATURES_PATH):
    logging.error(f"Fichier des colonnes introuvable : {FEATURES_PATH}")
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

# Charger le modèle
start_time = time.time()
model = lgb.Booster(model_file=MODEL_PATH)
logging.info(f"Modèle chargé en {time.time() - start_time:.2f} secondes.")

# Charger les colonnes utilisées pour l'entraînement
with open(FEATURES_PATH, "r") as f:
    required_columns = f.read().strip().split(",")

# Valeurs par défaut pour les colonnes nécessaires
default_values = {col: 0.0 for col in required_columns}

@app.route('/', methods=['GET'])
def index():
    """Route principale pour vérifier si l'API est en ligne."""
    return jsonify({
        "message": "API en ligne",
        "status": "success",
        "routes": ["/", "/predict", "/test-data", "/test-model"]
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Route pour effectuer des prédictions."""
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

        # Filtrer uniquement les colonnes nécessaires pour le modèle
        df = df[required_columns]

        # Prédiction
        predict_start_time = time.time()
        predictions = model.predict(df)
        logging.info(f"Prédiction effectuée en {time.time() - predict_start_time:.2f} secondes.")

        # Retourner les résultats
        total_time = time.time() - total_start_time
        logging.info(f"Requête complète traitée en {total_time:.2f} secondes.")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-data', methods=['GET'])
def test_data():
    """Route pour tester la validation des données."""
    example_data = [{
        "CODE_GENDER": 1,
        "FLAG_OWN_CAR": 0,
        "CNT_CHILDREN": 2,
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0
    }]
    return jsonify({"example_data": example_data}), 200

@app.route('/test-model', methods=['GET'])
def test_model():
    """Route pour tester si le modèle est chargé correctement."""
    try:
        # Effectuer une prédiction factice avec des données aléatoires
        dummy_data = pd.DataFrame([default_values])
        prediction = model.predict(dummy_data)
        return jsonify({
            "status": "success",
            "dummy_prediction": prediction.tolist()
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
