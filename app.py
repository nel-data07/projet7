import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc
import time  # Pour mesurer le temps d'exécution

app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Charger le modèle
MODEL_PATH = os.path.join("models", "lightgbm_model.txt")
FEATURES_PATH = os.path.join("models", "selected_features.txt")

# Mesurer le temps de chargement du modèle
start_time = time.time()

if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(FEATURES_PATH):
    logging.error(f"Fichier des colonnes introuvable : {FEATURES_PATH}")
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

model = lgb.Booster(model_file=MODEL_PATH)
logging.info(f"Modèle chargé en {time.time() - start_time:.2f} secondes.")

# Charger les colonnes utilisées pour l'entraînement
with open(FEATURES_PATH, "r") as f:
    required_columns = f.read().split(",")

# Valeurs par défaut pour les colonnes nécessaires
default_values = {col: 0 for col in required_columns}

@app.route('/', methods=['GET'])
def index():
    return "API en ligne. Utilisez /predict pour les prédictions.", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mesurer le temps total de la requête
        total_start_time = time.time()

        # Lecture des données reçues
        input_data = request.get_json()
        logging.info(f"Données reçues : {input_data}")

        if not input_data:
            return jsonify({'error': "Aucune donnée reçue"}), 400

        # Transformer les données en DataFrame
        df = pd.DataFrame(input_data)

        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Utiliser une valeur par défaut (ex : 0)

        # Filtrer uniquement les colonnes nécessaires pour le modèle
        df = df[required_columns]

        # Mesurer le temps de prédiction
        predict_start_time = time.time()
        predictions = model.predict(df)
        logging.info(f"Prédiction effectuée en {time.time() - predict_start_time:.2f} secondes.")

        # Retour des prédictions
        total_time = time.time() - total_start_time
        logging.info(f"Requête complète traitée en {total_time:.2f} secondes.")
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
