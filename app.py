import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Charger le modèle
MODEL_PATH = os.path.join("models", "lightgbm_model.txt")
FEATURES_PATH = os.path.join("models", "selected_features.txt")

if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(FEATURES_PATH):
    logging.error(f"Fichier des colonnes introuvable : {FEATURES_PATH}")
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

model = lgb.Booster(model_file=MODEL_PATH)

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

        # Prédiction
        predictions = model.predict(df)

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
