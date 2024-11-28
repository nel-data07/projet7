import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Charger le modèle
MODEL_PATH = "lightgbm_model_final.txt"

if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

model = lgb.Booster(model_file=MODEL_PATH)

# Colonnes minimales nécessaires pour la prédiction
required_columns = ["CODE_GENDER", "FLAG_OWN_CAR", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]

# Valeurs par défaut pour les colonnes nécessaires
default_values = {
    "CODE_GENDER": 1,
    "FLAG_OWN_CAR": 0,
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 0,
    "AMT_CREDIT": 0,
    "AMT_ANNUITY": 0,
    "AMT_GOODS_PRICE": 0
}

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
        for col, default_value in default_values.items():
            if col not in df.columns:
                df[col] = default_value

        # Filtrer uniquement les colonnes nécessaires pour le modèle
        df = df[required_columns]

        # Prédiction
        predictions = model.predict(df)

        # Libérer la mémoire après chaque requête
        del df
        gc.collect()

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
