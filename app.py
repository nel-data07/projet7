import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

app = Flask(__name__)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Charger le modèle
MODEL_PATH = "lightgbm_model_final.txt"
model = lgb.Booster(model_file=MODEL_PATH)

# Colonnes attendues par le modèle
expected_columns = model.feature_name()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lecture des données reçues
        input_data = request.get_json()
        logging.info(f"Données reçues : {input_data}")

        # Transformer les données en DataFrame
        logging.info(f"Préparation des données...")
        df = pd.DataFrame(input_data)
 # Ajouter les colonnes manquantes
        for col, default_value in default_columns.items():
            if col not in df.columns:
                df[col] = default_value
        logging.info(f"Données après traitement : {df.head()}")

        # Prediction
        predictions = model.predict(df)
        logging.info(f"Prédictions générées : {predictions}")

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Colonnes attendues par le modèle :", expected_columns)
    # Exécuter l'application
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
