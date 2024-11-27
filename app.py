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
        df = pd.DataFrame(input_data)

        # Identifier les colonnes manquantes
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Colonnes manquantes détectées : {missing_columns}")
            # Ajouter les colonnes manquantes avec des zéros
            for col in missing_columns:
                df[col] = 0

        # Vérifier et aligner l'ordre des colonnes avec les colonnes attendues
        df = df.reindex(columns=expected_columns)

        # Lancer la prédiction
        predictions = model.predict(df)

        # Retourner les résultats
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Exécuter l'application
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
