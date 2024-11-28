import os
import logging
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines pour simplifier

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Chemin du modèle
MODEL_PATH = "lightgbm_model_final.txt"

# Vérification si le fichier modèle existe
if not os.path.exists(MODEL_PATH):
    logging.error(f"Modèle introuvable : {MODEL_PATH}")
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

# Charger le modèle une seule fois avec optimisation
try:
    model = lgb.Booster(model_file=MODEL_PATH, params={"device": "cpu", "max_bin": 255,"num_threads": 1})
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    raise e

# Colonnes attendues par le modèle
expected_columns = model.feature_name()

# Colonnes manquantes avec valeurs par défaut
default_columns = {
    'FLAG_OWN_REALTY': 0, 'REGION_POPULATION_RELATIVE': 0.0, 'DAYS_BIRTH': 0,
    'DAYS_EMPLOYED': 0, 'DAYS_REGISTRATION': 0, 'DAYS_ID_PUBLISH': 0, 'OWN_CAR_AGE': 0,
    'FLAG_MOBIL': 1, 'FLAG_EMP_PHONE': 0, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1,
    'FLAG_PHONE': 0, 'FLAG_EMAIL': 0, 'CNT_FAM_MEMBERS': 1, 'REGION_RATING_CLIENT': 1,
    'REGION_RATING_CLIENT_W_CITY': 1, 'HOUR_APPR_PROCESS_START': 10, 'REG_REGION_NOT_LIVE_REGION': 0,
    'REG_REGION_NOT_WORK_REGION': 0, 'LIVE_REGION_NOT_WORK_REGION': 0,
    'REG_CITY_NOT_LIVE_CITY': 0, 'REG_CITY_NOT_WORK_CITY': 0, 'LIVE_CITY_NOT_WORK_CITY': 0,
    'EXT_SOURCE_1': 0.0, 'EXT_SOURCE_2': 0.0, 'EXT_SOURCE_3': 0.0,
    'APARTMENTS_AVG': 0.0, 'BASEMENTAREA_AVG': 0.0, 'YEARS_BEGINEXPLUATATION_AVG': 0.0,
    'YEARS_BUILD_AVG': 0.0, 'COMMONAREA_AVG': 0.0, 'ELEVATORS_AVG': 0.0,
    'ENTRANCES_AVG': 0.0, 'FLOORSMAX_AVG': 0.0, 'FLOORSMIN_AVG': 0.0,
    'LANDAREA_AVG': 0.0, 'LIVINGAPARTMENTS_AVG': 0.0, 'LIVINGAREA_AVG': 0.0,
    'NONLIVINGAPARTMENTS_AVG': 0.0, 'NONLIVINGAREA_AVG': 0.0,
    'DAYS_EMPLOYED_PERC': 0.0, 'INCOME_CREDIT_PERC': 0.0, 'INCOME_PER_PERSON': 0.0,
    'ANNUITY_INCOME_PERC': 0.0, 'PAYMENT_RATE': 0.0
}

@app.route('/', methods=['GET'])
def index():
    return "API en ligne. Utilisez /predict pour les prédictions.", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Charger le modèle dans la fonction
        model = lgb.Booster(model_file=MODEL_PATH)

        # Lecture des données reçues
        input_data = request.get_json()
        logging.info(f"Données reçues : {input_data}")

        # Préparation des données
        df = pd.DataFrame(input_data)
        for col, default_value in default_columns.items():
            if col not in df.columns:
                df[col] = default_value
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Prédiction
        predictions = model.predict(df)

        # Libérer la mémoire après chaque requête
        del model, df
        gc.collect()

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Lancement de l'application sur le port {port}")
    app.run(host='0.0.0.0', port=port)
