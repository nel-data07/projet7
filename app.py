from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Charger le modèle LightGBM
model_path = os.path.join(os.path.dirname(__file__), "lightgbm_model_final.txt")
try:
    model = lgb.Booster(model_file=model_path)
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle LightGBM : {str(e)}")
    raise RuntimeError(f"Erreur lors du chargement du modèle LightGBM : {str(e)}")

@app.route('/')
def home():
    """Route principale pour vérifier que l'API fonctionne."""
    logging.info("Requête reçue sur la route '/'")
    return jsonify({"message": "API is running", "status": "success"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour effectuer des prédictions."""
    try:
        logging.info("Requête reçue sur la route '/predict'")
        
        # Vérifier si les données envoyées sont au format JSON
        if not request.is_json:
            logging.error("Les données reçues ne sont pas au format JSON.")
            return jsonify({"error": "Les données doivent être au format JSON."}), 400

        # Charger les données JSON
        data = request.get_json()
        logging.info(f"Données reçues : {data}")

        # Convertir les données en DataFrame
        df = pd.DataFrame(data)

        # Vérifier les colonnes attendues par le modèle
        expected_columns = model.feature_name()
        logging.info(f"Colonnes attendues par le modèle : {expected_columns}")

        # Identifier les colonnes manquantes
        missing_cols = [col for col in expected_columns if col not in df.columns]

        if missing_cols:
            logging.warning(f"Colonnes manquantes détectées : {missing_cols}")
            # Remplir les colonnes manquantes avec des valeurs par défaut
            default_values = {col: 0 for col in missing_cols}  # Remplacer 0 par des valeurs pertinentes si nécessaire
            missing_df = pd.DataFrame(default_values, index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        # Réorganiser les colonnes pour correspondre à l'ordre attendu
        df = df[expected_columns]

        # Faire les prédictions
        predictions = model.predict(df)
        logging.info(f"Prédictions générées : {predictions}")
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        logging.error(f"Erreur lors du traitement de la requête '/predict' : {str(e)}")
        return jsonify({"error": f"Une erreur s'est produite : {str(e)}"}), 500

if __name__ == '__main__':
    # Utilise le port défini par Render ou par défaut 8000
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"Lancement de l'application sur le port {port}")
    app.run(host='0.0.0.0', port=port)
