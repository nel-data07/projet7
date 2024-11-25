import os
from flask import Flask, jsonify, request
import lightgbm as lgb
import pandas as pd

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "LightGBM_Model", 
"model.pkl")

try:
    model = lgb.Booster(model_file=model_path)
    print("Modèle LightGBM chargé avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle LightGBM : 
{str(e)}")

# Initialiser Flask
app = Flask(__name__)

# Route principale
@app.route('/')
def home():
    return jsonify({"message": "API is running", "status": "success"}), 
200

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si des données ont été envoyées
        if not request.is_json:
            return jsonify({"error": "Les données doivent être au format 
JSON."}), 400

        # Récupérer les données
        data = request.get_json()
        df = pd.DataFrame(data)

        # Vérifier les colonnes manquantes
        expected_columns = model.feature_name()
        missing_columns = [col for col in expected_columns if col not in 
df.columns]
        if missing_columns:
            return jsonify({"error": f"Colonnes manquantes : 
{missing_columns}"}), 400

        # Effectuer la prédiction
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": f"Une erreur s'est produite : 
{str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

