from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import os

app = Flask(__name__)

# Définir le chemin vers le modèle
model_path = os.path.join(os.path.dirname(__file__), "lightgbm_model_final.txt")
model = None  # Modèle initialisé à None, chargé à la demande

@app.route('/')
def home():
    """Route principale pour vérifier que l'API fonctionne."""
    return jsonify({"message": "API is running", "status": "success"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour effectuer des prédictions."""
    global model
    try:
        # Charger le modèle si ce n'est pas déjà fait
        if model is None:
            model = lgb.Booster(model_file=model_path)

        # Vérifier si les données envoyées sont au format JSON
        if not request.is_json:
            return jsonify({"error": "Les données doivent être au format JSON."}), 400

        # Charger les données en JSON
        data = request.get_json()
        df = pd.DataFrame(data)

        # Vérifier les colonnes attendues par le modèle
        expected_columns = model.feature_name()
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Ajouter les colonnes manquantes avec une valeur par défaut

        # Préserver uniquement les colonnes nécessaires
        df = df[expected_columns]

        # Faire les prédictions
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        return jsonify({"error": f"Une erreur s'est produite : {str(e)}"}), 500

if __name__ == '__main__':
    # Utilise le port défini par Render ou par défaut 8000
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

