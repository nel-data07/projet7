from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import os

# Définir le chemin vers le modèle
model_path = os.path.join(os.path.dirname(__file__), "lightgbm_model_final.txt")

# Charger le modèle LightGBM
try:
    model = lgb.Booster(model_file=model_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle LightGBM : {str(e)}")

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    """Route principale pour vérifier que l'API fonctionne."""
    return jsonify({"message": "API is running", "status": "success"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour effectuer des prédictions."""
    try:
        # Vérifiez si les données sont au format JSON
        if not request.is_json:
            return jsonify({"error": "Les données doivent être au format JSON."}), 400

        # Charger les données envoyées
        data = request.get_json()
        df = pd.DataFrame(data)

        # Vérifier les colonnes attendues
        expected_columns = model.feature_name()
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            # Ajouter les colonnes manquantes avec des valeurs par défaut (par exemple 0)
            for col in missing_columns:
                df[col] = 0

        # Vérifier si des colonnes superflues sont présentes
        df = df[expected_columns]

        # Faire les prédictions
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        # Retourner une erreur en cas de problème
        return jsonify({"error": f"Une erreur s'est produite : {str(e)}"}), 500

if __name__ == '__main__':
    # Lire le port assigné par Render, par défaut 8000 pour local
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)


