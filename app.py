import os
import lightgbm as lgb
from flask import Flask, request, jsonify

app = Flask(__name__)

# Définir le chemin vers le fichier modèle
model_path = os.path.join(os.path.dirname(__file__), "lightgbm_model_final.txt")

try:
    # Charger le modèle LightGBM
    model = lgb.Booster(model_file=model_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle LightGBM : {str(e)}")

@app.route("/")
def home():
    return jsonify({"message": "L'API fonctionne !"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Vérifiez si les données sont envoyées au format JSON
        if not request.is_json:
            return jsonify({"error": "Les données doivent être au format JSON."}), 400

        # Récupérez les données
        data = request.get_json()

        # Vérifiez que c'est une liste de dictionnaires
        if not isinstance(data, list):
            return jsonify({"error": "Les données doivent être une liste de dictionnaires."}), 400

        # Convertir en DataFrame
        import pandas as pd
        df = pd.DataFrame(data)

        # Faire une prédiction
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

