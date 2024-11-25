from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd

# Charger le modèle LightGBM
try:
    model = lgb.Booster(model_file="lightgbm_model_final.txt")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle LightGBM : 
{str(e)}")

app = Flask(__name__)

@app.route('/')
def home():
    """Route principale pour vérifier que l'API fonctionne."""
    return {
        "message": "API is running",
        "status": "success",
        "version": "1.0.0"
    }, 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si des données ont été envoyées
        if not request.is_json:
            return jsonify({"error": "Les données doivent être au format 
JSON."}), 400

        # Récupérer les données JSON
        data = request.get_json()

        # Vérifier si le JSON est une liste ou un dictionnaire
        if not isinstance(data, list):
            return jsonify({"error": "Les données doivent être une liste 
de dictionnaires."}), 400

        # Convertir les données en DataFrame
        df = pd.DataFrame(data)

        # Vérifier si le DataFrame est vide
        if df.empty:
            return jsonify({"error": "Les données envoyées sont vides."}), 
400

        # Faire les prédictions
        predictions = model.predict(df)

        # Retourner les prédictions sous forme de liste
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        # Gérer les erreurs et retourner une réponse claire
        return jsonify({"error": f"Une erreur s'est produite : 
{str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

