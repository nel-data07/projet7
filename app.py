from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd

# Charger le modèle LightGBM
model = lgb.Booster(model_file="lightgbm_model_final.txt")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées via POST
        data = request.json  # Attendez-vous à un JSON au format [{"feature1": val1, "feature2": val2, ...}]
        df = pd.DataFrame(data)

        # Prédictions
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

