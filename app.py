{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "88e9829e-fc03-459e-930e-8de794816ef1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Le rapport de Data Drift a été généré : 'data_drift_report.html'\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from evidently.report import Report\n",
    "from evidently.metric_preset import DataDriftPreset\n",
    "\n",
    "# Charger les datasets\n",
    "train_data = pd.read_csv(\"/Users/Nelly/Desktop/projet 7/data/application_train.csv\")\n",
    "test_data = pd.read_csv(\"/Users/Nelly/Desktop/projet 7/data/application_test.csv\")\n",
    "\n",
    "# Créer un rapport de data drift\n",
    "data_drift_report = Report(metrics=[DataDriftPreset()])\n",
    "\n",
    "# Sélectionner uniquement les colonnes nécessaires pour l'analyse\n",
    "common_columns = list(set(train_data.columns).intersection(set(test_data.columns)))\n",
    "train_data = train_data[common_columns]\n",
    "test_data = test_data[common_columns]\n",
    "\n",
    "# Générer le rapport\n",
    "data_drift_report.run(reference_data=train_data, current_data=test_data)\n",
    "\n",
    "# Exporter le rapport en HTML\n",
    "data_drift_report.save_html(\"data_drift_report.html\")\n",
    "\n",
    "print(\"Le rapport de Data Drift a été généré : 'data_drift_report.html'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ba0b62ce-5366-46ef-b17c-096e9e7449d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://192.168.1.101:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import lightgbm as lgb\n",
    "import pandas as pd\n",
    "\n",
    "# Charger le modèle LightGBM\n",
    "model = lgb.Booster(model_file=\"lightgbm_model_final.txt\")\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        # Récupérer les données envoyées via POST\n",
    "        data = request.json  # Attendez-vous à un JSON au format [{\"feature1\": val1, \"feature2\": val2, ...}]\n",
    "        df = pd.DataFrame(data)\n",
    "        \n",
    "        # Prédictions\n",
    "        predictions = model.predict(df)\n",
    "        return jsonify({\"predictions\": predictions.tolist()})\n",
    "    except Exception as e:\n",
    "        return jsonify({\"error\": str(e)}), 400\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b86cf7e-2d09-4897-a04a-efc429065691",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
