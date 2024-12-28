import os
import logging
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import shap

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Initialiser une application Flask
app = Flask(__name__)
CORS(app)

# Activer les logs
logging.basicConfig(level=logging.INFO)

# Chemins des fichiers nécessaires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_lgb_no.pkl")
CLIENTS_DATA_PATH = os.path.join(BASE_DIR, "clients_data.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "selected_features.txt")

# Vérifications et chargements initiaux
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Modèle ou fichier des features introuvable.")

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Charger les features nécessaires
with open(FEATURES_PATH, "r") as f:
    required_features = f.read().strip().split(",")

# Charger les données clients
if os.path.exists(CLIENTS_DATA_PATH):
    clients_data = pd.read_csv(CLIENTS_DATA_PATH)
    logging.info(f"Fichier clients_data.csv chargé avec succès. Nombre de clients : {len(clients_data)}")
else:
    clients_data = pd.DataFrame()
    logging.warning("Le fichier clients_data.csv est introuvable ou vide.")

@app.route("/", methods=["GET"])
def index():
    """Endpoint pour afficher saisir et afficher prediction a partir d'un formulaire"""
    html_form = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Prédiction crédit</title>
        <script>
            async function fetchPrediction() {
                const id = document.getElementById("clientId").value;
                const resultDiv = document.getElementById("result");
                resultDiv.textContent = "Chargement...";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ SK_ID_CURR: id })
                    });

                    const data = await response.json();
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3>Résultat :</h3>
                            <p>ID Client : ${data.SK_ID_CURR}</p>
                            <p>Probabilité : ${data.probability_of_default}</p>
                            <p>Décision : ${data.decision}</p>
                        `;
                    } else {
                        resultDiv.textContent = `Erreur : ${data.error}`;
                    }
                } catch (error) {
                    resultDiv.textContent = `Erreur réseau : ${error.message}`;
                }
            }
        </script>
    </head>
    <body>
        <h1>Prédiction de Défaut de Paiement</h1>
        <form onsubmit="event.preventDefault(); fetchPrediction();">
            <label>Entrez l'ID client :</label>
            <input type="number" id="clientId" required>
            <button type="submit">Envoyer</button>
        </form>
        <div id="result"></div>
    </body>
    </html>
    """
    return html_form

@app.route("/get_client_ids", methods=["GET"])
def get_client_ids():
    """Récupérer la liste des IDs clients disponibles"""
    if clients_data.empty:
        return jsonify({"client_ids": []}), 200
    return jsonify({"client_ids": clients_data["SK_ID_CURR"].tolist()}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Faire une prédiction pour un client donné"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))

        # Trouver les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr]
        if client_data.empty:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Préparer les données pour la prédiction
        data_for_prediction = client_data[required_features]
        logging.info(f"Données prêtes pour la prédiction :\n{data_for_prediction}")

        # Prédiction avec le modèle
        predictions = model.predict_proba(data_for_prediction)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive
        logging.info(f"Probabilité de défaut de paiement : {probability_of_default}")

        # Décision basée sur le seuil
        decision = "Crédit refusé" if probability_of_default > 0.09 else "Crédit accepté"
        
        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Informations descriptives du client
        client_info = client_data.iloc[0].to_dict()

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default":  round(probability_of_default, 4),
            "shap_values": shap_values.tolist(),
            "feature_names": required_features,
            "client_info": client_info,
            "decision": decision        
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_global_importance", methods=["GET"])
def get_global_importance():
    """Calculer les importances globales des caractéristiques"""
    try:
        if clients_data.empty:
            return jsonify({"error": "Les données clients sont vides ou indisponibles."}), 404

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(clients_data[required_features])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        global_importances = pd.DataFrame({
            "Feature": required_features,
            "Global Importance": abs(shap_values).mean(axis=0)
        }).sort_values(by="Global Importance", ascending=False)

        return jsonify({
            "status": "success",
            "global_importances": global_importances.to_dict(orient="records")
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors du calcul des importances globales : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_with_custom_values", methods=["POST"])
def predict_with_custom_values():
    """Faire une prédiction avec des valeurs modifiées par l'utilisateur"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        sk_id_curr = int(data.get("SK_ID_CURR"))

        # Trouver les données du client
        client_data = clients_data[clients_data["SK_ID_CURR"] == sk_id_curr].copy()
        if client_data.empty:
            return jsonify({"error": f"Client {sk_id_curr} introuvable."}), 404

        # Mise à jour des valeurs si elles sont fournies dans la requête
        for key, value in data.items():
            if key in client_data.columns and value is not None:
                client_data[key] = value

        # Préparer les données pour la prédiction
        data_for_prediction = client_data[required_features]
        logging.info(f"Données prêtes pour la prédiction avec valeurs personnalisées :\n{data_for_prediction}")

        # Prédiction avec le modèle
        predictions = model.predict_proba(data_for_prediction)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive
        logging.info(f"Probabilité de défaut de paiement avec valeurs personnalisées : {probability_of_default}")

        # Décision basée sur le seuil
        decision = "Crédit refusé" if probability_of_default > 0.09 else "Crédit accepté"
        
        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_for_prediction)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Informations descriptives du client
        client_info = client_data.iloc[0].to_dict()

        # Retourner la réponse
        return jsonify({
            "SK_ID_CURR": sk_id_curr,
            "probability_of_default": round(probability_of_default, 4),
            "shap_values": shap_values.tolist(),
            "feature_names": required_features,
            "client_info": client_info,
            "decision": decision            
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction avec valeurs personnalisées : {e}")
        return jsonify({"error": str(e)}), 500

# Chemin vers le fichier contenant la liste d'IDs disponibles
ID_FILE_PATH = "current_id.txt"

def read_available_ids():
    """Lire la liste d'IDs disponibles à partir du fichier."""
    if os.path.exists(ID_FILE_PATH):
        with open(ID_FILE_PATH, "r") as f:
            ids = f.read().strip().split(",")
            return [int(id) for id in ids if id.strip().isdigit()]
    return []

def write_available_ids(ids):
    """Écrire la liste mise à jour d'IDs disponibles dans le fichier."""
    with open(ID_FILE_PATH, "w") as f:
        f.write(",".join(map(str, ids)))

@app.route("/get_next_client_id", methods=["GET"])
def get_next_client_id():
    """Renvoie un ID disponible pour un nouveau client."""
    available_ids = read_available_ids()
    
    if not available_ids:
        return jsonify({"error": "Aucun ID disponible dans la liste."}), 404

    # Attribuer le premier ID disponible
    next_id = available_ids.pop(0)
    
    # Mettre à jour le fichier avec les IDs restants
    write_available_ids(available_ids)
    
    return jsonify({"next_id": next_id}), 200

@app.route("/predict_new_client", methods=["POST"])
def predict_new_client():
    """Faire une prédiction pour un nouveau client avec des valeurs par défaut et médianes"""
    try:
        # Récupérer les données envoyées
        data = request.get_json()

        # Créer une ligne avec des valeurs par défaut
        default_client = {col: 0 for col in required_features}

        # Remplacer les colonnes avec les données fournies
        for key, value in data.items():
            if key in default_client:
                default_client[key] = value

        # Convertir en DataFrame
        new_client_df = pd.DataFrame([default_client])

        # Remplacer les valeurs par défaut avec les médianes des données clients existantes
        if not clients_data.empty:
            for col in new_client_df.columns:
                if new_client_df[col].iloc[0] == 0:  # Vérifier si la valeur est par défaut (0)
                    if col in clients_data.columns:
                        median_value = clients_data[col].median()
                        new_client_df[col] = median_value
                    else:
                        logging.warning(f"Colonne {col} non trouvée dans les données existantes.")

        # Prédiction avec le modèle
        predictions = model.predict_proba(new_client_df)
        probability_of_default = predictions[0][1]  # Probabilité pour la classe positive

        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(new_client_df)

        # Gestion des SHAP values pour plusieurs classes
        if len(shap_values) > 1:
            shap_values = shap_values[1]  # SHAP values pour la classe positive
        else:
            shap_values = shap_values[0]

        # Retourner la réponse
        return jsonify({
            "probability_of_default": probability_of_default,
            "shap_values": shap_values.tolist(),
            "feature_names": required_features
        }), 200

    except Exception as e:
        logging.error(f"Erreur lors de la prédiction pour un nouveau client : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
