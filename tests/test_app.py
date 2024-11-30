import pytest
import joblib
from flask import json
import os

# Définir le chemin absolu vers le dossier backend et les fichiers nécessaires
BACKEND_PATH = "/Users/Nelly/Desktop/projet7/backend"
MODEL_PATH = os.path.join(BACKEND_PATH, "best_model_lgb_bal.pkl")
FEATURES_PATH = os.path.join(BACKEND_PATH, "selected_features.txt")

# Débogage : imprimer les chemins utilisés
print(f"Chemin utilisé pour le fichier modèle : {MODEL_PATH}")
print(f"Chemin utilisé pour le fichier des colonnes : {FEATURES_PATH}")

# Vérifiez que les fichiers existent
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Fichier des colonnes introuvable : {FEATURES_PATH}")

# Charger les colonnes nécessaires
with open(FEATURES_PATH, "r") as f:
    REQUIRED_COLUMNS = f.read().strip().split(",")
print(f"Colonnes chargées : {REQUIRED_COLUMNS}")

# Charger le modèle
print("Chargement du modèle...")
MODEL = joblib.load(MODEL_PATH)
print("Modèle chargé avec succès.")

# Charger l'application Flask
from backend.app import app

@pytest.fixture
def client():
    """Créer un client de test pour l'application Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_model_loaded():
    """Test pour vérifier que le modèle est correctement chargé."""
    assert MODEL is not None, "Le modèle n'a pas été chargé correctement."
    assert hasattr(MODEL, "predict_proba"), "Le modèle chargé n'a pas la méthode 'predict_proba'."

def test_features_loaded():
    """Test pour vérifier que les colonnes nécessaires sont chargées."""
    assert len(REQUIRED_COLUMNS) > 0, "Les colonnes nécessaires n'ont pas été correctement chargées."
    assert "CODE_GENDER" in REQUIRED_COLUMNS, "'CODE_GENDER' devrait être une colonne nécessaire."

def test_home(client):
    """Test de la route principale."""
    response = client.get('/')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["message"] == "API en ligne"
    assert json_data["status"] == "success"

def test_predict_valid(client):
    """Test de la prédiction avec des données valides."""
    data = [
        {
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "CNT_CHILDREN": 2,
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0
        }
    ]
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_json = response.get_json()
    assert "predictions" in response_json, "Les prédictions ne sont pas présentes dans la réponse."
    assert isinstance(response_json["predictions"], list), "Les prédictions devraient être une liste."

def test_predict_missing_columns(client):
    """Test de la prédiction avec des colonnes manquantes."""
    data = [{"CODE_GENDER": 1}]
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_json = response.get_json()
    assert "predictions" in response_json, "Les prédictions ne sont pas présentes dans la réponse."
    assert isinstance(response_json["predictions"], list), "Les prédictions devraient être une liste."

def test_predict_empty_request(client):
    """Test de la prédiction avec une requête vide."""
    response = client.post('/predict', json=[])
    assert response.status_code == 400, "Le serveur devrait retourner un code 400 pour une requête vide."
    response_json = response.get_json()
    assert "error" in response_json, "Un message d'erreur devrait être retourné pour une requête vide."

def test_predict_null_values(client):
    """Test de la prédiction avec des valeurs nulles."""
    data = [
        {
            "CODE_GENDER": None,
            "FLAG_OWN_CAR": 0,
            "CNT_CHILDREN": None,
            "AMT_INCOME_TOTAL": None,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": None,
            "AMT_GOODS_PRICE": 450000.0
        }
    ]
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    response_json = response.get_json()
    assert "predictions" in response_json, "Les prédictions ne sont pas présentes dans la réponse."
    assert isinstance(response_json["predictions"], list), "Les prédictions devraient être une liste."
