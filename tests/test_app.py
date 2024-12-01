import pytest
from flask import json
import os
import sys
from app import app

# Ajoutez un chemin absolu vers le dossier backend
sys.path.insert(0, '/Users/Nelly/Desktop/projet7/backend')

@pytest.fixture
def client():
    """Créer un client de test pour l'application Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test de la route principale."""
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "API en ligne"

def test_predict_client(client):
    """Test de la prédiction pour un client existant."""
    data = {"SK_ID_CURR": 100001}
    response = client.post('/predict_client', json=data)
    assert response.status_code == 200
    result = response.get_json()
    assert "prediction" in result
    assert "shap_values" in result

def test_get_client_ids(client):
    """Test de récupération des IDs des clients."""
    response = client.get('/get_client_ids')
    assert response.status_code == 200
    result = response.get_json()
    assert "client_ids" in result

def test_get_next_client_id(client):
    """Test de récupération du prochain ID client."""
    response = client.get('/get_next_client_id')
    assert response.status_code == 200
    result = response.get_json()
    assert "next_id" in result
