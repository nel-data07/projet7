import pytest
import os
import sys

# Ajoutez la racine du projet au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Importez votre application Flask

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test pour vérifier si la route principale fonctionne"""
    response = client.get('/')
    assert response.status_code == 200

def test_predict(client):
    """Test pour l'endpoint de prédiction"""
    response = client.post('/predict', json={"feature1": 1.5, "feature2": 
2.5})
    assert response.status_code == 200
    assert "predictions" in response.get_json()

