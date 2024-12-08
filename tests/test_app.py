import pytest
from app import app  # Importez votre application Flask

@pytest.fixture
def client():
    """Configurer un client Flask pour les tests."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test pour vérifier si la route principale fonctionne."""
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "API en ligne"

def test_check_app_module():
    """Vérifie le module Flask utilisé."""
    print("Chemin du module Flask :", app)

def test_model_is_loaded():
    """Test pour vérifier que le modèle est bien chargé."""
    from app import model
    assert model is not None, "Le modèle n'a pas été chargé."
    assert hasattr(model, "predict_proba"), "Le modèle ne semble pas avoir de méthode 'predict_proba'."

def test_clients_data_is_loaded():
    """Test pour vérifier que les données clients sont bien chargées."""
    from app import clients_data  # Importez clients_data depuis app.py
    assert clients_data is not None, "Les données clients n'ont pas été chargées."
    assert not clients_data.empty, "Les données clients sont vides."
    assert "SK_ID_CURR" in clients_data.columns, "La colonne 'SK_ID_CURR' est absente des données clients."


