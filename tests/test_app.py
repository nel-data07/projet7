import pytest
from app import app  # Importez votre application Flask

@pytest.fixture
def client():
    """Configurer un client Flask pour les tests."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_no_missing_values_in_clients_data():
    """Test pour vérifier qu'il n'y a pas de valeurs manquantes dans les colonnes nécessaires."""
    from app import clients_data, required_columns
    for col in required_columns:
        assert col in clients_data.columns, f"La colonne '{col}' est absente des données clients."
        assert clients_data[col].notnull().all(), f"La colonne '{col}' contient des valeurs manquantes."

def test_features_are_loaded():
    """Test pour vérifier que les colonnes de features sont bien chargées."""
    from app import required_columns
    assert required_columns is not None, "Les colonnes nécessaires n'ont pas été chargées."
    assert len(required_columns) > 0, "La liste des colonnes nécessaires est vide."
    assert isinstance(required_columns, list), "Les colonnes nécessaires ne sont pas au format liste."

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


