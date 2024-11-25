import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test de la route principale."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.get_json() == {"message": "API is running", "status": 
"success"}

def test_predict_valid(client):
    """Test de la prédiction avec des données valides."""
    data = [
        {
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "FLAG_OWN_REALTY": 1,
            "CNT_CHILDREN": 2,
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "REGION_POPULATION_RELATIVE": 0.01,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "DAYS_REGISTRATION": -4000,
            "DAYS_ID_PUBLISH": -1000,
            "OWN_CAR_AGE": 5.0,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 0,
            "FLAG_EMAIL": 0,
            "CNT_FAM_MEMBERS": 3,
            "REGION_RATING_CLIENT": 2,
            "REGION_RATING_CLIENT_W_CITY": 2,
            "HOUR_APPR_PROCESS_START": 10,
            "REG_REGION_NOT_LIVE_REGION": 0,
            "REG_REGION_NOT_WORK_REGION": 0,
            "LIVE_REGION_NOT_WORK_REGION": 0,
            "REG_CITY_NOT_LIVE_CITY": 0,
            "REG_CITY_NOT_WORK_CITY": 0,
            "LIVE_CITY_NOT_WORK_CITY": 0,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.7,
            "EXT_SOURCE_3": 0.8
        }
    ]
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert "predictions" in response.get_json()

def test_predict_missing_columns(client):
    """Test de la prédiction avec des colonnes manquantes."""
    data = [{"CODE_GENDER": 1}]
    response = client.post('/predict', json=data)
    assert response.status_code == 400

