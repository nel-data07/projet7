{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "fb89c71e-95b7-4bc6-8d00-21bcd2fb125d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pytest\n",
    "from app import app  # Importez votre application Flask\n",
    "\n",
    "@pytest.fixture\n",
    "def client():\n",
    "    app.config['TESTING'] = True\n",
    "    with app.test_client() as client:\n",
    "        yield client\n",
    "\n",
    "def test_index(client):\n",
    "    \"\"\"Test pour vérifier si la route principale fonctionne\"\"\"\n",
    "    response = client.get('/')\n",
    "    assert response.status_code == 200\n",
    "\n",
    "def test_predict(client):\n",
    "    \"\"\"Test pour l'endpoint de prédiction\"\"\"\n",
    "    response = client.post('/predict', json={\"feature1\": 1.5, \"feature2\": 2.5})\n",
    "    assert response.status_code == 200\n",
    "    assert \"predictions\" in response.get_json()"
   ]
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
