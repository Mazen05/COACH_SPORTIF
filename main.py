from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_program():
    payload = {
        "age": 22,
        "sexe": "M",  # ⚠️ bien mettre "sexe" (et pas "sex")
        "poids": 70,
        "taille": 180,
        "objectif": "endurance",
        "historique_sportif": "débutant"
    }
    response = client.post("/predict_program", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "label" in result
    assert "exercices" in result
    assert isinstance(result["exercices"], list)

def test_predict_performance():
    payload = {
        "nb_squats": 10,
        "nb_bench_press": 10,
        "heures_sommeil": 8,
        "qualité_nutrition": 80
    }
    response = client.post("/predict_performance", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "progression_estimee" in result
    assert isinstance(result["progression_estimee"], float)
