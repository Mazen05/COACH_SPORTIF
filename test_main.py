
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_program():
    payload = {
        "age": 22,
        "sex": "M",
        "poids": 70,
        "taille": 180,
        "objectif": "endurance",
        "historique_sportif": "débutant"
    }
    response = client.post("/predict_program", json=payload)
    assert response.status_code == 200
    assert "programme_recommande" in response.json()
    assert "details" in response.json()

def test_predict_perf():
    payload = {
        "nb_squats": 10,
        "nb_bench_press": 10,
        "heures_sommeil": 8,
        "qualite_nutrition": 80
    }
    response = client.post("/predict_perf", json=payload)
    assert response.status_code == 200
    assert "progression_estimee" in response.json()
