from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_program():
    response = client.post("/predict_program", data={
        "age": 25,
        "sexe": "M",
        "poids": 75,
        "taille": 180,
        "objectif": "Endurance",
        "historique_sportif": "Débutant"
    })
    assert response.status_code == 200
    assert "🏋️‍♂️" in response.text or "Programme non reconnu" in response.text

def test_predict_performance():
    response = client.post("/predict_performance", data={
        "nb_squats": 15,
        "nb_bench_press": 10,
        "heures_sommeil": 8,
        "qualité_nutrition": 80
    })
    assert response.status_code == 200
    assert "📈 Progression estimée" in response.text
