
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialisation de l'application
app = FastAPI()

# Chargement des modèles
model_reco = joblib.load("model_reco.pkl")
model_perf = joblib.load("model_perf.pkl")

# Définition des schémas de données pour Pydantic

class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: float
    taille: float
    objectif: str
    historique_sportif: str

class PerformanceInput(BaseModel):
    nb_squats: int
    nb_bench_press: int
    heures_sommeil: float
    qualité_nutrition: int

# Endpoint de prédiction de programme d'entraînement
@app.post("/predict_program")
def predict_program(data: UserProfile):
    df = pd.DataFrame([data.dict()])
    # Encodage à adapter selon ton pipeline réel
    encodage = {
        "sexe": {"H": 0, "F": 1},
        "objectif": {"perte de poids": 0, "prise de muscle": 1, "endurance": 2},
        "historique_sportif": {"débutant": 0, "intermédiaire": 1, "avancé": 2}
    }
    for col, mapping in encodage.items():
        df[col] = df[col].map(mapping)
    prediction = model_reco.predict(df)[0]
    mapping_inverse = {0: "cardio", 1: "musculation", 2: "HIIT"}
    return {"programme_recommandé": mapping_inverse.get(prediction, "inconnu")}

# Endpoint de prédiction de performance future
@app.post("/predict_progress")
def predict_progress(data: PerformanceInput):
    df = pd.DataFrame([data.dict()])
    prediction = model_perf.predict(df)[0]
    return {"progression_estimée": round(prediction, 2)}
