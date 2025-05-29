from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd

# Schémas de données
class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: int
    taille: int
    objectif: str
    historique_sportif: str

class PerformanceInput(BaseModel):
    nb_squats: int
    nb_bench_press: int
    heures_sommeil: float
    qualité_nutrition: float

def define_routes(app: FastAPI):

    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/predict_program")
    def predict_program(data: UserProfile):
        df = pd.DataFrame([data.dict()])

        encodage = {
            "sexe": {"M": 0, "F": 1},
            "objectif": {"perte de poids": 0, "prise de muscle": 1, "endurance": 2},
            "historique_sportif": {"débutant": 0, "intermédiaire": 1, "avancé": 2}
        }
        for col, mapping in encodage.items():
            df[col] = df[col].map(mapping)

        model = joblib.load("model_reco.pkl")
        pred_code = int(model.predict(df)[0])

        programme_labels = {
            0: ("cardio", ["Tapis de course", "Vélo", "Burpees", "Jumping jacks", "Montées de genoux"]),
            1: ("HIIT", ["Pompes", "Squats", "Gainage", "Burpees", "Fentes"]),
            2: ("musculation", ["Développé couché", "Squat barre", "Soulevé de terre", "Rowing", "Tractions"])
        }
        label, exercices = programme_labels.get(pred_code, ("Inconnu", []))
        return {"programme_recommandé": label, "details": {"exercices": exercices}}

    @app.post("/predict_performance")
    def predict_performance(data: PerformanceInput):
        df = pd.DataFrame([data.dict()])
        model = joblib.load("model_perf.pkl")
        progression = float(model.predict(df)[0])
        return {"progression_estimee": round(progression, 2)}
