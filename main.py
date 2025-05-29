from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

# CORS pour le front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques s'ils existent
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration des templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class UserProfile(BaseModel):
    age: int
    sexe: str
    poids: int
    taille: int
    objectif: str
    historique_sportif: str


class PerformanceInput(BaseModel):
    nb_squats: float
    nb_bench_press: float
    heures_sommeil: float
    qualité_nutrition: float


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

    programmes = {
        0: {
            "label": "cardio",
            "exercices": ["Tapis de course", "Vélo", "Burpees", "Jumping jacks", "Montées de genoux"]
        },
        1: {
            "label": "HIIT",
            "exercices": ["Squats sautés", "Pompes", "Mountain climbers", "Burpees", "Lunges"]
        },
        2: {
            "label": "musculation",
            "exercices": ["Squat", "Développé couché", "Soulevé de terre", "Tractions", "Rowing"]
        }
    }

    return programmes.get(pred_code, {"label": "Inconnu", "exercices": []})


@app.post("/predict_performance")
def predict_performance(data: PerformanceInput):
    df = pd.DataFrame([data.dict()])
    model = joblib.load("model_perf.pkl")
    prediction = float(model.predict(df)[0])
    return {"progression_estimee": round(prediction, 2)}
